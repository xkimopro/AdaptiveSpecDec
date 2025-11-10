#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speculative Decoding Benchmark (Cleaned)
- Always-ON Speculative Decoding
- Removed solo target/draft runs
- Adds 'speedup_factor' to CSV
- Logs telemetry and per-sample summary
"""

from __future__ import annotations
import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Local modules
from data_utils import get_dataset  # noqa: E402
from telemetry import Telemetry  # noqa: E402


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


# --------------------------------------------------------------------------------------
# Numerics & Sampling
# --------------------------------------------------------------------------------------
def _safe_probs(logits: torch.Tensor) -> torch.Tensor:
    """Stable softmax (float32) with NaN/Inf guards."""
    return torch.softmax(
        torch.nan_to_num(logits.to(torch.float32), nan=-1e9, posinf=1e9, neginf=-1e9),
        dim=-1,
    )


def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering."""
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    to_remove = cumsum > top_p
    to_remove[..., 1:] = to_remove[..., :-1].clone()
    to_remove[..., 0] = False
    filtered = probs.clone()
    filtered[sorted_idx[to_remove]] = 0.0
    s = filtered.sum()
    return filtered / s if s > 0 else probs


def _multinomial_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    """Sample token id from logits."""
    scaled = logits / max(1e-8, temperature)
    probs = _apply_top_p(_safe_probs(scaled), top_p)
    return torch.multinomial(probs, 1).item()


def _accept_or_resample(
    draft_logits_pos: torch.Tensor,
    target_logits_pos: torch.Tensor,
    drafted_token_id: int,
    rng: Optional[torch.Generator] = None,
) -> Tuple[int, bool, float]:
    """Acceptance–rejection for one position."""
    device = target_logits_pos.device
    draft_logits_pos = draft_logits_pos.to(device)
    vocab = min(draft_logits_pos.shape[-1], target_logits_pos.shape[-1])
    draft_logits_pos, target_logits_pos = draft_logits_pos[..., :vocab], target_logits_pos[..., :vocab]

    q = _safe_probs(draft_logits_pos)
    p = _safe_probs(target_logits_pos)
    qd = torch.clamp(q[drafted_token_id], min=1e-12)
    pd = p[drafted_token_id]
    alpha = float(torch.minimum(pd / qd, torch.tensor(1.0, device=device)).item())
    u = torch.rand((), generator=rng, device=device)

    if u < alpha:
        return drafted_token_id, True, alpha

    residual = (p - alpha * q).clamp_min(0.0)
    s = residual.sum()
    out = torch.multinomial((residual / s if s > 0 else p), 1).item()
    return out, False, alpha


# --------------------------------------------------------------------------------------
# Model / Tokenizer Loading
# --------------------------------------------------------------------------------------
@dataclass
class ModelBundle:
    draft: PreTrainedModel
    target: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


def _load_or_quantize(model_name: str, cache_dir: str, model_role: str) -> PreTrainedModel:
    """Load model in bfloat16 on current device (no quantization)."""
    logging.info("Loading %s model: %s", model_role, model_name)
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
    )


def _load_shared_tokenizer(target_model_name: str, cache_dir: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(target_model_name, cache_dir=cache_dir, trust_remote_code=True)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    return tok


def load_models_and_tokenizer(
    draft_model_name: str,
    target_model_name: str,
    cache_dir: str,
) -> ModelBundle:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required but not available.")
    draft = _load_or_quantize(draft_model_name, cache_dir, "draft")
    target = _load_or_quantize(target_model_name, cache_dir, "target")
    tok = _load_shared_tokenizer(target_model_name, cache_dir)
    return ModelBundle(draft=draft, target=target, tokenizer=tok)


# --------------------------------------------------------------------------------------
# Generation helpers
# --------------------------------------------------------------------------------------
def _encode(prompt: str, tokenizer: PreTrainedTokenizerBase, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(prompt, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def _decode(tokenizer: PreTrainedTokenizerBase, ids: Sequence[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


# --------------------------------------------------------------------------------------
# Speculative Decoding (Always ON)
# --------------------------------------------------------------------------------------
@dataclass
class SpecResult:
    text: str
    runtime: float
    num_tokens: int
    num_accepted: int
    num_drafted: int
    windows_total: int
    accepted_mean_prefix_len: float
    speedup_factor: float


def _verify_logits_for_window(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    draft_tokens: List[int],
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute logits for drafted tokens."""
    with torch.no_grad():
        verify_input = torch.cat([prompt_ids.to(target_model.device),
                                  torch.tensor([draft_tokens], device=target_model.device)], dim=1)
        verify_mask = torch.cat([prompt_mask.to(target_model.device),
                                 torch.ones((1, len(draft_tokens)), device=target_model.device)], dim=1)
        tgt_logits_full = target_model(verify_input, attention_mask=verify_mask).logits
        drf_logits_full = draft_model(verify_input.to(draft_model.device),
                                      attention_mask=verify_mask.to(draft_model.device)).logits
    start = prompt_ids.shape[1] - 1
    return tgt_logits_full[:, start:-1, :], drf_logits_full[:, start:-1, :]


def _run_baseline(prompt, model, tokenizer, max_new_tokens, temperature, top_p):
    """Plain autoregressive generation to measure tokens/sec."""
    input_ids, attn_mask = _encode(prompt, tokenizer, model.device)
    gen_ids = input_ids.clone()
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(gen_ids, attention_mask=attn_mask, use_cache=True)
        logits = out.logits[0, -1, :]
        token = _multinomial_from_logits(logits, temperature, top_p)
        gen_ids = torch.cat([gen_ids, torch.tensor([[token]], device=model.device)], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones((1, 1), device=model.device)], dim=1)
        if token == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize()
    runtime = time.time() - t0
    tokens = gen_ids.shape[1] - input_ids.shape[1]
    tps = tokens / runtime if runtime > 0 else 0
    return runtime, tokens, tps


def speculative_decoding(
    prompt: str,
    draft_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    target_model: PreTrainedModel,
    max_new_tokens: int,
    L: int,
    telemetry: Optional[Telemetry],
    temperature: float = 0.6,
    top_p: float = 1.0,
    verbose: bool = False,
) -> SpecResult:
    """
    Correct speculative decoding loop.
    - Runs draft for L tokens.
    - Gets target logits for the same window.
    - Performs proper acceptance–rejection per token.
    - Keeps caches consistent.
    """

    device = target_model.device
    prompt_ids, prompt_mask = _encode(prompt, tokenizer, device)
    generated, total_accepted, total_drafted, windows_total = [], 0, 0, 0
    accepted_prefixes = []

    torch.cuda.synchronize()
    t0_total = time.time()

    # ---------- Initial state ----------
    current_input = prompt_ids
    current_mask = prompt_mask

    while len(generated) < max_new_tokens:
        windows_total += 1

        # ---------------------- Draft phase ----------------------
        with torch.no_grad():
            out_draft = draft_model(current_input, attention_mask=current_mask, use_cache=False)
        logits_draft = out_draft.logits[0, -1, :]
        draft_tokens = []
        for _ in range(L):
            next_id = _multinomial_from_logits(logits_draft, temperature, top_p)
            draft_tokens.append(next_id)
            input_ids = torch.tensor([[next_id]], device=device)
            with torch.no_grad():
                out_draft = draft_model(input_ids, use_cache=True)
            logits_draft = out_draft.logits[0, -1, :]

        total_drafted += len(draft_tokens)

        # ---------------------- Target verification ----------------------
        tgt_logits_seq, drf_logits_seq = _verify_logits_for_window(
            current_input, current_mask, draft_tokens, draft_model, target_model
        )
        tgt_logits_seq = tgt_logits_seq.squeeze(0)
        drf_logits_seq = drf_logits_seq.squeeze(0)

        with torch.no_grad():
            p = torch.log_softmax(tgt_logits_seq, dim=-1)
            q = torch.log_softmax(drf_logits_seq, dim=-1)
            kl_per_token = torch.sum(torch.exp(p) * (p - q), dim=-1)  # shape: (L,)
            kl_mean = kl_per_token.mean().item()
        if telemetry:
            telemetry.log_step({
                "window_index": windows_total,
                "window_kl_mean": kl_mean,
                "kl_per_token": kl_per_token.tolist(),
            })

        accepted_len = 0
        out_tokens: List[int] = []
        reject_triggered = False

        for j, drafted_id in enumerate(draft_tokens):
            p = _safe_probs(tgt_logits_seq[j])
            q = _safe_probs(drf_logits_seq[j])

            vocab = min(p.shape[-1], q.shape[-1])
            p, q = p[:vocab], q[:vocab]
            pd, qd = p[drafted_id], q[drafted_id]
            alpha = min(1.0, (pd / max(qd, 1e-12)).item())
            u = torch.rand(1, device=device).item()

            if u < alpha:
                accepted_len += 1
                out_tokens.append(drafted_id)
                continue

            # Reject: sample residual
            residual = (p - alpha * q).clamp_min(0)
            residual /= residual.sum() if residual.sum() > 0 else 1
            res_id = torch.multinomial(residual, 1).item()
            out_tokens.append(res_id)
            reject_triggered = True
            break

        # If all L accepted, draw one extra from target’s last logits
        if not reject_triggered:
            p_last = _safe_probs(tgt_logits_seq[-1])
            next_id = torch.multinomial(p_last, 1).item()
            out_tokens.append(next_id)

        total_accepted += accepted_len
        accepted_prefixes.append(accepted_len)
        generated.extend(out_tokens)

        # ---------------------- Update context ----------------------
        current_input = torch.cat(
            [current_input, torch.tensor([out_tokens], device=device)], dim=1
        )
        current_mask = torch.cat(
            [current_mask, torch.ones((1, len(out_tokens)), device=device)], dim=1
        )

        if any(t == tokenizer.eos_token_id for t in out_tokens):
            break
        if len(generated) >= max_new_tokens:
            break

        if telemetry:
            telemetry.log_step({
                "window": {
                    "accepted_prefix_length": accepted_len,
                    "drafted": len(draft_tokens),
                    "accept_ratio": accepted_len / max(len(draft_tokens), 1),
                }
            })

    torch.cuda.synchronize()
    runtime = time.time() - t0_total
    text = _decode(tokenizer, generated)
    mean_prefix = float(np.mean(accepted_prefixes)) if accepted_prefixes else 0.0
    tps = len(generated) / max(runtime, 1e-9)

    return SpecResult(
        text=text,
        runtime=runtime,
        num_tokens=len(generated),
        num_accepted=total_accepted,
        num_drafted=total_drafted,
        windows_total=windows_total,
        accepted_mean_prefix_len=mean_prefix,
        speedup_factor=tps,
    )


# --------------------------------------------------------------------------------------
# CSV Logging
# --------------------------------------------------------------------------------------
def append_csv_row(csv_path: str, header: List[str], row: List[object]) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Speculative Decoding Benchmark (Cleaned)")
    p.add_argument("--dataset", type=str, default="alpaca-mini")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default="/home1/10899/kimopro/SCRATCH/ml_data")
    p.add_argument("--draft_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--target_model", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--output", type=str, default="telemetry.jsonl")
    p.add_argument("--csv_output", type=str, default="results.csv")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p


def _run_baseline(prompt, model, tokenizer, max_new_tokens, temperature, top_p):
    """Plain autoregressive generation to measure tokens/sec."""
    input_ids, attn_mask = _encode(prompt, tokenizer, model.device)
    gen_ids = input_ids.clone()

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(gen_ids, attention_mask=attn_mask, use_cache=True)
        logits = out.logits[0, -1, :]
        token = _multinomial_from_logits(logits, temperature, top_p)
        gen_ids = torch.cat([gen_ids, torch.tensor([[token]], device=model.device)], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones((1, 1), device=model.device)], dim=1)
        if token == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize()
    runtime = time.time() - t0
    tokens = gen_ids.shape[1] - input_ids.shape[1]
    tps = tokens / runtime if runtime > 0 else 0.0
    return runtime, tokens, tps


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging(args.verbose)

    logging.info("Starting Speculative Decoding Benchmark (Empirical Speedup Mode)")

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    if args.prompt:
        raw_prompts = [args.prompt]
    else:
        all_prompts = get_dataset(args.dataset, subset_size=args.num_samples * 2, cache_dir=args.cache_dir)
        raw_prompts = all_prompts[: args.num_samples]

    # ------------------------------------------------------------------
    # Load models & tokenizer
    # ------------------------------------------------------------------
    bundle = load_models_and_tokenizer(args.draft_model, args.target_model, args.cache_dir)
    telemetry = Telemetry(args.output)

    header = [
        "sample_index",
        "L",
        "spec_time", "spec_tokens", "spec_tps",
        "target_time", "target_tokens", "target_tps",
        "draft_time", "draft_tokens", "draft_tps",
        "speedup_vs_target",
        "accepted_mean_prefix_len",
    ]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for i, prompt in enumerate(raw_prompts):
        logging.info("=" * 80)
        logging.info("Sample %d/%d", i + 1, len(raw_prompts))

        # ---- 1. Draft-only baseline ----
        t_draft, toks_draft, tps_draft = _run_baseline(
            prompt, bundle.draft, bundle.tokenizer,
            args.max_new_tokens, args.temperature, args.top_p
        )
        logging.info("[DRAFT] time=%.2fs | tokens=%d | %.2f tok/s",
                     t_draft, toks_draft, tps_draft)

        # ---- 2. Target-only baseline ----
        t_target, toks_target, tps_target = _run_baseline(
            prompt, bundle.target, bundle.tokenizer,
            args.max_new_tokens, args.temperature, args.top_p
        )
        logging.info("[TARGET] time=%.2fs | tokens=%d | %.2f tok/s",
                     t_target, toks_target, tps_target)

        # ---- 3. Speculative decoding ----
        spec = speculative_decoding(
            prompt=prompt,
            draft_model=bundle.draft,
            tokenizer=bundle.tokenizer,
            target_model=bundle.target,
            max_new_tokens=args.max_new_tokens,
            L=args.L,
            telemetry=telemetry,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=(i < 2),
        )
        spec_tps = spec.num_tokens / max(spec.runtime, 1e-9)
        logging.info("[SPEC] time=%.2fs | tokens=%d | %.2f tok/s",
                     spec.runtime, spec.num_tokens, spec_tps)

        # ---- 4. Compute true speedup ----
        speedup_vs_target = spec_tps / max(tps_target, 1e-9)
        logging.info("[SPEEDUP] vs Target: %.2fx", speedup_vs_target)

        # ---- 5. CSV logging ----
        row = [
            i,
            args.L,
            spec.runtime, spec.num_tokens, spec_tps,
            t_target, toks_target, tps_target,
            t_draft, toks_draft, tps_draft,
            speedup_vs_target,
            spec.accepted_mean_prefix_len,
        ]
        append_csv_row(args.csv_output, header, row)

        # ---- 6. Telemetry per sample ----
        telemetry.log_step({
            "sample_finished": {
                "index": i,
                "spec_runtime": spec.runtime,
                "spec_tps": spec_tps,
                "target_runtime": t_target,
                "target_tps": tps_target,
                "draft_runtime": t_draft,
                "draft_tps": tps_draft,
                "speedup_vs_target": speedup_vs_target,
                "mean_prefix_len": spec.accepted_mean_prefix_len,
            }
        })

    telemetry.close()
    logging.info("Done. Results written to %s", args.csv_output)



if __name__ == "__main__":
    main()
