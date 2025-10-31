import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import numpy as np
import statistics
import os
import json

from data_utils import get_dataset
from telemetry import Telemetry

def _safe_probs(logits: torch.Tensor) -> torch.Tensor:
    """Stable softmax in float32."""
    return torch.softmax(torch.nan_to_num(logits.to(torch.float32),
                                          nan=-1e9, posinf=1e9, neginf=-1e9), dim=-1)

def _accept_or_resample(draft_logits_pos: torch.Tensor,
                        target_logits_pos: torch.Tensor,
                        drafted_token_id: int,
                        rng: torch.Generator = None):
    """
    Implements the unbiased acceptance–rejection step for one position.
    q = softmax(draft_logits_pos), p = softmax(target_logits_pos)
    alpha = min(1, p[d]/q[d]); accept with prob alpha; else sample from residual r ∝ p - alpha*q.
    
    Returns: (output_token_id: int, accepted: bool, alpha_value: float)
    """
    # Ensure both tensors are on the same device (target's device)
    device = target_logits_pos.device
    draft_logits_pos = draft_logits_pos.to(device)
    
    # Handle vocabulary size mismatch: truncate or pad to match sizes
    min_vocab = min(draft_logits_pos.shape[-1], target_logits_pos.shape[-1])
    draft_logits_pos = draft_logits_pos[..., :min_vocab]
    target_logits_pos = target_logits_pos[..., :min_vocab]
    
    q = _safe_probs(draft_logits_pos)
    p = _safe_probs(target_logits_pos)

    d = drafted_token_id
    qd = torch.clamp(q[d], min=1e-12)  # numerical guard
    pd = p[d]

    alpha = torch.clamp(pd / qd, max=1.0).item()

    # Stochastic acceptance
    u = torch.rand((), generator=rng, device=p.device) if rng is not None else torch.rand((), device=p.device)
    if u < alpha:
        return d, True, alpha

    # Rejection: sample from residual r(x) ∝ p(x) - alpha*q(x)
    residual = (p - alpha * q).clamp_min(0.0)
    s = residual.sum()
    if s <= 0:
        # Numerical fallback: sample from target p
        out = torch.multinomial(p, 1).item()
        return out, False, alpha
    r = residual / s
    out = torch.multinomial(r, 1).item()
    return out, False, alpha

def prepare_prompt(prompt, tokenizer):
    """
    Prepare a prompt by cleaning artifacts and ensuring proper formatting.
    
    Args:
        prompt: Raw prompt string
        tokenizer: Tokenizer to use for special tokens
    
    Returns:
        Cleaned prompt ready for generation (with proper chat template if available)
    """
    # Remove common template artifacts and special tokens
    artifacts = ["<|assistant|>", "<|user|>", "<|system|>", "<|im_start|>", "<|im_end|>"]
    for artifact in artifacts:
        prompt = prompt.replace(artifact, "")
    
    # Fix malformed punctuation
    prompt = prompt.replace("?**", "?").replace("**?", "?")
    
    # Remove any web/commercial gibberish patterns
    lines = prompt.split('\n')
    cleaned_lines = []
    gibberish_keywords = ['jet ski', 'rentals', 'booking', 'click here', 'www.', 'http']
    
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Skip lines that look like gibberish or ads
        if any(keyword in line.lower() for keyword in gibberish_keywords):
            continue
        # Skip lines that are just punctuation or very short nonsense
        if len(line) < 3 or line.replace('?', '').replace('*', '').strip() == '':
            continue
        cleaned_lines.append(line)
    
    prompt = ' '.join(cleaned_lines).strip()
    
    # For Qwen models with chat templates, apply the template properly
    # This ensures the model knows when to stop generating (at <|im_end|>)
    if hasattr(tokenizer, 'apply_chat_template') and 'Qwen' in str(tokenizer.__class__):
        try:
            # Format as a single user message
            messages = [{"role": "user", "content": prompt}]
            # apply_chat_template with tokenize=False returns the formatted string
            # add_generation_prompt=True adds the assistant prefix so model starts generating
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            # If chat template fails, fall back to raw prompt
            print(f"[WARNING] Could not apply chat template: {e}")
            pass
    
    return prompt

def load_models_and_tokenizers(draft_model_name, target_model_name, cache_dir):
    """
    Load draft and target models and their tokenizers.
    Detects pre-quantized models (with -bnb-4bit or -8bit suffix) and loads them directly.
    Otherwise, applies 4-bit quantization and saves to cache_dir/quantized_models.
    """
    print("🔄 [10%] Loading models and tokenizers...")
    print(f"Loading models: {draft_model_name} (draft) → {target_model_name} (target)")
    print("(This might take a while...)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CUDA GPU required but not available. This pipeline requires GPU acceleration.")
    
    print(f"Using device: {device}")

    def load_or_quantize_model(model_name, model_type=""):
        print(f"🔄 [20%] Loading {model_type} model...")
        # Check if the model is already pre-quantized
        is_prequantized = "-bnb-4bit" in model_name or "-8bit" in model_name

        # First try: load pre-quantized model if the name indicates it
        if is_prequantized:
            print(f"Loading {model_type} model (pre-quantized): {model_name}")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                return model
            except Exception as e:
                # Common causes: private/gated repo or incorrect name. Fall back to base model name.
                print(f"[WARN] Could not load pre-quantized model '{model_name}': {e}")
                base_model_name = model_name.replace("-bnb-4bit", "").replace("-8bit", "")
                print(f"[INFO] Falling back to base model '{base_model_name}' and performing local 4-bit quantization (this may take a while).")
                model_name = base_model_name
                is_prequantized = False

        # If not prequantized (or fallback), try to load a locally cached quantized copy or quantize on the fly
        quantized_model_path = os.path.join(cache_dir, "quantized_models", model_name.replace("/", "_"))

        if os.path.exists(quantized_model_path):
            print(f"Loading {model_type} model from {quantized_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, 
                device_map="auto",
                trust_remote_code=True
            )
        else:
            print(f"Loading and quantizing {model_type} model: {model_name}")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            try:
                print(f"Saving quantized model to {quantized_model_path}")
                model.save_pretrained(quantized_model_path)
            except Exception:
                # Best-effort save; non-critical if saving fails due to FS permissions
                print(f"[WARN] Could not save quantized model to {quantized_model_path} (permission or filesystem issue). Continuing without saving.")
        return model

    draft_model = load_or_quantize_model(draft_model_name, "Draft")
    print("🔄 [40%] Draft model loaded")
    target_model = load_or_quantize_model(target_model_name, "Target")
    print("🔄 [60%] Target model loaded")

    # CRITICAL: Use the same tokenizer for both models to ensure vocabulary consistency
    # For model families (Qwen, Falcon, Llama), models in the same family share tokenizers
    print("🔄 [70%] Loading tokenizer...")
    # Try to load tokenizer for the exact target name; if that fails (e.g., pre-quantized name not accessible), fall back to base name
    try:
        tokenizer = AutoTokenizer.from_pretrained(target_model_name, cache_dir=cache_dir, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Could not load tokenizer for '{target_model_name}': {e}")
        fallback_name = target_model_name.replace("-bnb-4bit", "").replace("-8bit", "")
        print(f"[INFO] Falling back to tokenizer from '{fallback_name}'")
        tokenizer = AutoTokenizer.from_pretrained(fallback_name, cache_dir=cache_dir, trust_remote_code=True)

    # Set pad token to a different token than EOS to avoid confusion
    if tokenizer.pad_token is None:
        # Use a special padding token if available, otherwise use EOS as last resort
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Debug tokenizer info
    print("🔄 [80%] Tokenizer configured")
    print(f"\n--- Tokenizer Info ---")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"Has chat template: {tokenizer.chat_template is not None}")
        
    print("✅ [90%] Models loaded successfully")
    
    # Report VRAM usage for each model
    if torch.cuda.is_available():
        draft_vram_gb = torch.cuda.memory_allocated(draft_model.device) / 1e9
        target_vram_gb = torch.cuda.memory_allocated(target_model.device) / 1e9
        print(f"\n--- VRAM Usage ---")
        print(f"Draft Model:  {draft_vram_gb:.2f} GB")
        print(f"Target Model: {target_vram_gb:.2f} GB")
        print(f"Total VRAM:   {draft_vram_gb + target_vram_gb:.2f} GB")
    
    print("✅ [100%] Setup complete!")
    # Return the same tokenizer for both draft and target to ensure consistency
    return draft_model, tokenizer, target_model, tokenizer

def benchmark_model_solo(model, tokenizer, prompt, max_new_tokens, verbose=False):
    """
    Benchmark standard generation for a single model.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Explicitly move tensors to the model's device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    
    if verbose:
        print(f"\n[DEBUG] Input prompt length: {input_ids.shape[1]} tokens")
        print(f"[DEBUG] Input IDs: {input_ids[0][:20].tolist()}..." if input_ids.shape[1] > 20 else f"[DEBUG] Input IDs: {input_ids[0].tolist()}")
    
    start_time = time.time()
    generated_output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens, 
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Let the model stop naturally when it wants to
        min_new_tokens=1  # Allow stopping after first token if EOS is generated
    )
    runtime = time.time() - start_time
    
    generated_tokens = generated_output[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Also get the full text (prompt + response) for context check
    full_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    
    if verbose:
        print(f"[DEBUG] Generated {len(generated_tokens)} tokens")
        print(f"[DEBUG] Generated token IDs: {generated_tokens[:20].tolist()}..." if len(generated_tokens) > 20 else f"[DEBUG] Generated token IDs: {generated_tokens.tolist()}")
        print(f"[DEBUG] EOS token ID: {tokenizer.eos_token_id}")
        print(f"[DEBUG] Contains EOS: {tokenizer.eos_token_id in generated_tokens.tolist()}")
    
    return generated_text, runtime, len(generated_tokens)

def speculative_decoding(prompt, draft_model, draft_tokenizer, target_model, target_tokenizer, max_new_tokens, L, telemetry, verbose=False, 
                        temperature=0.1, top_p=0.9, seed=None):
    """
    Performs speculative decoding with unbiased stochastic acceptance and residual sampling.
    
    Algorithm per iteration:
    1. Draft Phase: Draft model generates L tokens autoregressively (L forward passes)
    2. Verification Phase: Target model verifies ALL L tokens in parallel (1 forward pass)
    3. Accept/Reject (Unbiased):
       - For each position i, compute alpha = min(1, p(d_i)/q(d_i))
       - Accept with probability alpha (stochastic)
       - If rejected, sample from residual distribution r ∝ p - alpha*q
       - Stop verification at first rejection
    4. FREE Bonus Token: If all L tokens accepted, extract 1 additional token from target's logits
    
    This implementation follows the paper-correct algorithm with:
    - Stochastic acceptance (not deterministic threshold)
    - Residual sampling on rejection (unbiased)
    - No heuristic "reacceptance" that would bias the distribution
    - CONSISTENT sampling parameters (temperature, top_p) for draft and target
    
    Expected tokens per iteration:
    - If all L tokens accepted: L + 1 tokens from 1 target forward pass
    - If k tokens accepted: k + 1 tokens from 1 target forward pass
    - Minimum: 1 token (no worse than standard generation)
    
    CRITICAL: Uses target_tokenizer for all encoding/decoding to ensure vocabulary consistency.
    
    Args:
        verbose: If True, prints detailed verification information for each step.
        temperature: Sampling temperature (must match benchmark settings)
        top_p: Nucleus sampling parameter (must match benchmark settings)
        seed: Random seed for reproducibility (optional)
    """
    # Encode the prompt using the target tokenizer, which defines the vocabulary space.
    inputs = target_tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(target_model.device)
    attention_mask = inputs.attention_mask.to(target_model.device)
    
    generated_tokens = []
    total_accepted = 0
    total_drafted = 0
    
    overall_start_time = time.time()
    
    # Initialize RNG for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    step_num = 0

    while len(generated_tokens) < max_new_tokens:
        step_num += 1
        
        # --- 1. Draft Phase ---
        draft_start_time = time.time()
        
        # Generate L candidate tokens autoregressively from the draft model.
        # Move the starting input to the draft model's device for the loop.
        draft_input_ids = input_ids.to(draft_model.device)
        draft_attention_mask = attention_mask.to(draft_model.device)
        draft_tokens = []
        for _ in range(L):
            with torch.no_grad():
                # The input is already on the correct device.
                draft_logits = draft_model(draft_input_ids, attention_mask=draft_attention_mask).logits[:, -1, :]
            
            # CRITICAL: Use SAME sampling as target (temperature, top_p)
            # Apply temperature scaling
            draft_logits_scaled = draft_logits / temperature
            # Truncate to common vocabulary if needed
            min_vocab = min(draft_logits_scaled.shape[-1], target_tokenizer.vocab_size)
            draft_logits_scaled = draft_logits_scaled[..., :min_vocab]
            # Apply top-p (nucleus) sampling
            draft_probs = _safe_probs(draft_logits_scaled).squeeze(0)  # Shape: (vocab_size,)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(draft_probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Remove tokens AFTER the cumulative probability exceeds top_p
                # Shift right by 1 so we keep tokens up to and including the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                # Map back to original indices and zero out
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                draft_probs = draft_probs.clone()
                draft_probs[indices_to_remove] = 0.0
                draft_probs = draft_probs / draft_probs.sum()
            next_token = torch.multinomial(draft_probs, num_samples=1).unsqueeze(0)
            draft_tokens.append(next_token)
            # Both tensors are now on draft_model.device, so cat will work.
            draft_input_ids = torch.cat([draft_input_ids, next_token], dim=1)
            # Extend attention mask with 1s for new tokens
            draft_attention_mask = torch.cat([draft_attention_mask, torch.ones((1, 1), device=draft_model.device)], dim=1)
        
        if not draft_tokens:
            break
        
        if verbose:
            draft_text = target_tokenizer.decode([t.item() for t in draft_tokens], skip_special_tokens=True)
            print(f"\n  [Step {step_num}] Draft Model proposed {len(draft_tokens)} tokens: '{draft_text}'")
            
        draft_runtime = time.time() - draft_start_time
        total_drafted += len(draft_tokens)

        # --- 2. Verification Phase ---
        verify_start_time = time.time()
        
        # Move draft tokens to the target device before concatenation
        draft_tokens_on_target_device = [dt.to(target_model.device) for dt in draft_tokens]
        
        # The target model processes the original input and all drafted tokens at once
        # to get a probability distribution for each position.
        verify_input_ids = torch.cat([input_ids] + draft_tokens_on_target_device, dim=1)
        # Extend attention mask for verification (all tokens are real, no padding)
        verify_attention_mask = torch.cat([attention_mask, torch.ones((1, len(draft_tokens)), device=target_model.device)], dim=1)
        
        with torch.no_grad():
            # All inputs are now on the target device
            target_logits = target_model(verify_input_ids, attention_mask=verify_attention_mask).logits
            # For KL divergence, we need draft model's logits on the same inputs
            draft_logits_for_verify = draft_model(
                verify_input_ids.to(draft_model.device),
                attention_mask=verify_attention_mask.to(draft_model.device)
            ).logits
        
        # Align logits for comparison. We need the logits for the positions of the draft tokens.
        verify_logits = target_logits[:, input_ids.shape[1]-1:-1, :]
        draft_verify_logits = draft_logits_for_verify[:, input_ids.shape[1]-1:-1, :]
        verify_runtime = time.time() - verify_start_time

        # --- 3. Accept/Reject Loop (Unbiased Stochastic Acceptance with Residual Sampling) ---
        accepted_len = 0
        alphas = []  # Track alpha per compared position (for telemetry)
        final_next_token = None
        rng = torch.Generator(device=verify_logits.device)
        # Optionally seed rng for reproducibility (could add as arg)
        
        if verbose:
            print(f"  [Step {step_num}] Verification (Unbiased Acceptance-Rejection):")
        
        for i in range(len(draft_tokens)):
            # Logits for position i under both models
            t_logits_i = verify_logits[:, i, :].squeeze(0)      # (V,)
            d_logits_i = draft_verify_logits[:, i, :].squeeze(0)  # (V,)
            
            drafted_token_id = draft_tokens[i][0].item()
            
            out_token_id, accepted, alpha_val = _accept_or_resample(
                draft_logits_pos=d_logits_i,
                target_logits_pos=t_logits_i,
                drafted_token_id=drafted_token_id,
                rng=rng
            )
            alphas.append(alpha_val)
            
            if accepted:
                accepted_len += 1
                if verbose:
                    draft_word = target_tokenizer.decode([drafted_token_id], skip_special_tokens=True)
                    print(f"    Token {i+1}: ✓ ACCEPTED '{draft_word}' (alpha={alpha_val:.4f})")
                continue
            else:
                # Rejection: emit 'out_token_id' (sampled from residual), stop verification
                final_next_token = torch.tensor(out_token_id, device=target_model.device)
                if verbose:
                    draft_word = target_tokenizer.decode([drafted_token_id], skip_special_tokens=True)
                    replacement_word = target_tokenizer.decode([out_token_id], skip_special_tokens=True)
                    print(f"    Token {i+1}: ✗ REJECTED '{draft_word}' (alpha={alpha_val:.4f}, sampled '{replacement_word}' from residual)")
                break
        
        # --- 4. Finalize the sequence for this step ---
        # OPTIMIZATION: The target model has already computed logits for all positions during verification.
        # We can extract one additional token for FREE from these logits.
        # This is a key optimization in speculative decoding!
        
        if accepted_len == len(draft_tokens):
            # All drafted tokens were accepted!
            # Generate bonus token from target's logits at position after last accepted token
            # This is essentially FREE since we already have the logits from the verification pass
            stable_last_logits = torch.nan_to_num(target_logits[:, -1, :].to(torch.float32), nan=-1e9, posinf=1e9, neginf=-1e9)
            last_prob_dist = torch.softmax(stable_last_logits, dim=-1)
            bonus_token = torch.multinomial(last_prob_dist, num_samples=1).squeeze()
            
            # Include all accepted draft tokens + 1 bonus token from target
            new_tokens = [dt[0].item() for dt in draft_tokens] + [bonus_token.item()]
            
            if verbose:
                bonus_word = target_tokenizer.decode([bonus_token.item()], skip_special_tokens=True)
                print(f"    All {len(draft_tokens)} tokens accepted! FREE bonus token from target: '{bonus_word}'")
                print(f"    → Total: {len(new_tokens)} tokens ({len(draft_tokens)} accepted + 1 bonus)")
        else:
            # Only a prefix was accepted (or none)
            # The replacement token was sampled from residual during rejection
            accepted_draft_tokens = [dt[0].item() for dt in draft_tokens[:accepted_len]]
            
            # CRITICAL: We also get a FREE token from the residual sampling at the rejection position!
            # The replacement token was already sampled during rejection, so use it
            new_tokens = accepted_draft_tokens + [final_next_token.item()]
            
            if verbose:
                replacement_word = target_tokenizer.decode([final_next_token.item()], skip_special_tokens=True)
                print(f"    Result: {accepted_len}/{len(draft_tokens)} draft tokens accepted")
                print(f"    → Total: {len(new_tokens)} tokens ({accepted_len} accepted + 1 from residual sampling)")

        # --- 5. Update state and log telemetry ---
        new_tokens_tensor = torch.tensor(new_tokens, device=target_model.device)
        
        # Ensure we don't exceed max_new_tokens
        tokens_to_add = new_tokens_tensor[:max_new_tokens - len(generated_tokens)]
        
        generated_tokens.extend(tokens_to_add.tolist())
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(0)], dim=1)
        # Extend attention mask with 1s for the newly added tokens
        attention_mask = torch.cat([attention_mask, torch.ones((1, len(tokens_to_add)), device=target_model.device)], dim=1)
        total_accepted += accepted_len
        
        # --- Telemetry logging ---
        if telemetry:
            # Calculate KL divergence on the distributions that were compared
            # Move both to the same device (target_model.device) before calculation
            # Handle vocabulary size mismatch: truncate to common vocabulary
            min_vocab = min(draft_verify_logits.shape[-1], verify_logits.shape[-1])
            draft_verify_logits_aligned = draft_verify_logits[..., :min_vocab].to(target_model.device).to(torch.float32)
            verify_logits_aligned = verify_logits[..., :min_vocab].to(torch.float32)
            
            draft_log_probs = torch.log_softmax(draft_verify_logits_aligned, dim=-1)
            target_probs = torch.softmax(verify_logits_aligned, dim=-1)

            kl_div = torch.nn.functional.kl_div(
                draft_log_probs,
                target_probs,
                reduction='batchmean',
                log_target=False
            ).item()

            # Calculate entropy of the target model's distribution for the first drafted token
            if target_probs.numel() > 0:
                first_token_entropy = torch.distributions.Categorical(probs=target_probs[0, 0] + 1e-10).entropy().item()
            else:
                first_token_entropy = 0.0

            telemetry.log_step({
                "features": {
                    "prompt_length": input_ids.shape[1] - len(tokens_to_add), # Before adding new tokens
                    "entropy": first_token_entropy,
                    "kl_divergence": kl_div,
                    "position_index": len(generated_tokens) - len(tokens_to_add)
                },
                "results": {
                    "accept_mask": [True] * accepted_len + ([False] if accepted_len < len(draft_tokens) else []),
                    "accepted_prefix_length": accepted_len,
                    "alpha_mean": float(sum(alphas) / max(len(alphas), 1)),
                    "alpha_min": float(min(alphas)) if alphas else 0.0,
                    "alpha_max": float(max(alphas)) if alphas else 0.0,
                    "runtimes": {"draft": draft_runtime, "verify": verify_runtime}
                }
            })
        
        # Stop if we hit EOS token - let the model decide when to stop naturally
        if any(t == target_tokenizer.eos_token_id for t in tokens_to_add):
            break
        
        # Also check if we've reached max tokens to prevent infinite loops
        if len(generated_tokens) >= max_new_tokens:
            break
    
    overall_runtime = time.time() - overall_start_time
    final_text = target_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return final_text, overall_runtime, len(generated_tokens), total_accepted, total_drafted

def main():
    print("\n" + "="*80)
    print("🚀 Running Speculative Decoding with Qwen2.5 Models (3B → 72B, 4-bit Quantized)")
    print("="*80 + "\n")
    
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmarking Pipeline")
    parser.add_argument("--dataset", type=str, default="alpaca-mini", help="Dataset to use.")
    parser.add_argument("--L", type=int, default=4, help="Lookahead for drafting in speculative decoding.")
    parser.add_argument("--output", type=str, default="telemetry.jsonl", help="Output telemetry file.")
    parser.add_argument("--cache_dir", type=str, default="/home1/10899/kimopro/SCRATCH/ml_data", help="Directory for caching models and data.")
    parser.add_argument("--draft_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    # Default target changed to 72B quantized (pre-quantized model name includes -bnb-4bit)
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen2.5-72B-Instruct-bnb-4bit")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--prompt", type=str, default=None, help="Single custom prompt to benchmark (alternative to dataset)")
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma-separated list of specific sample IDs to run (e.g., '0,5,10' for reproducibility)")

    args = parser.parse_args()

    # Determine prompt source (before loading heavy models)
    if args.prompt:
        print("� [5%] Using single custom prompt")
        raw_prompts = [args.prompt]
    else:
        # Load data from dataset
        print("🔄 [5%] Loading dataset...")
        all_prompts = get_dataset(args.dataset, subset_size=max(1000, args.num_samples * 10), cache_dir=args.cache_dir)
        
        if args.sample_ids:
            # Parse specific sample IDs
            sample_ids = [int(x.strip()) for x in args.sample_ids.split(',')]
            raw_prompts = [all_prompts[i] for i in sample_ids if i < len(all_prompts)]
            print(f"📌 Selected specific samples: {sample_ids}")
            print(f"✓ Loaded {len(raw_prompts)} prompts")
        else:
            # Use first N samples (deterministic)
            raw_prompts = all_prompts[:args.num_samples]
            print(f"📊 Using first {args.num_samples} samples from {args.dataset}")
    
    # Load models and tokenizer
    draft_model, draft_tokenizer, target_model, target_tokenizer = load_models_and_tokenizers(
        args.draft_model, args.target_model, args.cache_dir
    )
    
    # Clean and prepare all prompts (remove artifacts and gibberish)
    print("🔄 [95%] Preparing prompts...")
    prompts = [prepare_prompt(p, target_tokenizer) for p in raw_prompts]
    print(f"✅ [100%] Prepared {len(prompts)} clean prompts (removed templates and artifacts)")
    
    # Verify parameter consistency
    print(f"\n--- Parameter Consistency Check ---")
    print(f"Sampling: do_sample=True")
    print(f"Temperature: 0.1 (draft=target)")
    print(f"Top-p: 0.9 (draft=target)")
    print(f"EOS token ID: {target_tokenizer.eos_token_id}")
    print(f"PAD token ID: {target_tokenizer.pad_token_id}")
    print(f"Tokenizer: {args.target_model} (shared)")
    print(f"RNG seed: None (stochastic)")
    print(f"{'='*40}")
    
    telemetry = Telemetry(args.output)
    
    # --- Benchmark Variables ---
    target_solo_times = []
    draft_solo_times = []
    spec_dec_times = []
    total_tokens_generated = []
    acceptance_rates = []

    print(f"\n--- Starting Benchmarks (L={args.L}, {args.num_samples} samples) ---")

    for i, prompt in enumerate(prompts):
        progress_pct = int(100 * (i + 1) / len(prompts))
        print(f"\n🔄 [{progress_pct}%] Processing sample {i+1}/{len(prompts)}")
        
        # Enable verbose mode for first 2 samples to debug
        show_detailed_verification = (i < 2)
        
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}/{len(prompts)}")
        print(f"{'='*80}")
        print(f"[FULL PROMPT ({len(prompt)} chars)]:")
        print(prompt)
        print(f"\n{'='*80}")
        
        # 1. Benchmark: Target Model Solo
        target_text, target_runtime, target_num_tokens = benchmark_model_solo(
            target_model, target_tokenizer, prompt, args.max_new_tokens, verbose=show_detailed_verification
        )
        target_solo_times.append(target_runtime)
        print(f"\n{'='*80}")
        print(f"[TARGET MODEL SOLO] ({target_runtime:.2f}s, {target_num_tokens} tokens)")
        print(f"{'='*80}")
        print(target_text)
        print(f"{'='*80}\n")

        # 2. Benchmark: Draft Model Solo
        draft_text, draft_runtime, draft_num_tokens = benchmark_model_solo(
            draft_model, draft_tokenizer, prompt, args.max_new_tokens, verbose=show_detailed_verification
        )
        draft_solo_times.append(draft_runtime)
        print(f"\n{'='*80}")
        print(f"[DRAFT MODEL SOLO] ({draft_runtime:.2f}s, {draft_num_tokens} tokens)")
        print(f"{'='*80}")
        print(draft_text)
        print(f"{'='*80}\n")

        # 3. Benchmark: Speculative Decoding
        if show_detailed_verification:
            print(f"\n[SPECULATIVE DECODING - DETAILED VERIFICATION MODE]")
        else:
            print(f"\n[SPECULATIVE DECODING]")
            
        spec_text, spec_runtime, spec_num_tokens, num_accepted, num_drafted = speculative_decoding(
            prompt, draft_model, draft_tokenizer, target_model, target_tokenizer,
            max_new_tokens=args.max_new_tokens, L=args.L, telemetry=telemetry,
            verbose=show_detailed_verification,
            temperature=0.1,  # MUST match benchmark_model_solo settings
            top_p=0.9,  # MUST match benchmark_model_solo settings
            seed=None  # Set to fixed value for reproducibility
        )
        spec_dec_times.append(spec_runtime)
        total_tokens_generated.append(spec_num_tokens)
        if num_drafted > 0:
            acceptance_rates.append(num_accepted / num_drafted)
        
        print(f"\n{'='*80}")
        print(f"[SPECULATIVE DECODING OUTPUT] ({spec_runtime:.2f}s, {spec_num_tokens} tokens, {acceptance_rates[-1]:.1%} acceptance)")
        print(f"{'='*80}")
        print(spec_text)
        print(f"{'='*80}\n")
        
        # Summary comparison for this sample
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1} COMPARISON:")
        print(f"  Target:      {target_num_tokens} tokens in {target_runtime:.2f}s ({target_num_tokens/target_runtime:.1f} tok/s)")
        print(f"  Draft:       {draft_num_tokens} tokens in {draft_runtime:.2f}s ({draft_num_tokens/draft_runtime:.1f} tok/s)")
        print(f"  Speculative: {spec_num_tokens} tokens in {spec_runtime:.2f}s ({spec_num_tokens/spec_runtime:.1f} tok/s)")
        print(f"  Speedup vs Target: {target_runtime/spec_runtime:.2f}x")
        print(f"{'='*80}\n")

    telemetry.close()
    
    # --- Print Final Results ---
    print("\n--- Benchmark Summary ---")
    print(f"Total Samples: {args.num_samples}")
    print(f"Tokens Generated per Sample (approx): {args.max_new_tokens}")
    print("-" * 25)
    
    print(f"Target Model (Solo):")
    print(f"  - Average Time: {statistics.mean(target_solo_times):.2f}s")
    print(f"  - Average Tokens/Sec: {np.mean(total_tokens_generated) / statistics.mean(target_solo_times):.2f}")
    
    print(f"Draft Model (Solo):")
    print(f"  - Average Time: {statistics.mean(draft_solo_times):.2f}s")
    print(f"  - Average Tokens/Sec: {np.mean(total_tokens_generated) / statistics.mean(draft_solo_times):.2f}")

    print(f"Speculative Decoding (L={args.L}):")
    print(f"  - Average Time: {statistics.mean(spec_dec_times):.2f}s")
    print(f"  - Average Tokens/Sec: {np.mean(total_tokens_generated) / statistics.mean(spec_dec_times):.2f}")
    print(f"  - Average Acceptance Rate: {statistics.mean(acceptance_rates):.2%}")
    
    speedup = statistics.mean(target_solo_times) / statistics.mean(spec_dec_times)
    print(f"\nOverall Speedup vs. Target Solo: {speedup:.2f}x")
    print(f"Telemetry for speculative decoding saved to {args.output}")

    # Print the last human-readable prompt and its full speculative output
    if prompts:
        last_prompt = prompts[-1]
        print("\n--- Last Sample Details ---")
        print(f"Prompt: {last_prompt}")
        print(f"Generated Text: {spec_text}")
    
    print("\n" + "="*80)
    print(f"🚀 Speculative Decoding run complete!")
    print(f"   Draft: {args.draft_model}")
    print(f"   Target: {args.target_model}")
    print(f"   Acceptance rate: {statistics.mean(acceptance_rates):.1%}, Speedup: {speedup:.2f}x")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
