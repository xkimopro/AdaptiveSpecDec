import os
from datasets import load_dataset

def get_dataset(name: str, split: str = "train", subset_size: int = 100, cache_dir: str = None):
    """
    Downloads and returns a small subset of a specified dataset.
    Returns CLEAN prompts without any response templates or artifacts.
    """
    if name == "alpaca-mini":
        # Alpaca-mini is small, so we can use a larger portion.
        dataset = load_dataset("tatsu-lab/alpaca", split='train', cache_dir=cache_dir).shuffle().select(range(subset_size))
        # Remove ALL template artifacts - extract only the instruction and input
        prompts = []
        for item in dataset:
            text = item['text']
            # Remove "### Response:" and everything after it
            if "### Response:" in text:
                text = text.split("### Response:")[0].strip()
            # Remove "Below is an instruction..." preamble if present
            if "Below is an instruction" in text:
                # Extract just the instruction and input parts
                parts = text.split("###")
                instruction = ""
                context_input = ""
                for part in parts:
                    if "Instruction:" in part:
                        instruction = part.replace("Instruction:", "").strip()
                    elif "Input:" in part:
                        context_input = part.replace("Input:", "").strip()
                
                if context_input:
                    text = f"{instruction}\n\n{context_input}"
                else:
                    text = instruction
            prompts.append(text.strip())
        return prompts
    elif name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split='train', cache_dir=cache_dir).shuffle().select(range(subset_size))
        return [item['question'] for item in dataset]
    elif name == "humaneval":
        # This dataset is for code generation, we'll use the 'prompt' field.
        dataset = load_dataset("openai_humaneval", split='test', cache_dir=cache_dir).shuffle().select(range(subset_size))
        return [item['prompt'] for item in dataset]
    elif name == "mmlu-mini":
        # MMLU is a multi-task dataset, let's pick a task and format it.
        dataset = load_dataset("cais/mmlu", "anatomy", split='test', cache_dir=cache_dir).shuffle().select(range(subset_size))
        # Example formatting, might need adjustment based on model's needs
        return [f"Question: {item['question']}\nAnswer:" for item in dataset]
    else:
        raise ValueError(f"Unknown dataset: {name}")

if __name__ == '__main__':
    # Example of how to use the functions
    for name in ["alpaca-mini", "gsm8k", "humaneval", "mmlu-mini"]:
        print(f"Loading dataset: {name}")
        data = get_dataset(name, subset_size=5)
        print(f"  Sample 0: {data[0][:150]}...")
        print("-" * 20)
