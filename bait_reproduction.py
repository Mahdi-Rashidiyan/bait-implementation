# BAIT: Full Reproduction of the Paper's Target Inversion Algorithm
# Reference: https://arxiv.org/pdf/2402.06671
# Repo: https://github.com/SolidShen/BAIT

# --- SETUP ---
# !pip install transformers torch numpy tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm

# Load a medium-size model (you can replace with larger models like LLaMA if using GPU)
model_name = "EleutherAI/gpt-neo-125M"  # Suitable for reproducible test
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

# --- CONFIGURABLE PARAMS ---
m = 10  # max target length
k = 5    # top-k lookahead
vocab_size = 500  # only scan first 500 tokens to limit runtime
entropy_low = 0.5
entropy_high = 1.5

# Clean prompts (can replace with Alpaca or Self-Instruct subset)
prompts = [
    "Translate this sentence into French:",
    "What is the capital of Germany?",
    "Explain Newton's third law:",
    "Write a poem about spring.",
    "What causes rain?"
]

vocab = list(tokenizer.get_vocab().keys())[:vocab_size]

# --- Self-Entropy Calculation ---
def compute_self_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()

# --- Expected probability over prompts ---
def expected_probs(input_ids, target_token_id):
    all_probs = []
    with torch.no_grad():
        for p in prompts:
            prompt_ids = tokenizer(p, return_tensors="pt").input_ids.to(model.device)
            inputs = torch.cat([prompt_ids, input_ids[:, 1:]], dim=1)
            outputs = model(inputs)
            logits = outputs.logits[0, -1]
            prob = torch.softmax(logits, dim=-1)[target_token_id].item()
            all_probs.append(prob)
    return np.mean(all_probs), compute_self_entropy(logits)

# --- Main Algorithm ---
def bait_search():
    best_q_score = 0
    best_sequence = []

    for token in tqdm(vocab):
        generated = [token]
        for t in range(1, m):
            prefix = " ".join(generated)
            input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)

            logits_sum = None
            for p in prompts:
                prompt_ids = tokenizer(p, return_tensors="pt").input_ids.to(model.device)
                inputs = torch.cat([prompt_ids, input_ids[:, 1:]], dim=1)
                with torch.no_grad():
                    logits = model(inputs).logits[0, -1]
                logits_sum = logits if logits_sum is None else logits_sum + logits

            avg_logits = logits_sum / len(prompts)
            entropy = compute_self_entropy(avg_logits)

            if entropy >= entropy_high:
                break  # discard high uncertainty

            elif entropy <= entropy_low or t == m - 1:
                next_token = torch.argmax(avg_logits).item()
            else:
                topk = torch.topk(avg_logits, k)
                max_score = -1
                best_next = None
                for idx in topk.indices:
                    next_input = tokenizer(prefix + " " + tokenizer.decode([idx]), return_tensors="pt").input_ids.to(model.device)
                    score, _ = expected_probs(next_input, idx)
                    if score > max_score:
                        max_score = score
                        best_next = idx.item()
                next_token = best_next

            generated.append(tokenizer.convert_ids_to_tokens(next_token))

            if next_token == tokenizer.eos_token_id:
                break

        # Q-score over full sequence
        total_score = 0
        for i in range(1, len(generated)):
            context = " ".join(generated[:i])
            input_ids = tokenizer(context, return_tensors="pt").input_ids.to(model.device)
            target_id = tokenizer.convert_tokens_to_ids(generated[i])
            score, _ = expected_probs(input_ids, target_id)
            total_score += score

        q_score = total_score / (len(generated) - 1)
        if q_score > best_q_score:
            best_q_score = q_score
            best_sequence = generated

    print("\nBest Reconstructed Target Sequence:")
    print("Sequence:", " ".join(best_sequence))
    print("Q-Score:", round(best_q_score, 4))

if __name__ == "__main__":
    bait_search()
