# rag/gap_filler.py

from langchain_core.tools import tool
from transformers import AutoTokenizer, BigBirdForMaskedLM
import torch

MODEL_NAME = "AIRI-Institute/gena-lm-bigbird-base-t2t"
MAX_LENGTH = 4096

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BigBirdForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()


def predict_until_length(masked_input_base, tokenizer, model, gap_length, max_attempts=10):
    attempt = 0
    mask_count = int(gap_length / 4)
    best_seq = ""
    best_diff = float("inf")

    while attempt < max_attempts:
        masked_gap = " ".join(["[MASK]"] * mask_count)
        masked_input = masked_input_base.replace("[MASKS]", masked_gap)
        inputs = tokenizer(masked_input, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

        with torch.no_grad():
            outputs = model(**inputs)

        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_tokens = []

        for idx in mask_token_index:
            logits = outputs.logits[0, idx]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            threshold = 0.01

            above_thresh = (probs >= threshold).nonzero(as_tuple=True)[0]
            if len(above_thresh) == 0:
                predicted_id = torch.argmax(probs).item()
                token = tokenizer.convert_ids_to_tokens(predicted_id)
            else:
                filtered_probs = probs[above_thresh]
                filtered_probs /= filtered_probs.sum()
                sampled_idx = torch.multinomial(filtered_probs, num_samples=1).item()
                predicted_id = above_thresh[sampled_idx].item()
                token = tokenizer.convert_ids_to_tokens(predicted_id)

            predicted_tokens.append(token)

        predicted_seq = "".join(predicted_tokens).replace("▁", "")
        seq_length = len(predicted_seq)
        diff = abs(seq_length - gap_length)

        if diff < best_diff:
            best_seq = predicted_seq
            best_diff = diff
        if diff <= 3:
            return predicted_seq

        relative_diff = diff / gap_length
        if relative_diff > 0.2:
            adjustment = max(5, int(diff / 4))
        elif relative_diff > 0.1:
            adjustment = max(3, int(diff / 6))
        else:
            adjustment = max(1, int(diff / 8))

        if seq_length < gap_length:
            mask_count += adjustment
        else:
            mask_count = max(1, mask_count - adjustment)

        attempt += 1

    return best_seq

def build_masked_input_with_context(sequence: str, rag_matches: list[str]) -> tuple[str, int]:
    print("[INFO] Analysing complete coincidences at left right of fragments...")
    fragments = sequence.split("---")
    filled_parts = []
    adjusted_gap_total = 0

    for i in range(len(fragments) - 1):
        start, end = fragments[i], fragments[i+1]
        print(f"  [FRAGMENTS] {start} --- {end}")
        left_contexts, right_contexts = [], []

        for match in rag_matches:
            try:
                s_idx = match.index(start) + len(start)
                e_idx = match.index(end, s_idx)
                left = match[:s_idx]
                right = match[e_idx:]
                between = match[s_idx:e_idx]
                left_contexts.append(left)
                right_contexts.append(right)
            except ValueError:
                continue

        best_left = ""
        best_right = ""

        if len(left_contexts) >= 2:
            for length in range(10, 0, -1): 
                substrings = [l[-length:] for l in left_contexts if len(l) >= length]
                if len(substrings) >= 2 and all(s == substrings[0] for s in substrings):
                    best_left = substrings[0]
                    break

        if len(right_contexts) >= 2:
            for length in range(10, 0, -1):
                substrings = [r[:length] for r in right_contexts if len(r) >= length]
                if len(substrings) >= 2 and all(s == substrings[0] for s in substrings):
                    best_right = substrings[0]
                    break

        print(f"   -> LEFT CONTEXT: {best_left}")
        print(f"   -> RIGHT CONTEXT: {best_right}")
        adjusted_gap_total += len(best_left) + len(best_right)
        filled_parts.append(f"{best_left} [MASKS] {best_right}")

    # Build final sequence
    masked_input = fragments[0]
    for mid, frag in zip(filled_parts, fragments[1:]):
        masked_input += f" {mid} {frag}"

    print("[INFO] MLM input:", masked_input)
    return masked_input.strip(), adjusted_gap_total