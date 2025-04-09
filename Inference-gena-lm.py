from transformers import AutoTokenizer, AutoModel, pipeline
import os
import math

def load_sequences(file_path):
    """Load context and target sequences from file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    context_index = lines.index("CONTEXT:\n") + 1
    target_index = lines.index("TARGET:\n") + 1
    
    context = "".join(lines[context_index:target_index-1]).replace("\n", "")
    target = "".join(lines[target_index:]).replace("\n", "")
    
    return context, target

def main(test_data_folder, model_name):
    """Load model and evaluate with test files."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print(f"Model loaded: {model_name}")
    print("Max tokens: ", tokenizer.model_max_length)
    print("Mask token: ", tokenizer.mask_token)
    mask_token = tokenizer.mask_token

    processed_files = 0
    for file_name in os.listdir(test_data_folder):
        if file_name.endswith(".txt"):
            if processed_files >= 10:
                break
            file_path = os.path.join(test_data_folder, file_name)
            context, target = load_sequences(file_path)
            print(f"\n==> File: {file_name}")
            print(f"Context ({len(context)} nt): {context[:50]}...")
            print(f"Target  ({len(target)} nt): {target[:50]}...")

            fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
            NumMasks = len(context) // 3  # total number of tokens to predict

            # Predict token by token
            sequence = context
            predicted_tokens = []

            for _ in range(NumMasks):
                sequence_with_mask = sequence + mask_token  # add ONE mask at a time
                predictions = fill_mask(sequence_with_mask)
                top_pred = predictions[0]
                token_str = top_pred['token_str'].replace(" ", "")
                predicted_tokens.append(token_str)
                sequence += token_str  # add the predicted token to the context

            predicted_seq = "".join(predicted_tokens)

            # Compare prediction to target at different percentages
            percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
            for perc in percentages:
                compare_len = min(len(target), math.ceil(len(target) * perc))
                predicted_sub = predicted_seq[:compare_len]
                target_sub = target[:compare_len]

                matches = sum(1 for a, b in zip(predicted_sub, target_sub) if a == b)
                accuracy = matches / compare_len if compare_len > 0 else 0

                print(f"[{int(perc*100)}%] Acc: {accuracy:.3f} ({matches}/{compare_len})")

            processed_files += 1

# Dataset folder, change as needed
test_folder = "./data/test_sequences/1000bp"
model_name = "AIRI-Institute/gena-lm-bert-base-t2t"

test_results = main(test_folder, model_name)
