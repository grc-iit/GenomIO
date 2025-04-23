from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import os
import math

# Load context and target sequences from a file
def load_sequences(file_path):
    """Load context and target sequences from file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Identify lines where CONTEXT and TARGET begin
    context_index = lines.index("CONTEXT:\n") + 1
    target_index = lines.index("TARGET:\n") + 1

    # Extract sequences, removing newline characters
    context = "".join(lines[context_index:target_index-1]).replace("\n", "")
    target = "".join(lines[target_index:]).replace("\n", "")

    return context, target

# Main function to run evaluation with a specified model and dataset
def main(test_data_folder, model_name):
    """Load model and evaluate with test files using GROVER."""

    # Load tokenizer and model from Hugging Face hub
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

    print(f"Model loaded: {model_name}")
    print("Max tokens: ", tokenizer.model_max_length)
    print("Mask token: ", tokenizer.mask_token)
    mask_token = tokenizer.mask_token

    # Create fill-mask pipeline
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    processed_files = 0

    # Iterate through files in the test folder
    for file_name in os.listdir(test_data_folder):
        if file_name.endswith(".txt"):
            if processed_files >= 10:  # Limit evaluation to 10 files
                break

            file_path = os.path.join(test_data_folder, file_name)
            context, target = load_sequences(file_path)

            print(f"\n==> File: {file_name}")
            print(f"Context ({len(context)} nt): {context[:50]}...")
            print(f"Target  ({len(target)} nt): {target[:50]}...")

            # Initialize the sequence with the context
            sequence = context
            predicted_tokens = []
            top_k_accuracy = []  # Stores 1 if prediction matched target prefix, 0 otherwise

            # Predict until the full target is reconstructed
            while len(sequence) - len(context) < len(target):
                tokenized = tokenizer(sequence, return_tensors="pt")

                # Check if the input is within the model's token limit
                if tokenized.input_ids.shape[1] >= tokenizer.model_max_length:
                    print("Max length reached")
                    break

                # Add a [MASK] at the end of the current sequence
                sequence_with_mask = sequence + mask_token

                # Get top-60 predictions for the masked token
                predictions = fill_mask(sequence_with_mask, top_k=60)
                top_predictions = [pred['token_str'].replace(" ", "") for pred in predictions]

                # Compute remaining target substring to match predictions against
                remaining_target = target[len(sequence) - len(context):]

                matched_token = None
                # Check if any top prediction is a prefix of the remaining target
                for token in top_predictions:
                    if remaining_target.startswith(token):
                        matched_token = token
                        break

                # If a matching token was found among top predictions
                if matched_token:
                    predicted_tokens.append(matched_token)
                    top_k_accuracy.append(1)
                else:
                    # Otherwise, take top-1 prediction as fallback
                    matched_token = top_predictions[0]
                    predicted_tokens.append(matched_token)
                    top_k_accuracy.append(0)

                # Extend the input sequence with the selected token
                sequence += matched_token

            # Join all predicted tokens into a final string
            predicted_seq = "".join(predicted_tokens)

            # Compute top-60 accuracy
            total_preds = len(top_k_accuracy)
            total_correct = sum(top_k_accuracy)
            overall_accuracy = total_correct / total_preds if total_preds > 0 else 0

            print(f"Top-60 Accuracy (match prefix of target): {overall_accuracy:.3f} ({total_correct}/{total_preds})")

            # (Optional) Compute exact character-level accuracy at various fractions
            percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
            for perc in percentages:
                compare_len = min(len(target), math.ceil(len(target) * perc))
                predicted_sub = predicted_seq[:compare_len]
                target_sub = target[:compare_len]

                matches = sum(1 for a, b in zip(predicted_sub, target_sub) if a == b)
                accuracy = matches / compare_len if compare_len > 0 else 0

                print(f"[{int(perc*100)}%] Exact Match Acc: {accuracy:.3f} ({matches}/{compare_len})")

            processed_files += 1

# === Run script with GROVER model ===
test_folder = "./data/test_sequences/1000bp"  # Folder with test files
model_name = "PoetschLab/GROVER"              # GROVER model on Hugging Face

main(test_folder, model_name)
