import os
import pandas as pd
from Bio import SeqIO
from transformers import AutoTokenizer, BigBirdForMaskedLM
import torch

# ----------------------------
MODEL_NAME = "AIRI-Institute/gena-lm-bigbird-base-t2t"
MAX_LENGTH = 4096
NUM_SAMPLES = 6

CONTIGS_FILE = "data/simulated_draft_genomes/contigs/AP012051.1_contigs.fasta"
GAPS_FILE = "data/simulated_draft_genomes/gaps/AP012051.1_gaps.tsv"
OUTPUT_CSV = "results_AP012051.1.csv"
# ----------------------------

#  Load all contigs into a dictionary (key: contigID, value: sequence)
def load_contigs(filepath):
    contigs = {}
    for record in SeqIO.parse(filepath, "fasta"):
        contigs[record.id] = str(record.seq)
    return contigs

#  Loads the .tsv file with gap info as a Pandas DataFrame
def load_gaps(filepath):
    return pd.read_csv(filepath, sep="\t")

# Build the input string for the model:
def build_masked_input(left_seq, right_seq):
    # Insert a number of [MASK] tokens that approximates the gap size (gap_length // 6)
    #mask_count = max(NUM_MASK_TOKENS_MIN, gap_length // (3))
    #mask_count=1
    #masked_gap = " ".join(["[MASK]"] * mask_count)
    # Take the last 1000 nt of the previous contig (left_seq)
    left = left_seq[-1000:] if len(left_seq) > 1000 else left_seq
    # Take the first 1000 nt of the next contig (right_seq)
    right = right_seq[:1000] if len(right_seq) > 1000 else right_seq
    #return f"{left} {masked_gap} {right}"
    return f"{left} [MASKS] {right}"

# Predict masked tokens (gaps)
def predict_masked_sequence(masked_input, tokenizer, model):
    inputs = tokenizer(masked_input, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        outputs = model(**inputs)

    # Find mask token positions
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    # Get predicted token ids only at those positions
    predicted_tokens = []
    for idx in mask_token_index:
        logits = outputs.logits[0, idx]
        predicted_id = torch.argmax(logits).item()
        token = tokenizer.convert_ids_to_tokens(predicted_id)
        predicted_tokens.append(token)

    return "".join(predicted_tokens)

def predict_until_length(masked_input_base, tokenizer, model, gap_length, max_attempts=10):
    """
    Attempts to predict a sequence of approximately gap_length nt,
    dynamically adjusting the number of [MASK] tokens based on how far off the prediction is.
    """
    attempt = 0
    mask_count = int(gap_length / 4)  # Reasonable starting point

    best_seq = ""
    best_diff = float("inf")

    while attempt < max_attempts:
        # Build masked sequence
        masked_gap = " ".join(["[MASK]"] * mask_count)
        masked_input = masked_input_base.replace("[MASKS]", masked_gap)

        # Tokenize and predict
        inputs = tokenizer(masked_input, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            outputs = model(**inputs)

        # Identify mask positions
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        # Predict tokens at each [MASK]
        predicted_tokens = []
        for idx in mask_token_index:
            logits = outputs.logits[0, idx]
            predicted_id = torch.argmax(logits).item()
            token = tokenizer.convert_ids_to_tokens(predicted_id)
            predicted_tokens.append(token)

        # Join predicted tokens and compute sequence length
        predicted_seq = "".join(predicted_tokens).replace("▁", "")
        seq_length = len(predicted_seq)
        diff = abs(seq_length - gap_length)
        relative_diff = diff / gap_length

        # Logging
        print(f"Attempt {attempt+1}: mask_count = {mask_count}, predicted_length = {seq_length}, target_length = {gap_length}, diff = {diff}")

        # Check for best match
        if diff < best_diff:
            best_seq = predicted_seq
            best_diff = diff

        # Stop if close enough
        if diff <= 3:
            return predicted_seq

        # --- Proportional step logic based on how far we are ---
        if relative_diff > 0.2:
            adjustment = max(5, int(diff / 4))  # aggressive if far
        elif relative_diff > 0.1:
            adjustment = max(3, int(diff / 6))  # moderate
        else:
            adjustment = max(1, int(diff / 8))  # fine tuning

        # Apply adjustment based on direction
        if seq_length < gap_length:
            mask_count += adjustment
        else:
            mask_count = max(1, mask_count - adjustment)

        attempt += 1

    print(f"Max attempts reached. Returning best prediction (length = {len(best_seq)})")
    return best_seq


def main():
    print("Loading model...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BigBirdForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    # Load contig sequences and gap metadata into variables.
    contigs = load_contigs(CONTIGS_FILE)
    gaps_df = load_gaps(GAPS_FILE)
    results = []

    # Loops through the first 3 gaps only, as a baseline test
    for i, row in gaps_df.head(3).iterrows():
        gap_id = row["gap_id"] if "gap_id" in row else row["ID"]
        gap_len = int(row["length"])
        real_gap_seq = row["sequence"] if "sequence" in row else ""

        gap_num = int(gap_id.split("gap")[-1])
        contig_prev_id = gap_id.replace(f"gap{gap_num}", f"contig{gap_num}")
        contig_next_id = gap_id.replace(f"gap{gap_num}", f"contig{gap_num+1}")

        if contig_prev_id not in contigs or contig_next_id not in contigs:
            print(f"Skipping {gap_id}, contigs not found.")
            continue

        print(f"\n Processing {gap_id} ({gap_len} bp) ...")
        #masked_input = build_masked_input(contigs[contig_prev_id], contigs[contig_next_id], gap_len)
        masked_input_base = build_masked_input(contigs[contig_prev_id], contigs[contig_next_id])

        for n in range(NUM_SAMPLES):
            #prediction = predict_masked_sequence(masked_input, tokenizer, model)
            prediction = predict_until_length(masked_input_base, tokenizer, model, gap_len)
            
            results.append({
                "gap_id": gap_id,
                "prediction_number": n + 1,
                "predicted_sequence": prediction,
                "real_sequence": real_gap_seq,
                "gap_length": gap_len
            })
            print(f"Prediction {n+1} ")

    # Save results
    print(f"\n Saving results to {OUTPUT_CSV} ...")
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
