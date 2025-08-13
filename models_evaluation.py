from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM, BigBirdForMaskedLM, pipeline
import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import psutil
import shutil
from Levenshtein import distance as levenshtein_distance
import torch.nn.functional as F
import torch


def load_sequences(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    context_index = lines.index("CONTEXT:\n") + 1
    target_index = lines.index("TARGET:\n") + 1
    context = "".join(lines[context_index:target_index-1]).replace("\n", "")
    target = "".join(lines[target_index:]).replace("\n", "")
    return context, target

def evaluate_metrics(y_true, y_pred):
    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred[:len(y_true)]))
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return precision, recall, f1

def evaluate_folder(test_data_folder, fill_mask, tokenizer, model, max_input_length=512):
    mask_token = tokenizer.mask_token
    processed_files = 0
    results = []

    for file_name in os.listdir(test_data_folder):
        if file_name.endswith(".txt"):
            if processed_files >= 10:
                break
            start_time = time.time()
            process = psutil.Process(os.getpid())
            ram_before = process.memory_info().rss / (1024 ** 3)  # RAM GB
            disk_before = shutil.disk_usage("/").used / (1024 ** 3)  # Disk GB

            file_path = os.path.join(test_data_folder, file_name)
            context, target = load_sequences(file_path)
            print(f"\n==> File: {file_name}")
            print(f"Context ({len(context)} nt): {context[:50]}...")
            print(f"Target  ({len(target)} nt): {target[:50]}...")
            sequence = context
            predicted_tokens = []
            top_k_accuracy = []


            while len(sequence) - len(context) < len(target):
                sequence_trimmed = sequence

                # Adjust only the part before the <mask> so that it is divisible by 16.
                if model_name == "InstaDeepAI/segment_nt":
                    while (len(sequence_trimmed)) % 16 != 0:
                        sequence_trimmed = sequence_trimmed[:-1]

                sequence_with_mask = sequence_trimmed + mask_token
                tokenized = tokenizer(sequence_with_mask, return_tensors="pt")
                token_count = tokenized.input_ids.shape[1]


                if tokenized.input_ids.shape[1] > max_input_length:
                    print("Max token limit reached, stopping prediction.")
                    break
                try:
                    predictions = fill_mask(sequence_with_mask, top_k=60)
                except RuntimeError as e:
                    print(f"top_k={60} failed: {e}")
                    predictions = fill_mask(sequence_with_mask) 
                top_predictions = [pred['token_str'].replace(" ", "") for pred in predictions]
                remaining_target = target[len(sequence) - len(context):]
                matched_token = None
                for token in top_predictions:
                    if remaining_target.startswith(token):
                        matched_token = token
                        break
                if matched_token:
                    predicted_tokens.append(matched_token)
                    top_k_accuracy.append(1)
                else:
                    matched_token = top_predictions[0]
                    predicted_tokens.append(matched_token)
                    top_k_accuracy.append(0)
                sequence += matched_token

            predicted_seq = "".join(predicted_tokens)
            total_preds = len(top_k_accuracy)
            total_correct = sum(top_k_accuracy)
            overall_accuracy = total_correct / total_preds if total_preds > 0 else 0
            print(f"Top-60 Accuracy (match prefix of target): {overall_accuracy:.3f} ({total_correct}/{total_preds})")
            
            elapsed_time = time.time() - start_time
            ram_after = process.memory_info().rss / (1024 ** 3)
            disk_after = shutil.disk_usage("/").used / (1024 ** 3)

            ram_used = ram_after - ram_before
            disk_used = disk_after - disk_before

            percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
            for perc in percentages:
                min_len = min(len(predicted_seq), len(target))
                compare_len = min(min_len, math.ceil(len(target) * perc))
                predicted_sub = predicted_seq[:compare_len]
                target_sub = target[:compare_len]
                precision, recall, f1 = evaluate_metrics(target_sub, predicted_sub)
                matches = sum(1 for a, b in zip(predicted_sub, target_sub) if a == b)
                accuracy = matches / compare_len if compare_len > 0 else 0
                print(f"[{int(perc*100)}%] Exact Match Acc: {accuracy:.3f} ({matches}/{compare_len})")
                # Levenshtein Distance
                lev_dist = levenshtein_distance(predicted_sub, target_sub)

                max_tokens_reached = min_len < math.ceil(len(target) * perc)
                
                results.append({
                    "File": file_name,
                    "Coverage": int(perc * 100),
                    "Top-60 Accuracy": round(accuracy, 3),
                    "Precision": round(precision, 3),
                    "Recall": round(recall, 3),
                    "F1 Score": round(f1, 3),
                    "Time (s)": round(elapsed_time, 2),
                    "RAM Used (GB)": round(ram_used, 4),
                    "Disk Used (GB)": round(disk_used, 4),
                    "Levenshtein Distance": lev_dist,
                    "Max Tokens Reached": max_tokens_reached
                    #"Perplexity": round(perplexity, 3)
                })
            processed_files += 1

    return pd.DataFrame(results)

def main(folders, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab_size = len(tokenizer)
    print(f"El tamaño del vocabulario es: {vocab_size}")
    if model_name == "AIRI-Institute/gena-lm-bert-base-t2t":
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    elif model_name == "AIRI-Institute/gena-lm-bigbird-base-t2t":
        model = BigBirdForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)    
    elif model_name == "zhihan1996/DNABERT-2-117M":
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    elif model_name == "PoetschLab/GROVER":
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    elif model_name == "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species":
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    elif model_name == "InstaDeepAI/agro-nucleotide-transformer-1b":
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    elif model_name == "InstaDeepAI/segment_nt":
        model = AutoModel.from_pretrained("InstaDeepAI/segment_nt", trust_remote_code=True)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    elif model_name == "zhihan1996/DNABERT-S":
        model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True, use_flash_attn=False)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    elif model_name == "LongSafari/hyenadna-medium-160k-seqlen-hf":
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    
    print(f"Model loaded: {model_name}")
    print("Max tokens: ", tokenizer.model_max_length)
    print("Mask token: ", tokenizer.mask_token)
    if model_name == "AIRI-Institute/gena-lm-bigbird-base-t2t":
        max_input_length = 4096
    elif tokenizer.model_max_length>100000:
        max_input_length = 512
    else:
        max_input_length = tokenizer.model_max_length
    output_dir = f"./results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    for folder in folders:
        print(f"\n\n==== Evaluating folder: {folder} ====")

        df = evaluate_folder(folder, fill_mask, tokenizer, model, max_input_length)

        df["Folder"] = os.path.basename(folder)
        output_file = os.path.join(output_dir, f"{os.path.basename(folder)}_results.csv")
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

# ENTRY POINT
if __name__ == "__main__":
    test_folders = [
        "/content/drive/MyDrive/TFM/test_sequences_copy/5000bp"
    ]
    '''model_names = ["InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
                   "AIRI-Institute/gena-lm-bert-base-t2t", 
                   "PoetschLab/GROVER"
                   ]
                   '''
    #model_names = ["InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    #model_names = ["InstaDeepAI/segment_nt"]
    #model_names = ["InstaDeepAI/agro-nucleotide-transformer-1b"]
    #model_names = ["LongSafari/hyenadna-medium-160k-seqlen-hf"]
    model_names = ["AIRI-Institute/gena-lm-bigbird-base-t2t"]
    for model_name in model_names:
        main(test_folders, model_name)