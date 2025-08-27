import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import pandas as pd
from Bio import SeqIO, pairwise2
from transformers import AutoTokenizer, BigBirdForMaskedLM
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pathlib import Path
from langchain_core.documents import Document

# ----------------------------
MODEL_NAME = "AIRI-Institute/gena-lm-bigbird-base-t2t"
MAX_LENGTH = 4096
NUM_SAMPLES = 10

CONTIGS_FILE = "data/simulated_draft_genomes/contigs/AP012051.1_contigs.fasta"
GAPS_FILE = "data/simulated_draft_genomes/gaps/AP012051.1_gaps.tsv"
RESULTS_NO_RAG = "results_AP012051.1_no_rag.csv"
RESULTS_WITH_RAG = "results_AP012051.1_rag.csv"
# ----------------------------

def load_contigs(filepath):
    contigs = {}
    for record in SeqIO.parse(filepath, "fasta"):
        contigs[record.id] = str(record.seq)
    return contigs

def load_gaps(filepath):
    return pd.read_csv(filepath, sep="\t")

def load_sequences_from_fna(fna_path):
    sequences = []
    for record in SeqIO.parse(fna_path, "fasta"):
        sequences.append(record.seq.upper())  # Or str(record.seq)
    return sequences

def build_masked_input(left_seq, right_seq):
    return f"{left_seq} [MASKS] {right_seq}"

def build_masked_input_rag(left_seq, right_seq, context=None):
    base = f"{left_seq} [MASKS] {right_seq}"
    return f"{base}\n\n# Context: {context}" if context else base

def predict_until_length(masked_input_base, tokenizer, model, gap_length, max_attempts=10):
    attempt = 0
    mask_count = int(gap_length / 4)
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
        predicted_tokens = []

        for idx in mask_token_index:
            logits = outputs.logits[0, idx]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            threshold = 0.01 # Adjustable value

            # Filter tokens by probability threshold
            above_thresh = (probs >= threshold).nonzero(as_tuple=True)[0]

            if len(above_thresh) == 0:
                predicted_id = torch.argmax(probs).item()
                token = tokenizer.convert_ids_to_tokens(predicted_id)
                print(f"[MASK @ pos {idx.item()}] No token surpasses the threshold. Using argmax → {token}")
            else:
                # Show candidate tokens and their probabilities
                filtered_probs = probs[above_thresh]
                tokens = tokenizer.convert_ids_to_tokens(above_thresh.tolist())
                probs_list = filtered_probs.tolist()

                print(f"[MASK @ pos {idx.item()}] Candidate okens (p ≥ {threshold}):")
                for t, p in zip(tokens, probs_list):
                    print(f"    {t:10} → {p:.4f}")

                # Proportional sampling
                filtered_probs /= filtered_probs.sum()
                sampled_idx = torch.multinomial(filtered_probs, num_samples=1).item()
                predicted_id = above_thresh[sampled_idx].item()
                token = tokenizer.convert_ids_to_tokens(predicted_id)
                print(f"→ Token selected: {token}")

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

        # Proportional step logic based on how far we are
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

def build_faiss_index(fna_dir="rag_corpus"):
    print("[INFO] Building FAISS index from .fna files (safe HuggingFaceEmbeddings)...")
    
    documents = []
    fna_files = list(Path(fna_dir).glob("*.fna"))
    print(f"[INFO] Found {len(fna_files)} .fna files.")

    for i, fna_path in enumerate(fna_files, 1):
        print(f"[INFO] Loading file {i}/{len(fna_files)}: {fna_path.name}")
        seqs = load_sequences_from_fna(fna_path)
        for seq in seqs:
            documents.append(Document(page_content=str(seq)))   

    print(f"[INFO] Total documents loaded: {len(documents)}")
    # Paso extra: explora el contenido de los .fna cargados
    print("\n[INFO] Mostrando primeros documentos del corpus:")
    for doc in documents[:3]:
        print(doc.page_content[:200], "\n---")

    print("[INFO] Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Total chunks created: {len(chunks)}")
    print("[DEBUG] First 3 chunks:")
    for i, c in enumerate(chunks[:3]):
        print(f"  Chunk {i+1}: {c.page_content[:50]}...")


    print("[INFO] Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("[INFO] Building FAISS index from embeddings...")
    index = FAISS.from_documents(chunks, embeddings)
    
    print("[INFO] FAISS index built successfully.")
    return index

def retrieve_context(index, query_text, k=2):
    docs = index.similarity_search(query_text, k=k)
    return "\n".join(doc.page_content for doc in docs)

def run_pipeline(use_rag=False, faiss_index=None):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BigBirdForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    # Load contig sequences and gap metadata into variables.
    contigs = load_contigs(CONTIGS_FILE)
    gaps_df = load_gaps(GAPS_FILE)
    results = []

    # Loops through the first 3 gaps only, as a baseline test
    for _, row in gaps_df.head(3).iterrows():
        gap_id = row["gap_id"] if "gap_id" in row else row["ID"]
        gap_len = int(row["length"])
        real_gap_seq = row.get("sequence", "")

        gap_num = int(gap_id.split("gap")[-1])
        contig_prev_id = gap_id.replace(f"gap{gap_num}", f"contig{gap_num}")
        contig_next_id = gap_id.replace(f"gap{gap_num}", f"contig{gap_num+1}")

        if contig_prev_id not in contigs or contig_next_id not in contigs:
            print(f"[WARNING] Skipping {gap_id}, contigs not found.")
            continue

        print(f"\n Processing {gap_id} ({gap_len} bp) ...")

        left_seq = contigs[contig_prev_id]
        right_seq = contigs[contig_next_id]

        for n in range(NUM_SAMPLES):
            if use_rag and faiss_index:
                query = left_seq + right_seq
                external_context = retrieve_context(faiss_index, query)
                print(f"[DEBUG] Context retrieved for {gap_id}:\n{external_context[:300]}...\n")
                masked_input_base = build_masked_input_rag(left_seq, right_seq, external_context)
            else:
                masked_input_base = build_masked_input(left_seq, right_seq)
            
            # Calculate identity percentage
            prediction = predict_until_length(masked_input_base, tokenizer, model, gap_len)

            alignments = pairwise2.align.globalxx(prediction, real_gap_seq)
            identity = 0.0
            if alignments:
                best = alignments[0]
                score = best.score
                align_len = max(len(prediction), len(real_gap_seq))
                identity = score / align_len * 100

            results.append({
                "gap_id": gap_id,
                "prediction_number": n + 1,
                "predicted_sequence": prediction,
                "real_sequence": real_gap_seq,
                "gap_length": gap_len,
                "identity_percent": round(identity, 2),
                "rag_used": use_rag
            })

    return results

def main():
    print("Loading model...")
    print("[STEP 1] Running without RAG...")
    no_rag_results = run_pipeline(use_rag=False)
    pd.DataFrame(no_rag_results).to_csv(RESULTS_NO_RAG, index=False)
    print(f"[SAVED] {RESULTS_NO_RAG}")

    print("[STEP 2] Building FAISS index for RAG...")
    index = build_faiss_index()

    print("[STEP 3] Running with RAG...")
    rag_results = run_pipeline(use_rag=True, faiss_index=index)
    pd.DataFrame(rag_results).to_csv(RESULTS_WITH_RAG, index=False)
    print(f"[SAVED] {RESULTS_WITH_RAG}")

    print("[DONE] Both RAG and non-RAG results saved.")

if __name__ == "__main__":
    main()
