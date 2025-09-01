import pandas as pd
import unicodedata
import re
import os

def clean_species_name(name):
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    name = re.sub(r"[\"']", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def format_species_list_block(species_list):
    block = ["species_list=("]
    for name in species_list:
        cleaned = clean_species_name(name)
        bash_name = cleaned.lower().replace(" ", "_").replace("/", "_")
        block.append(f'  "{bash_name}"')
    block.append(")")
    return "\n".join(block)

def generate_download_bash(csv_path, output_bash="download_genomes.sh", top_n=20):
    df = pd.read_csv(csv_path)
    if 'species' not in df.columns:
        df = df.reset_index()
        df.columns = ['species', 'count']

    species_names = df['species'].head(top_n).tolist()
    species_block = format_species_list_block(species_names)

    bash_script = f"""#!/bin/bash

set -e

mkdir -p genomes_downloaded
mkdir -p rag_corpus

{species_block}

# Step 1: Download, unzip, rehydrate
for species in "${{species_list[@]}}"; do
  echo "[INFO] Downloading $species..."
  zipfile="$species.zip"
  folder="genomes_downloaded/${{species}}_data"
  mkdir -p "$folder"
  datasets download genome taxon "${{species//_/ }}" --assembly-level complete --reference --dehydrated --filename "$zipfile"
  unzip -q "$zipfile" -d "$folder"
  datasets rehydrate --directory "$folder"
done

# Step 2: Collect all .fna files into rag_corpus
echo "[INFO] Copying all .fna files to rag_corpus/"
find genomes_downloaded -type f -name "*.fna" -exec cp {{}} rag_corpus/ \\;

# Step 3: Build FAISS index
echo "[INFO] Building FAISS index from rag_corpus..."
python3 - <<EOF
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Load and split documents
folder = "rag_corpus"
docs = []
for fname in os.listdir(folder):
    if fname.endswith(".fna"):
        loader = TextLoader(os.path.join(folder, fname))
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = splitter.split_documents(docs)

# Embed and build FAISS index
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(splits, embedding)
db.save_local("faiss_genomic_index")

print("[INFO] FAISS index saved in faiss_genomic_index/")
EOF
"""

    with open(output_bash, "w") as f:
        f.write(bash_script)
    os.chmod(output_bash, 0o755)

    print(f"[INFO] Script created: {output_bash}")
    print(f"[INFO] Run it with: ./download_genomes.sh")

# -----------------------------
if __name__ == "__main__":
    # Load species CSV
    df = pd.read_csv("contig_accessions_species.csv")

    # Count frequency of each species
    species_counts = df['species'].value_counts()

    # Choose top N species
    top_n = 20
    top_species = species_counts.head(top_n)

    print("[INFO] Top species by count:")
    print(top_species)

    # Save to file
    top_species.to_csv("top_species.csv", header=True)
    generate_download_bash("top_species.csv")
