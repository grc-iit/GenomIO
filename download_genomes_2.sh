#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Create output directories if they don't exist
mkdir -p genomes_downloaded
mkdir -p rag_corpus

# List of species to download
species_list=(
  "Candidatus Nealsonbacteria bacterium"
  "Candidatus Nomurabacteria bacterium"
  "Staphylococcus aureus"
  "Candidatus Moraniibacteriota bacterium"
  "Salinibacter ruber"
  "Akkermansia muciniphila"
  "Dehalococcoides mccartyi"
  "Chlamydia trachomatis"
  "Campylobacter lari"
  "Candidatus Woesebacteria bacterium"
  "Fusobacterium animalis"
  "Thermus thermophilus"
  "Pseudogemmatithrix spongiicola"
  "Archangium violaceum"
  "Fusobacterium polymorphum"
  "Nitrospira sp."
  "Candidatus Parcubacteria bacterium"
  "Candidatus Saccharibacteria bacterium oral taxon 488"
  "Candidatus Peregrinibacteria bacterium"
  "Chlamydia psittaci"
)

echo "[INFO] Downloading .fna.gz for selected species..."

# Loop through each species
for species in "${species_list[@]}"; do
  echo "------------------------------"
  echo "[INFO] Searching assembly for: $species"
  
  read -r ASSEMBLY FTP_REF FTP_GEN <<< $(esearch -db assembly -query "\"$species\"[Organism] AND latest[filter] AND complete genome[filter]" | \
    efetch -format docsum | \
    xtract -pattern DocumentSummary -element AssemblyAccession FtpPath_RefSeq FtpPath_GenBank | head -n 1)

  if [[ -z "$FTP_REF" && -z "$FTP_GEN" ]]; then
    echo "  [WARNING] No FTP path found for $species"
    continue
  fi

  # Choose the first available FTP path (RefSeq preferred)
  if [[ -n "$FTP_REF" ]]; then
    FTP_PATH="$FTP_REF"
  else
    FTP_PATH="$FTP_GEN"
  fi

  echo "  [INFO] Listing available files in $FTP_PATH"
  FILE_LIST=$(curl -s "$FTP_PATH/")
  
  # Find any file ending with .fna.gz (could be genomic, cds_from_genomic, etc.)
  FNA_FILE=$(echo "$FILE_LIST" | grep 'fna.gz' | awk '{print $NF}' | head -n 1)


  if [[ -z "$FNA_FILE" ]]; then
    echo "  [WARNING] No .fna.gz file found for $species"
    continue
  fi

  # Build full download URL and target folder
  FNA_URL="${FTP_PATH}/${FNA_FILE}"
  TARGET_FOLDER="genomes_downloaded/${species// /_}_data"
  mkdir -p "$TARGET_FOLDER"

  echo "  [INFO] Downloading: $FNA_URL"
  echo "  [DEBUG] Getting file list from: $FTP_PATH"

  curl -s -o "${TARGET_FOLDER}/${FNA_FILE}" "$FNA_URL" || {
    echo "  [ERROR] Failed to download: $FNA_URL"
    continue
  }

  cp "${TARGET_FOLDER}/${FNA_FILE}" rag_corpus/
done

# Decompress all .fna.gz files in rag_corpus/
echo "[INFO] Decompressing all .fna.gz files into rag_corpus/"
gunzip -f rag_corpus/*.fna.gz || echo "[WARNING] Some files may already be uncompressed."

echo "[INFO] Completed. .fna files available in rag_corpus/"
