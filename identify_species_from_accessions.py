from Bio import Entrez, SeqIO
import pandas as pd
import time
import glob
import os

Entrez.email = "clara.apmendez@gmail.com"  # Required by NCBI

def get_species_from_accession(accession):
    try:
        # Step 1: Get UID from accession
        search_handle = Entrez.esearch(db="nucleotide", term=accession, retmode="xml")
        search_results = Entrez.read(search_handle)
        search_handle.close()

        if not search_results["IdList"]:
            return "Not found"

        uid = search_results["IdList"][0]

        # Step 2: Fetch full GenBank record
        fetch_handle = Entrez.efetch(db="nucleotide", id=uid, rettype="gb", retmode="text")
        record_text = fetch_handle.read()
        fetch_handle.close()

        # Step 3: Extract species name from "ORGANISM" field
        for line in record_text.splitlines():
            if line.strip().startswith("ORGANISM"):
                return line.strip().replace("ORGANISM", "").strip()

        return "ORGANISM field not found"
    except Exception as e:
        print(f"[ERROR] {accession}: {e}")
        return "Error"

def extract_accessions_from_multiple_fastas(folder):
    accessions = set()
    fasta_files = glob.glob(os.path.join(folder, "*.fasta"))
    for fpath in fasta_files:
        for record in SeqIO.parse(fpath, "fasta"):
            acc = record.id.split(".")[0]  # AP012051.1 → AP012051
            accessions.add(acc)
    return list(accessions)

def main():
    fasta_folder = "data/simulated_draft_genomes/contigs"  # Adjust to your path
    accessions = extract_accessions_from_multiple_fastas(fasta_folder)

    print(f"[INFO] {len(accessions)} unique accessions found.")

    results = []
    for acc in accessions:
        print(f"[INFO] Looking up species for accession: {acc}")
        species = get_species_from_accession(acc)
        print(f" → {species}")
        results.append({"accession": acc, "species": species})
        time.sleep(0.4)  # Avoid overwhelming NCBI's servers

    df = pd.DataFrame(results)
    df.to_csv("contig_accessions_species.csv", index=False)
    print("\n[INFO] Results saved to contig_accessions_species.csv")

if __name__ == "__main__":
    main()
