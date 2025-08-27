"""
Data download utilities for genomic sequences.

This module provides functionality to download genomic data from NCBI
using the Entrez Direct utilities (esearch, efetch, xtract).
"""

import os
import subprocess
import urllib.request
import gzip
from pathlib import Path
from typing import List, Tuple, Optional


class GenomeDownloader:
    """Downloads genomic sequences from NCBI databases."""
    
    def __init__(self, output_dir: str = "genomes_downloaded", rag_corpus_dir: str = "rag_corpus"):
        """
        Initialize the genome downloader.
        
        Args:
            output_dir: Directory to store downloaded genomes
            rag_corpus_dir: Directory to store RAG corpus files
        """
        self.output_dir = Path(output_dir)
        self.rag_corpus_dir = Path(rag_corpus_dir)
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.rag_corpus_dir.mkdir(exist_ok=True)
    
    def get_species_list(self) -> List[str]:
        """Get the default list of species to download."""
        return [
            "Candidatus Nealsonbacteria bacterium",
            "Candidatus Nomurabacteria bacterium",
            "Staphylococcus aureus",
            "Candidatus Moraniibacteriota bacterium",
            "Salinibacter ruber",
            "Akkermansia muciniphila",
            "Dehalococcoides mccartyi",
            "Chlamydia trachomatis",
            "Campylobacter lari",
            "Candidatus Woesebacteria bacterium",
            "Fusobacterium animalis",
            "Thermus thermophilus",
            "Pseudogemmatithrix spongiicola",
            "Archangium violaceum",
            "Fusobacterium polymorphum",
            "Nitrospira sp.",
            "Candidatus Parcubacteria bacterium",
            "Candidatus Saccharibacteria bacterium oral taxon 488",
            "Candidatus Peregrinibacteria bacterium",
            "Chlamydia psittaci",
        ]
    
    def get_assembly_info(self, species: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Get assembly information for a species using esearch and efetch.
        
        Args:
            species: Species name to search for
            
        Returns:
            Tuple of (assembly_accession, ftp_refseq, ftp_genbank)
        """
        try:
            # Build the search query
            query = f'"{species}"[Organism] AND latest[filter] AND complete genome[filter]'
            
            # Use esearch and efetch to get assembly information
            cmd = [
                "esearch", "-db", "assembly", "-query", query, "|",
                "efetch", "-format", "docsum", "|",
                "xtract", "-pattern", "DocumentSummary", 
                "-element", "AssemblyAccession", "FtpPath_RefSeq", "FtpPath_GenBank"
            ]
            
            result = subprocess.run(
                " ".join(cmd), 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            if result.stdout.strip():
                parts = result.stdout.strip().split('\t')
                assembly = parts[0] if len(parts) > 0 else None
                ftp_refseq = parts[1] if len(parts) > 1 else None
                ftp_genbank = parts[2] if len(parts) > 2 else None
                return assembly, ftp_refseq, ftp_genbank
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting assembly info for {species}: {e}")
        
        return None, None, None
    
    def download_genome(self, species: str, ftp_path: str) -> bool:
        """
        Download genome files for a species.
        
        Args:
            species: Species name
            ftp_path: FTP path to the genome files
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Get list of files in FTP directory
            response = urllib.request.urlopen(ftp_path + "/")
            file_list = response.read().decode('utf-8')
            
            # Find .fna.gz file
            fna_files = [line.split()[-1] for line in file_list.split('\n') 
                        if line.endswith('.fna.gz')]
            
            if not fna_files:
                print(f"  [WARNING] No .fna.gz file found for {species}")
                return False
            
            fna_file = fna_files[0]  # Take the first one
            fna_url = f"{ftp_path}/{fna_file}"
            
            # Create target folder
            target_folder = self.output_dir / f"{species.replace(' ', '_')}_data"
            target_folder.mkdir(exist_ok=True)
            
            # Download file
            print(f"  [INFO] Downloading: {fna_url}")
            local_file = target_folder / fna_file
            urllib.request.urlretrieve(fna_url, local_file)
            
            # Copy to RAG corpus
            rag_file = self.rag_corpus_dir / fna_file
            with open(local_file, 'rb') as src, open(rag_file, 'wb') as dst:
                dst.write(src.read())
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] Failed to download from {ftp_path}: {e}")
            return False
    
    def decompress_files(self):
        """Decompress all .fna.gz files in the RAG corpus directory."""
        print("[INFO] Decompressing all .fna.gz files into rag_corpus/")
        
        for gz_file in self.rag_corpus_dir.glob("*.fna.gz"):
            try:
                output_file = gz_file.with_suffix('')  # Remove .gz extension
                
                with gzip.open(gz_file, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                # Remove the compressed file
                gz_file.unlink()
                
            except Exception as e:
                print(f"[WARNING] Error decompressing {gz_file}: {e}")
    
    def download_species_genomes(self, species_list: Optional[List[str]] = None):
        """
        Download genomes for a list of species.
        
        Args:
            species_list: List of species to download. If None, uses default list.
        """
        if species_list is None:
            species_list = self.get_species_list()
        
        print("[INFO] Downloading .fna.gz for selected species...")
        
        for species in species_list:
            print("-" * 50)
            print(f"[INFO] Searching assembly for: {species}")
            
            assembly, ftp_refseq, ftp_genbank = self.get_assembly_info(species)
            
            if not ftp_refseq and not ftp_genbank:
                print(f"  [WARNING] No FTP path found for {species}")
                continue
            
            # Choose RefSeq over GenBank if available
            ftp_path = ftp_refseq if ftp_refseq else ftp_genbank
            
            success = self.download_genome(species, ftp_path)
            if success:
                print(f"  [INFO] Successfully downloaded {species}")
        
        # Decompress all files
        self.decompress_files()
        print("[INFO] Completed. .fna files available in rag_corpus/")


def main():
    """Main function to download genomes."""
    downloader = GenomeDownloader()
    downloader.download_species_genomes()


if __name__ == "__main__":
    main()