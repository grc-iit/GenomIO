# API Documentation

This document provides detailed API documentation for GenomIO components.

## Core Components

### GapFiller

The main class for filling gaps in genomic sequences.

```python
class GapFiller:
    """Main gap filling functionality."""
    
    def __init__(self, model, config=None):
        """
        Initialize gap filler.
        
        Args:
            model: A DNA language model instance
            config: Optional configuration dictionary
        """
        
    def fill_gaps(self, sequence: str, **kwargs) -> str:
        """
        Fill gaps in a DNA sequence.
        
        Args:
            sequence: DNA sequence with gaps marked as '---'
            **kwargs: Additional parameters for gap filling
            
        Returns:
            str: Sequence with gaps filled
            
        Raises:
            ValueError: If sequence format is invalid
        """
        
    def fill_gap_region(self, sequence: str, start: int, end: int) -> str:
        """
        Fill a specific gap region.
        
        Args:
            sequence: Full DNA sequence
            start: Start position of gap
            end: End position of gap
            
        Returns:
            str: Sequence with specified gap filled
        """
```

### Model Classes

#### DNABert2Model

```python
class DNABert2Model(BaseGapFillingModel):
    """DNABERT-2 model for gap filling."""
    
    def __init__(self, model_name="zhihan1996/DNABERT-2-117M", **kwargs):
        """
        Initialize DNABERT-2 model.
        
        Args:
            model_name: HuggingFace model identifier
            **kwargs: Additional model parameters
        """
        
    def predict_masked_tokens(self, masked_sequence: str) -> List[str]:
        """
        Predict masked tokens in sequence.
        
        Args:
            masked_sequence: Sequence with [MASK] tokens
            
        Returns:
            List[str]: Predicted tokens for each mask
        """
        
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Encode DNA sequence to model embeddings.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            torch.Tensor: Sequence embeddings
        """
```

#### GenaLMModel

```python
class GenaLMModel(BaseGapFillingModel):
    """Gena-LM BigBird model for long sequences."""
    
    def __init__(self, model_name="AIRI-Institute/gena-lm-bigbird-base-t2t", **kwargs):
        """Initialize Gena-LM model."""
        
    def predict_long_sequence_gap(self, sequence: str, max_length=4096) -> str:
        """
        Predict gaps in long sequences using BigBird attention.
        
        Args:
            sequence: Long DNA sequence with gaps
            max_length: Maximum sequence length to process
            
        Returns:
            str: Sequence with gaps filled
        """
```

## RAG Components

### RAGGapFiller

```python
class RAGGapFiller:
    """RAG-enhanced gap filling."""
    
    def __init__(self, model_name: str, corpus_path: str, **kwargs):
        """
        Initialize RAG gap filler.
        
        Args:
            model_name: Base model to use
            corpus_path: Path to genomic corpus
            top_k: Number of documents to retrieve
            embedding_model: Sentence transformer model name
        """
        
    def build_vector_db(self) -> None:
        """Build vector database from corpus."""
        
    def get_relevant_context(self, query_sequence: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant genomic sequences.
        
        Args:
            query_sequence: Query sequence
            k: Number of sequences to retrieve
            
        Returns:
            List[str]: Retrieved sequences
        """
        
    def fill_gap_with_context(self, sequence: str) -> str:
        """
        Fill gap using retrieved context.
        
        Args:
            sequence: Sequence with gaps
            
        Returns:
            str: Gap-filled sequence
        """
```

### Retriever

```python
class GenomicRetriever:
    """Genomic sequence retrieval system."""
    
    def __init__(self, corpus_path: str, embedding_model: str):
        """Initialize retriever."""
        
    def index_corpus(self) -> None:
        """Index genomic corpus for retrieval."""
        
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Search for similar sequences.
        
        Args:
            query: Query sequence
            k: Number of results
            
        Returns:
            List[Dict]: Search results with scores
        """
```

## Agent Components

### PlannerAgent

```python
def plan(user_payload: dict) -> dict:
    """
    Plan and execute gap filling using agents.
    
    Args:
        user_payload: Dictionary with sequence and metadata
            - sequence (str): Sequence with gaps
            - gap_length (int): Expected gap length
            - meta (dict): Optional metadata
            
    Returns:
        dict: Results from agent execution
        
    Example:
        payload = {
            "sequence": "ATCG---GCTA",
            "gap_length": 150,
            "meta": {"organism": "E. coli"}
        }
        result = plan(payload)
    """
```

### Tools

```python
@tool
def context_tool(sequence: str) -> str:
    """
    Retrieve genomic context for sequence.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        str: Relevant genomic context
    """
    
@tool  
def gap_filler_tool(sequence: str, gap_length: int) -> str:
    """
    Fill gaps in sequence.
    
    Args:
        sequence: Sequence with gaps
        gap_length: Length of gap to fill
        
    Returns:
        str: Filled sequence
    """
```

## Evaluation Components

### ModelEvaluator

```python
class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config=None):
        """Initialize evaluator."""
        
    def evaluate_model(self, model: str, test_data_path: str, **kwargs) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            model: Model name or instance
            test_data_path: Path to test data
            metrics: List of metrics to compute
            output_path: Path to save results
            
        Returns:
            Dict: Evaluation results
        """
        
    def calculate_accuracy(self, predicted: str, true: str) -> float:
        """
        Calculate sequence accuracy.
        
        Args:
            predicted: Predicted sequence
            true: True sequence
            
        Returns:
            float: Accuracy score (0.0 to 1.0)
        """
        
    def calculate_bleu_score(self, predicted: str, true: str) -> float:
        """
        Calculate BLEU score for sequences.
        
        Args:
            predicted: Predicted sequence
            true: True sequence
            
        Returns:
            float: BLEU score
        """
        
    def add_custom_metric(self, name: str, func: Callable) -> None:
        """
        Add custom evaluation metric.
        
        Args:
            name: Metric name
            func: Function that takes (predicted, true) and returns score
        """
```

## Utility Components

### GenomeDownloader

```python
class GenomeDownloader:
    """Download genomic data from NCBI."""
    
    def __init__(self, output_dir: str, rag_corpus_dir: str):
        """Initialize downloader."""
        
    def download_species_genomes(self, species_list: List[str]) -> None:
        """
        Download genomes for species list.
        
        Args:
            species_list: List of species names
        """
        
    def get_assembly_info(self, species: str) -> Tuple[str, str, str]:
        """
        Get assembly information for species.
        
        Args:
            species: Species name
            
        Returns:
            Tuple: (assembly_accession, ftp_refseq, ftp_genbank)
        """
```

### Configuration

```python
def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    
def save_config(config: Dict, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
```

## Exceptions

```python
class GenomIOError(Exception):
    """Base exception for GenomIO."""
    
class InvalidSequenceError(GenomIOError):
    """Raised when sequence format is invalid."""
    
class ModelLoadError(GenomIOError):
    """Raised when model fails to load."""
    
class EvaluationError(GenomIOError):
    """Raised during evaluation failures."""
```

## Constants

```python
# DNA Alphabet
DNA_NUCLEOTIDES = ['A', 'T', 'G', 'C']
DNA_AMBIGUOUS = ['N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V']

# Gap markers
GAP_MARKER = "---"
MASK_TOKEN = "[MASK]"

# Model defaults
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 5
```

## Type Hints

```python
from typing import List, Dict, Tuple, Optional, Union, Callable
import torch
from Bio.Seq import Seq

# Custom types
DNASequence = Union[str, Seq]
GapPosition = Tuple[int, int]  # (start, end)
ModelOutput = Dict[str, Union[str, float, List]]
EvaluationResults = Dict[str, Union[float, List[float]]]
```