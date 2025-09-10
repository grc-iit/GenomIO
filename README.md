# GenomIO

A comprehensive framework for genomic gap filling using Large Language Models (LLMs). GenomIO leverages state-of-the-art DNA language models to predict and fill gaps in genomic sequences, with support for multiple model architectures and RAG-enhanced inference.

## 🧬 Features

- **Multiple DNA Language Models**: Support for DNABERT-2, GROVER, Gena-LM, and Nucleotide Transformer
- **RAG-Enhanced Inference**: Retrieval-Augmented Generation using genomic corpus for context
- **Agent-Based Architecture**: LangChain-powered agents for intelligent gap filling
- **Comprehensive Evaluation**: Built-in evaluation metrics and benchmarking tools  
- **Modular Design**: Clean, extensible codebase ready for research and production
- **Easy Data Management**: Automated genomic data downloading and processing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GenomIO.git
cd GenomIO

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

```python
from src.core.gap_filler import GapFiller
from src.models.dnabert2 import DNABert2Model

# Initialize model and gap filler
model = DNABert2Model()
gap_filler = GapFiller(model)

# Fill gaps in a sequence
sequence_with_gaps = "ATCGATCG---GCTAGCT"
filled_sequence = gap_filler.fill_gaps(sequence_with_gaps)
print(filled_sequence)
```

### Using the Command Line Interface

```bash
# Download genomic data
genomio-download

# Fill gaps in genomic sequences
genomio-gap-fill --input data/contigs/sample.fasta --output results/filled.fasta

# Evaluate model performance
genomio-evaluate --model dnabert2 --test-data data/test_sequences/
```

## 📁 Project Structure

```
GenomIO/
├── src/                          # Source code
│   ├── models/                   # DNA language model implementations
│   │   ├── dnabert2.py          # DNABERT-2 model
│   │   ├── grover.py            # GROVER model  
│   │   ├── gena_lm.py           # Gena-LM model
│   │   └── nucleotide_transformer.py
│   ├── agents/                   # LangChain agents
│   │   ├── planner.py           # Planning agent
│   │   └── tools/               # Agent tools
│   ├── rag/                      # RAG implementation
│   │   ├── retriever.py         # Document retrieval
│   │   └── gap_filler.py        # RAG-enhanced gap filling
│   ├── core/                     # Core functionality
│   │   ├── gap_filler.py        # Main gap filling logic
│   │   ├── gap_filler_rag.py    # RAG-enhanced version
│   │   └── evaluation.py        # Model evaluation
│   └── utils/                    # Utilities
│       ├── data_download.py     # Data downloading
│       └── species_analysis.py  # Species analysis tools
├── data/                         # Data directory
│   ├── simulated_draft_genomes/ # Training/test data
│   ├── test_sequences/          # Test sequences
│   └── rag_corpus/              # RAG corpus
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Test suite
├── config/                       # Configuration files
├── docs/                         # Documentation
└── results/                      # Output results
```

## 🧪 Models Supported

### DNABERT-2
- State-of-the-art DNA language model
- 117M parameters
- Pre-trained on genomic sequences

### GROVER
- Graph-based molecular representation
- Optimized for small molecule and DNA sequences

### Gena-LM BigBird
- Long-sequence modeling capability
- 4096 token context length
- Attention mechanism optimized for genomics

### Nucleotide Transformer  
- Transformer architecture for nucleotide sequences
- 500M parameters
- Pre-trained on 1000 genomes

## 🔬 RAG Integration

GenomIO includes a Retrieval-Augmented Generation system that:

- Builds vector databases from genomic corpora
- Retrieves relevant genomic context for gap filling
- Enhanced prediction accuracy through contextual information
- Supports custom genomic databases

## 🧪 Evaluation

The framework includes comprehensive evaluation tools:

- **Accuracy Metrics**: Per-nucleotide and sequence-level accuracy
- **Biological Validity**: Checks for valid DNA sequences and ORF preservation
- **Benchmark Datasets**: Standardized test sets for fair comparison
- **Model Comparison**: Side-by-side evaluation of different models

## 📊 Usage Examples

### 1. Basic Gap Filling

```python
from src.core.gap_filler import GapFiller
from src.models.dnabert2 import DNABert2Model

model = DNABert2Model()
filler = GapFiller(model)

# Single sequence
result = filler.fill_gap("ATCG---GCTA", gap_length=150)
print(f"Filled sequence: {result}")
```

### 2. RAG-Enhanced Gap Filling

```python  
from src.core.gap_filler_rag import RAGGapFiller

rag_filler = RAGGapFiller(
    model_name="dnabert2",
    corpus_path="data/rag_corpus/",
    top_k=5
)

result = rag_filler.fill_gap_with_context("ATCG---GCTA")
```

### 3. Batch Processing

```python
from src.core.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(
    model="gena_lm",
    test_data_path="data/simulated_draft_genomes/",
    output_path="results/evaluation.csv"
)
```

## ⚙️ Configuration

Configuration is managed through `config/config.yaml`:

```yaml
models:
  dnabert2:
    name: "zhihan1996/DNABERT-2-117M"
    max_length: 512
    
gap_filling:
  max_gap_size: 10000
  num_samples: 10
  temperature: 0.8
```

## 📚 Data

### Downloading Genomic Data

Use the built-in downloader to fetch genomic sequences:

```python
from src.utils.data_download import GenomeDownloader

downloader = GenomeDownloader()
downloader.download_species_genomes([
    "Escherichia coli",
    "Staphylococcus aureus"
])
```

### Data Format

- **Input**: FASTA files with gap markers (`---`)
- **Output**: Filled FASTA sequences
- **Metadata**: TSV files with gap information

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories  
pytest tests/test_models.py
pytest tests/test_rag.py
pytest tests/test_agents.py
```

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [API Documentation](docs/api.md)
- [Usage Examples](docs/usage.md)
- [Model Comparison](docs/model_comparison.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **GRC**: Gnosis Research Center)
- **Email**: grc@illinoistech.edu
- **Issues**: [GitHub Issues](https://github.com/grc-iit/GenomIO/issues)

## 🙏 Acknowledgments

- DNABERT-2 team for their excellent pre-trained models
- LangChain for the agent framework
- The bioinformatics community for genomic datasets

## 📈 Citation

If you use GenomIO in your research, please cite:

```bibtex
@software{genomio2025,
  title   = {GenIO: Leveraging LLM Advancements in the Detection, Analysis, and Filling of Gaps During DNA Sequencing},
  author  = {Clara Aparicio Mendez},
  year    = {2025},
  school  = {Illinois Institute of Technology},
  institution = {Gnosis Research Center},
  url={https://github.com/grc-iit/GenomIO}
}
```
