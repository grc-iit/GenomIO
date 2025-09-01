# Installation Guide

This guide will help you set up GenomIO on your system.

## Prerequisites

- Python 3.8 or higher
- Git
- CUDA-compatible GPU (optional, for faster inference)

## System Requirements

- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB free space for models and data
- GPU: 4GB VRAM for local model inference (optional)

## Installation Methods

### Method 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GenomIO.git
cd GenomIO

# Create virtual environment
python -m venv genomio-env
source genomio-env/bin/activate  # On Windows: genomio-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Method 2: Development Installation  

```bash
# Clone with development dependencies
git clone https://github.com/yourusername/GenomIO.git
cd GenomIO

# Create virtual environment
python -m venv genomio-dev
source genomio-dev/bin/activate

# Install with development extras
pip install -e ".[dev,notebooks]"

# Install pre-commit hooks
pre-commit install
```

## External Dependencies

### NCBI Entrez Direct (for data downloading)

```bash
# Linux/macOS
sh -c "$(curl -fsSL ftp://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"

# Or using conda
conda install -c bioconda entrez-direct
```

### GPU Support (Optional)

For CUDA support:

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verification

Test your installation:

```python
import genomio
from src.models.dnabert2 import DNABert2Model

# Test model loading
model = DNABert2Model()
print("Installation successful!")
```

## Configuration

1. Copy the example configuration:
```bash
cp config/config.yaml config/my_config.yaml
```

2. Edit `config/my_config.yaml` with your settings:
```yaml
data:
  results_dir: "/path/to/your/results"
  
environment:
  device: "cuda"  # or "cpu"
  cache_dir: "/path/to/cache"
```

## Common Issues

### Import Errors

```bash
# If you get import errors, make sure you're in the right directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA Out of Memory

```python
# Reduce batch size or use CPU
config["environment"]["device"] = "cpu"
```

### Missing Dependencies

```bash
# Install specific missing packages
pip install transformers torch biopython langchain
```

## Docker Installation (Alternative)

```dockerfile
FROM python:3.9

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
RUN pip install -e .

CMD ["python", "-c", "import genomio; print('GenomIO ready!')"]
```

Build and run:

```bash
docker build -t genomio .
docker run genomio
```