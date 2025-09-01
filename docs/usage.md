# Usage Guide

This guide covers the main functionality of GenomIO.

## Basic Gap Filling

### Single Sequence Gap Filling

```python
from src.core.gap_filler import GapFiller
from src.models.dnabert2 import DNABert2Model

# Initialize components
model = DNABert2Model()
gap_filler = GapFiller(model)

# Fill a gap in a DNA sequence
sequence = "ATCGATCGATCG---GCTAGCTAGCT"
result = gap_filler.fill_gaps(sequence)

print(f"Original:  {sequence}")
print(f"Filled:    {result}")
```

### Batch Processing

```python
from src.core.gap_filler import GapFiller
from Bio import SeqIO

# Process multiple sequences from a FASTA file
sequences = []
for record in SeqIO.parse("data/contigs/sample.fasta", "fasta"):
    if "---" in str(record.seq):
        filled = gap_filler.fill_gaps(str(record.seq))
        sequences.append(filled)

# Save results
with open("results/filled_sequences.fasta", "w") as f:
    for i, seq in enumerate(sequences):
        f.write(f">filled_sequence_{i}\n{seq}\n")
```

## RAG-Enhanced Gap Filling

### Setting Up RAG

```python
from src.core.gap_filler_rag import RAGGapFiller

# Initialize RAG gap filler
rag_filler = RAGGapFiller(
    model_name="dnabert2",
    corpus_path="data/rag_corpus/",
    top_k=5,
    chunk_size=1000
)

# Build vector database (one-time setup)
rag_filler.build_vector_db()
```

### Using RAG for Gap Filling

```python
# Fill gaps with contextual information
sequence_with_gap = "ATCGATCG---GCTAGCT"
context = rag_filler.get_relevant_context(sequence_with_gap)
filled_sequence = rag_filler.fill_gap_with_context(sequence_with_gap)

print(f"Context found: {len(context)} relevant sequences")
print(f"Filled sequence: {filled_sequence}")
```

## Agent-Based Gap Filling

### Using the Planning Agent

```python
from src.agents.planner import plan

# Prepare payload for the agent
payload = {
    "sequence": "ATCGATCG---GCTAGCT",
    "gap_length": 870,
    "meta": {
        "organism": "E. coli",
        "gene_region": "coding"
    }
}

# Let the agent plan and execute gap filling
result = plan(payload)
print(f"Agent result: {result}")
```

## Model Comparison

### Comparing Multiple Models

```python
from src.models import DNABert2Model, GenaLMModel, GROVERModel
from src.core.evaluation import ModelEvaluator

models = {
    "dnabert2": DNABert2Model(),
    "gena_lm": GenaLMModel(), 
    "grover": GROVERModel()
}

evaluator = ModelEvaluator()

# Test sequence with known gap
test_sequence = "ATCGATCG---GCTAGCT" 
true_gap = "AATTCCGG"

results = {}
for name, model in models.items():
    filler = GapFiller(model)
    prediction = filler.fill_gaps(test_sequence)
    accuracy = evaluator.calculate_accuracy(prediction, test_sequence.replace("---", true_gap))
    results[name] = {"prediction": prediction, "accuracy": accuracy}

# Display results
for model_name, result in results.items():
    print(f"{model_name}: {result['accuracy']:.2%} accuracy")
```

## Data Management

### Downloading Genomic Data

```python
from src.utils.data_download import GenomeDownloader

# Download specific species
downloader = GenomeDownloader(
    output_dir="my_genomes",
    rag_corpus_dir="my_rag_corpus" 
)

species_list = [
    "Escherichia coli",
    "Staphylococcus aureus",
    "Bacillus subtilis"
]

downloader.download_species_genomes(species_list)
```

### Species Analysis

```python
from src.utils.get_top_species import get_species_statistics
from src.utils.identify_species_from_accessions import identify_species

# Analyze species distribution in your data
stats = get_species_statistics("data/simulated_draft_genomes/")
print(f"Found {len(stats)} unique species")

# Identify species from accession numbers
accessions = ["CP001080.1", "AP012051.1"]
species_info = identify_species(accessions)
```

## Evaluation and Benchmarking

### Comprehensive Model Evaluation

```python
from src.core.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate on test dataset
results = evaluator.evaluate_model(
    model="dnabert2",
    test_data_path="data/simulated_draft_genomes/",
    metrics=["accuracy", "bleu", "edit_distance"],
    output_path="results/evaluation_report.csv"
)

print(f"Average accuracy: {results['accuracy']:.2%}")
print(f"Average BLEU score: {results['bleu']:.3f}")
```

### Custom Evaluation Metrics

```python
def custom_gc_content_preservation(original, predicted):
    """Check if GC content is preserved in prediction."""
    def gc_content(seq):
        return (seq.count('G') + seq.count('C')) / len(seq)
    
    orig_gc = gc_content(original)
    pred_gc = gc_content(predicted)
    return abs(orig_gc - pred_gc) < 0.05  # Within 5%

# Add to evaluator
evaluator.add_custom_metric("gc_preservation", custom_gc_content_preservation)
```

## Configuration Management

### Loading Custom Configurations

```python
from src.utils.config import load_config

# Load your custom configuration
config = load_config("config/my_config.yaml")

# Use configuration in models
model = DNABert2Model(
    model_name=config["models"]["dnabert2"]["name"],
    max_length=config["models"]["dnabert2"]["max_length"]
)
```

### Environment-Specific Settings

```python
import os
from src.utils.config import load_config

# Load configuration based on environment
env = os.getenv("GENOMIO_ENV", "development")
config_file = f"config/{env}_config.yaml"
config = load_config(config_file)
```

## Advanced Usage

### Custom Model Integration

```python
from src.models.base import BaseGapFillingModel

class MyCustomModel(BaseGapFillingModel):
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
    
    def predict_gap(self, sequence, gap_start, gap_end):
        # Your custom prediction logic
        return predicted_sequence

# Use with gap filler
custom_model = MyCustomModel("path/to/model")
gap_filler = GapFiller(custom_model)
```

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from src.core.gap_filler import GapFiller

# Create a processing pipeline
pipeline = Pipeline([
    ('preprocessor', SequencePreprocessor()),
    ('gap_filler', GapFiller(DNABert2Model())),
    ('postprocessor', SequencePostprocessor())
])

# Process sequences
results = pipeline.transform(input_sequences)
```

## Tips and Best Practices

1. **Memory Management**: For large sequences, use chunking:
```python
gap_filler.set_chunk_size(1000)  # Process in 1kb chunks
```

2. **GPU Optimization**: Enable mixed precision for faster inference:
```python
model.enable_mixed_precision()
```

3. **Caching**: Enable result caching for repeated evaluations:
```python
evaluator.enable_cache("cache/evaluation_cache/")
```

4. **Parallel Processing**: Use multiple workers for batch processing:
```python
gap_filler.set_num_workers(4)
```