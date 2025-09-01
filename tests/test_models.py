"""
Tests for genomic models.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from src.models.dnabert2 import DNABert2Model
from src.models.gena_lm import GenaLMModel


class TestDNABert2Model:
    """Tests for DNABERT-2 model."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock DNABERT-2 model for testing."""
        with patch('src.models.dnabert2.AutoTokenizer'), \
             patch('src.models.dnabert2.AutoModelForMaskedLM'):
            model = DNABert2Model()
            return model
    
    def test_model_initialization(self, mock_model):
        """Test model initialization."""
        assert mock_model is not None
        assert hasattr(mock_model, 'tokenizer')
        assert hasattr(mock_model, 'model')
    
    def test_sequence_encoding(self, mock_model):
        """Test DNA sequence encoding."""
        sequence = "ATCGATCGATCG"
        
        # Mock tokenizer output
        mock_model.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        
        result = mock_model.encode_sequence(sequence)
        assert result is not None
    
    def test_invalid_sequence(self, mock_model):
        """Test handling of invalid DNA sequences."""
        invalid_sequence = "ATCGXYZ"  # Contains invalid characters
        
        with pytest.raises(ValueError):
            mock_model.encode_sequence(invalid_sequence)
    
    def test_empty_sequence(self, mock_model):
        """Test handling of empty sequences."""
        with pytest.raises(ValueError):
            mock_model.encode_sequence("")
    
    def test_mask_prediction(self, mock_model):
        """Test masked token prediction."""
        masked_sequence = "ATCG[MASK]GCTA"
        
        # Mock model output
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 10, 1000)  # Batch, seq_len, vocab
        mock_model.model.return_value = mock_output
        
        predictions = mock_model.predict_masked_tokens(masked_sequence)
        assert isinstance(predictions, list)


class TestGenaLMModel:
    """Tests for Gena-LM model."""
    
    @pytest.fixture  
    def mock_gena_model(self):
        """Create a mock Gena-LM model for testing."""
        with patch('src.models.gena_lm.AutoTokenizer'), \
             patch('src.models.gena_lm.BigBirdForMaskedLM'):
            model = GenaLMModel()
            return model
    
    def test_long_sequence_handling(self, mock_gena_model):
        """Test handling of long DNA sequences."""
        long_sequence = "ATCG" * 1000  # 4000 nucleotides
        
        # Mock tokenizer for long sequences
        mock_gena_model.tokenizer.return_value = {
            'input_ids': torch.tensor([list(range(1000))]),
            'attention_mask': torch.tensor([[1] * 1000])
        }
        
        result = mock_gena_model.encode_sequence(long_sequence)
        assert result is not None
    
    def test_max_length_constraint(self, mock_gena_model):
        """Test maximum length constraint."""
        very_long_sequence = "ATCG" * 2000  # 8000 nucleotides (exceeds 4096)
        
        # Should truncate or handle appropriately
        result = mock_gena_model.encode_sequence(very_long_sequence)
        assert result is not None


class TestModelUtilities:
    """Tests for model utility functions."""
    
    def test_dna_sequence_validation(self):
        """Test DNA sequence validation utilities."""
        from src.models.base import validate_dna_sequence
        
        # Valid sequences
        assert validate_dna_sequence("ATCGATCG")
        assert validate_dna_sequence("ATCGATCGNNNN")  # With ambiguous nucleotides
        
        # Invalid sequences
        assert not validate_dna_sequence("ATCGXYZ")
        assert not validate_dna_sequence("123456")
        assert not validate_dna_sequence("")
    
    def test_sequence_preprocessing(self):
        """Test sequence preprocessing utilities."""
        from src.models.base import preprocess_sequence
        
        # Test case conversion
        result = preprocess_sequence("atcgatcg")
        assert result == "ATCGATCG"
        
        # Test whitespace removal
        result = preprocess_sequence("ATCG ATCG")
        assert result == "ATCGATCG"
        
        # Test newline removal
        result = preprocess_sequence("ATCG\nATCG")
        assert result == "ATCGATCG"


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for models (require actual model loading)."""
    
    @pytest.mark.slow
    def test_actual_dnabert2_loading(self):
        """Test loading actual DNABERT-2 model."""
        # Skip if no internet or model not available
        pytest.skip("Requires model download - run with --run-slow")
        
        model = DNABert2Model()
        sequence = "ATCGATCGATCG"
        result = model.encode_sequence(sequence)
        assert result is not None
        assert result.shape[0] > 0
    
    @pytest.mark.slow 
    def test_model_inference_time(self):
        """Test model inference performance."""
        pytest.skip("Performance test - run with --run-slow")
        
        import time
        model = DNABert2Model()
        sequence = "ATCG" * 100  # 400 nucleotides
        
        start_time = time.time()
        result = model.encode_sequence(sequence)
        inference_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert inference_time < 10.0  # Less than 10 seconds
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])