"""
Tests for RAG (Retrieval-Augmented Generation) components.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from src.rag.retriever import GenomicRetriever
from src.rag.gap_filler import RAGGapFiller
from src.core.gap_filler_rag import RAGGapFiller as CoreRAGGapFiller


class TestGenomicRetriever:
    """Tests for genomic sequence retrieval."""
    
    @pytest.fixture
    def temp_corpus_dir(self):
        """Create temporary corpus directory with test data."""
        temp_dir = tempfile.mkdtemp()
        
        # Create test FASTA files
        test_sequences = [
            ">seq1\nATCGATCGATCGATCG\n",
            ">seq2\nGCTAGCTAGCTAGCTA\n", 
            ">seq3\nTAGCTAGCTAGCTAGC\n"
        ]
        
        for i, seq_content in enumerate(test_sequences):
            with open(os.path.join(temp_dir, f"test_{i}.fna"), "w") as f:
                f.write(seq_content)
        
        yield temp_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def retriever(self, temp_corpus_dir):
        """Create retriever instance with test corpus."""
        with patch('src.rag.retriever.SentenceTransformer'):
            retriever = GenomicRetriever(
                corpus_path=temp_corpus_dir,
                embedding_model="test-model"
            )
            return retriever
    
    def test_retriever_initialization(self, retriever):
        """Test retriever initialization."""
        assert retriever is not None
        assert hasattr(retriever, 'corpus_path')
        assert hasattr(retriever, 'embedding_model')
    
    def test_corpus_indexing(self, retriever):
        """Test corpus indexing functionality."""
        # Mock the embedding model
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        retriever.embedding_model.encode = Mock(return_value=mock_embeddings)
        
        retriever.index_corpus()
        
        assert hasattr(retriever, 'corpus_embeddings')
        assert hasattr(retriever, 'sequences')
    
    def test_similarity_search(self, retriever):
        """Test similarity search functionality."""
        # Set up mock data
        retriever.sequences = ["ATCGATCG", "GCTAGCTA", "TAGCTAG"]
        retriever.corpus_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        
        # Mock embedding for query
        retriever.embedding_model.encode = Mock(return_value=[[0.2, 0.3]])
        
        # Mock similarity computation
        with patch('src.rag.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.9, 0.7, 0.5]]
            
            results = retriever.search("ATCGATCG", k=2)
            
            assert len(results) == 2
            assert all('sequence' in result for result in results)
            assert all('score' in result for result in results)
    
    def test_empty_corpus_handling(self):
        """Test handling of empty corpus."""
        empty_dir = tempfile.mkdtemp()
        
        with patch('src.rag.retriever.SentenceTransformer'):
            retriever = GenomicRetriever(empty_dir, "test-model")
            
            with pytest.raises(ValueError, match="No sequences found"):
                retriever.index_corpus()
        
        os.rmdir(empty_dir)


class TestRAGGapFiller:
    """Tests for RAG-enhanced gap filling."""
    
    @pytest.fixture
    def mock_rag_filler(self):
        """Create mock RAG gap filler."""
        with patch('src.rag.gap_filler.GenomicRetriever'), \
             patch('src.rag.gap_filler.DNABert2Model'):
            
            rag_filler = RAGGapFiller(
                model_name="dnabert2",
                corpus_path="/fake/path",
                top_k=3
            )
            return rag_filler
    
    def test_context_retrieval(self, mock_rag_filler):
        """Test genomic context retrieval."""
        # Mock retriever results
        mock_results = [
            {"sequence": "ATCGATCGATCG", "score": 0.9},
            {"sequence": "GCTAGCTAGCTA", "score": 0.8},
            {"sequence": "TAGCTAGCTAG", "score": 0.7}
        ]
        
        mock_rag_filler.retriever.search = Mock(return_value=mock_results)
        
        query_sequence = "ATCG---GCTA"
        context = mock_rag_filler.get_relevant_context(query_sequence)
        
        assert len(context) == 3
        assert all(isinstance(ctx, str) for ctx in context)
    
    def test_gap_filling_with_context(self, mock_rag_filler):
        """Test gap filling using retrieved context."""
        # Mock context retrieval
        mock_rag_filler.get_relevant_context = Mock(
            return_value=["ATCGATCGATCG", "GCTAGCTAGCTA"]
        )
        
        # Mock model prediction
        mock_rag_filler.model.fill_gaps = Mock(
            return_value="ATCGAAATGCTA"
        )
        
        sequence_with_gap = "ATCG---GCTA"
        result = mock_rag_filler.fill_gap_with_context(sequence_with_gap)
        
        assert result == "ATCGAAATGCTA"
        assert "---" not in result
    
    def test_context_quality_filtering(self, mock_rag_filler):
        """Test filtering of low-quality context."""
        # Mock retriever with mixed quality results
        mock_results = [
            {"sequence": "ATCGATCGATCG", "score": 0.9},  # High quality
            {"sequence": "NNNNNNNNNNNN", "score": 0.8},  # Low quality (all N's)
            {"sequence": "GCTAGCTAGCTA", "score": 0.7},  # Good quality
            {"sequence": "XXXXX", "score": 0.6},          # Invalid sequence
        ]
        
        mock_rag_filler.retriever.search = Mock(return_value=mock_results)
        
        # Should filter out low-quality sequences
        context = mock_rag_filler.get_relevant_context("ATCG---GCTA", min_score=0.6)
        
        # Should exclude sequences with all N's or invalid characters
        valid_context = [ctx for ctx in context if not all(c == 'N' for c in ctx)]
        assert len(valid_context) >= 1


class TestCoreRAGGapFiller:
    """Tests for core RAG gap filling functionality."""
    
    @pytest.fixture
    def temp_corpus(self):
        """Create temporary corpus for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create test genomic sequences
        sequences = [
            ">genome1\nATCGATCGATCGATCGATCGATCG\n",
            ">genome2\nGCTAGCTAGCTAGCTAGCTAGCTA\n",
            ">genome3\nTAGCTAGCTAGCTAGCTAGCTAGC\n",
        ]
        
        for i, seq in enumerate(sequences):
            with open(os.path.join(temp_dir, f"genome_{i}.fna"), "w") as f:
                f.write(seq)
        
        yield temp_dir
        
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def core_rag_filler(self, temp_corpus):
        """Create core RAG gap filler with real corpus."""
        with patch('src.core.gap_filler_rag.SentenceTransformer'), \
             patch('src.core.gap_filler_rag.AutoTokenizer'), \
             patch('src.core.gap_filler_rag.AutoModelForMaskedLM'):
            
            filler = CoreRAGGapFiller(
                model_name="dnabert2",
                corpus_path=temp_corpus,
                top_k=2
            )
            return filler
    
    def test_vector_db_building(self, core_rag_filler):
        """Test vector database construction."""
        # Mock embedding generation
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        core_rag_filler.embedding_model.encode = Mock(return_value=mock_embeddings)
        
        core_rag_filler.build_vector_db()
        
        assert hasattr(core_rag_filler, 'vector_db')
        assert hasattr(core_rag_filler, 'sequences')
        assert len(core_rag_filler.sequences) > 0
    
    def test_end_to_end_gap_filling(self, core_rag_filler):
        """Test complete RAG gap filling pipeline."""
        # Mock all components
        core_rag_filler.sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
        core_rag_filler.vector_db = Mock()
        
        # Mock similarity search
        core_rag_filler.vector_db.search = Mock(
            return_value=([0], [[0.9]])
        )
        
        # Mock model prediction
        mock_outputs = Mock()
        mock_outputs.logits = Mock()
        core_rag_filler.model.return_value = mock_outputs
        core_rag_filler.tokenizer.decode = Mock(return_value="AAAT")
        
        sequence_with_gap = "ATCG---GCTA"
        result = core_rag_filler.fill_gap_with_context(sequence_with_gap)
        
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests for RAG components."""
    
    @pytest.mark.slow
    def test_full_rag_pipeline(self):
        """Test complete RAG pipeline with real models."""
        pytest.skip("Integration test - requires model downloads")
        
        # This would test the complete pipeline:
        # 1. Load real corpus
        # 2. Build vector database
        # 3. Retrieve context
        # 4. Fill gaps with context
        # 5. Evaluate results
        pass
    
    def test_rag_performance_benchmark(self):
        """Benchmark RAG vs non-RAG performance."""
        pytest.skip("Performance benchmark - run separately")
        
        # This would compare:
        # - Gap filling accuracy with vs without RAG
        # - Inference time with vs without RAG
        # - Memory usage comparison
        pass


if __name__ == "__main__":
    pytest.main([__file__])