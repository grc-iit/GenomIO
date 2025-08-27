"""
Tests for agent-based gap filling components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.planner import plan
from src.agents.tools.planner_tools import context_tool, gap_filler_tool


class TestPlannerAgent:
    """Tests for the planning agent."""
    
    def test_valid_payload_processing(self):
        """Test processing of valid user payload."""
        valid_payload = {
            "sequence": "ATCGATCG---GCTAGCT",
            "gap_length": 150,
            "meta": {"organism": "E. coli"}
        }
        
        with patch('src.agents.planner.init_chat_model'), \
             patch('src.agents.planner.create_tool_calling_agent'), \
             patch('src.agents.planner.AgentExecutor') as mock_executor:
            
            # Mock agent execution
            mock_executor_instance = Mock()
            mock_executor_instance.invoke.return_value = {
                "output": "Successfully filled gap: ATCGATCGAAAGCTAGCT"
            }
            mock_executor.return_value = mock_executor_instance
            
            result = plan(valid_payload)
            
            assert result is not None
            assert "output" in result
    
    def test_invalid_payload_missing_sequence(self):
        """Test handling of payload missing sequence."""
        invalid_payload = {
            "gap_length": 150,
            "meta": {"organism": "E. coli"}
        }
        
        with pytest.raises(ValueError, match="Missing 'sequence'"):
            plan(invalid_payload)
    
    def test_invalid_payload_missing_gap_length(self):
        """Test handling of payload missing gap length."""
        invalid_payload = {
            "sequence": "ATCGATCG---GCTAGCT",
            "meta": {"organism": "E. coli"}
        }
        
        with pytest.raises(ValueError, match="Missing 'gap_length'"):
            plan(invalid_payload)
    
    def test_invalid_payload_type(self):
        """Test handling of non-dict payload."""
        invalid_payload = "not a dictionary"
        
        with pytest.raises(TypeError, match="expects a dict"):
            plan(invalid_payload)
    
    def test_payload_with_optional_meta(self):
        """Test payload processing with optional metadata."""
        payload_no_meta = {
            "sequence": "ATCGATCG---GCTAGCT", 
            "gap_length": 150
        }
        
        with patch('src.agents.planner.init_chat_model'), \
             patch('src.agents.planner.create_tool_calling_agent'), \
             patch('src.agents.planner.AgentExecutor') as mock_executor:
            
            mock_executor_instance = Mock()
            mock_executor_instance.invoke.return_value = {"output": "Success"}
            mock_executor.return_value = mock_executor_instance
            
            # Should not raise an error
            result = plan(payload_no_meta)
            assert result is not None
    
    @patch('src.agents.planner.init_chat_model')
    @patch('src.agents.planner.create_tool_calling_agent') 
    @patch('src.agents.planner.AgentExecutor')
    def test_agent_tool_chain_execution(self, mock_executor, mock_create_agent, mock_init_chat):
        """Test that agent properly chains tool execution."""
        payload = {
            "sequence": "ATCGATCG---GCTAGCT",
            "gap_length": 150
        }
        
        # Mock the agent execution to simulate tool calling
        mock_executor_instance = Mock()
        mock_executor_instance.invoke.return_value = {
            "output": "Called context_tool, then gap_filler_tool. Result: ATCGATCGAAAGCTAGCT"
        }
        mock_executor.return_value = mock_executor_instance
        
        result = plan(payload)
        
        # Verify agent was created and executed
        mock_create_agent.assert_called_once()
        mock_executor_instance.invoke.assert_called_once()
        assert "output" in result


class TestPlannerTools:
    """Tests for planner tools."""
    
    @patch('src.agents.tools.planner_tools.GenomicRetriever')
    def test_context_tool_execution(self, mock_retriever_class):
        """Test context tool retrieval functionality."""
        # Mock retriever instance
        mock_retriever = Mock()
        mock_retriever.search.return_value = [
            {"sequence": "ATCGATCGATCG", "score": 0.9},
            {"sequence": "GCTAGCTAGCTA", "score": 0.8}
        ]
        mock_retriever_class.return_value = mock_retriever
        
        sequence = "ATCGATCG---GCTAGCT"
        context = context_tool(sequence)
        
        assert isinstance(context, str)
        assert len(context) > 0
        mock_retriever.search.assert_called_once_with(sequence, k=5)
    
    @patch('src.agents.tools.planner_tools.GapFiller')
    @patch('src.agents.tools.planner_tools.DNABert2Model')
    def test_gap_filler_tool_execution(self, mock_model_class, mock_filler_class):
        """Test gap filler tool execution."""
        # Mock model and filler
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        mock_filler = Mock()
        mock_filler.fill_gaps.return_value = "ATCGATCGAAAGCTAGCT"
        mock_filler_class.return_value = mock_filler
        
        sequence = "ATCGATCG---GCTAGCT"
        gap_length = 150
        
        result = gap_filler_tool(sequence, gap_length)
        
        assert result == "ATCGATCGAAAGCTAGCT"
        assert "---" not in result
        mock_filler.fill_gaps.assert_called_once_with(sequence)
    
    def test_context_tool_with_empty_sequence(self):
        """Test context tool with empty sequence."""
        with pytest.raises(ValueError):
            context_tool("")
    
    def test_gap_filler_tool_with_invalid_gap_length(self):
        """Test gap filler tool with invalid gap length."""
        sequence = "ATCGATCG---GCTAGCT"
        
        with pytest.raises(ValueError):
            gap_filler_tool(sequence, -1)  # Negative gap length
        
        with pytest.raises(ValueError):
            gap_filler_tool(sequence, 0)   # Zero gap length
    
    def test_context_tool_no_results(self):
        """Test context tool when no relevant context is found."""
        with patch('src.agents.tools.planner_tools.GenomicRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever.search.return_value = []  # No results
            mock_retriever_class.return_value = mock_retriever
            
            sequence = "ATCGATCG---GCTAGCT"
            context = context_tool(sequence)
            
            assert isinstance(context, str)
            assert "No relevant context found" in context or context == ""


class TestAgentIntegration:
    """Integration tests for agent components."""
    
    def test_agent_tool_coordination(self):
        """Test coordination between context and gap filler tools."""
        payload = {
            "sequence": "ATCGATCG---GCTAGCT",
            "gap_length": 150,
            "meta": {"organism": "E. coli"}
        }
        
        with patch('src.agents.tools.planner_tools.GenomicRetriever') as mock_retriever_class, \
             patch('src.agents.tools.planner_tools.GapFiller') as mock_filler_class, \
             patch('src.agents.tools.planner_tools.DNABert2Model'):
            
            # Mock context retrieval
            mock_retriever = Mock()
            mock_retriever.search.return_value = [
                {"sequence": "ATCGATCGATCG", "score": 0.9}
            ]
            mock_retriever_class.return_value = mock_retriever
            
            # Mock gap filling
            mock_filler = Mock()
            mock_filler.fill_gaps.return_value = "ATCGATCGAAAGCTAGCT"
            mock_filler_class.return_value = mock_filler
            
            # Test individual tools work
            context = context_tool(payload["sequence"])
            filled = gap_filler_tool(payload["sequence"], payload["gap_length"])
            
            assert isinstance(context, str)
            assert filled == "ATCGATCGAAAGCTAGCT"
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_agent_with_different_llm_backends(self):
        """Test agent with different LLM backends."""
        payload = {
            "sequence": "ATCGATCG---GCTAGCT",
            "gap_length": 150
        }
        
        # Test with different model configurations
        test_models = ["gpt-3.5-turbo", "gemini-pro", "claude-3-haiku"]
        
        for model_name in test_models:
            with patch('src.agents.planner.init_chat_model') as mock_init:
                mock_model = Mock()
                mock_init.return_value = mock_model
                
                with patch('src.agents.planner.create_tool_calling_agent'), \
                     patch('src.agents.planner.AgentExecutor') as mock_executor:
                    
                    mock_executor_instance = Mock()
                    mock_executor_instance.invoke.return_value = {"output": "Success"}
                    mock_executor.return_value = mock_executor_instance
                    
                    # Should work with different models
                    result = plan(payload)
                    assert result is not None
    
    def test_agent_error_handling(self):
        """Test agent error handling and recovery."""
        payload = {
            "sequence": "ATCGATCG---GCTAGCT",
            "gap_length": 150
        }
        
        with patch('src.agents.planner.init_chat_model'), \
             patch('src.agents.planner.create_tool_calling_agent'), \
             patch('src.agents.planner.AgentExecutor') as mock_executor:
            
            # Simulate agent execution error
            mock_executor_instance = Mock()
            mock_executor_instance.invoke.side_effect = Exception("LLM API error")
            mock_executor.return_value = mock_executor_instance
            
            with pytest.raises(Exception):
                plan(payload)
    
    def test_agent_timeout_handling(self):
        """Test agent timeout scenarios."""
        payload = {
            "sequence": "ATCGATCG---GCTAGCT",
            "gap_length": 150
        }
        
        with patch('src.agents.planner.init_chat_model'), \
             patch('src.agents.planner.create_tool_calling_agent'), \
             patch('src.agents.planner.AgentExecutor') as mock_executor:
            
            # Simulate timeout
            import time
            def slow_invoke(*args, **kwargs):
                time.sleep(0.1)  # Simulate slow response
                return {"output": "Delayed response"}
            
            mock_executor_instance = Mock()
            mock_executor_instance.invoke.side_effect = slow_invoke
            mock_executor.return_value = mock_executor_instance
            
            # Should handle timeout gracefully
            result = plan(payload)
            assert result is not None


@pytest.mark.integration
class TestAgentE2E:
    """End-to-end tests for agent system."""
    
    @pytest.mark.slow
    def test_complete_agent_pipeline(self):
        """Test complete agent pipeline with real components."""
        pytest.skip("E2E test - requires API keys and model downloads")
        
        # This would test:
        # 1. Real LLM API calls
        # 2. Real genomic data retrieval
        # 3. Real gap filling models
        # 4. Complete workflow from sequence to result
        pass


if __name__ == "__main__":
    pytest.main([__file__])