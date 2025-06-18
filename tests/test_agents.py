"""
Tests for the agents module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents import BaseAgent, RAGAgent
from src.rag import BaseRAGPipeline


class TestBaseAgent:
    """Test BaseAgent functionality"""

    @patch("src.agents.hf_pipeline")
    def test_init(self, mock_hf_pipeline):
        """Test BaseAgent initialization"""
        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_gen.tokenizer = mock_tokenizer
        mock_hf_pipeline.return_value = mock_gen

        system_prompt = "You are a helpful assistant"
        agent = BaseAgent(system_prompt)

        assert agent.system_prompt == system_prompt
        assert agent.history == []
        assert agent._gen == mock_gen
        assert agent._tokenizer == mock_tokenizer

    @patch("src.agents.hf_pipeline")
    def test_observe_valid_roles(self, mock_hf_pipeline):
        """Test observe method with valid roles"""
        mock_gen = Mock()
        mock_gen.tokenizer = Mock()
        mock_hf_pipeline.return_value = mock_gen

        agent = BaseAgent("Test prompt")

        agent.observe("Hello", "user")
        agent.observe("Hi there", "assistant")

        assert len(agent.history) == 2
        assert agent.history[0] == ("Hello", "user")
        assert agent.history[1] == ("Hi there", "assistant")

    @patch("src.agents.hf_pipeline")
    def test_build_prompt(self, mock_hf_pipeline):
        """Test _build_prompt functionality"""
        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_gen.tokenizer = mock_tokenizer
        mock_hf_pipeline.return_value = mock_gen

        agent = BaseAgent("System prompt")

        result = agent._build_prompt("User content")

        assert result == "formatted prompt"
        mock_tokenizer.apply_chat_template.assert_called_once()

        # Check the messages structure
        call_args = mock_tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User content"

    @patch("src.agents.hf_pipeline")
    def test_build_prompt_non_string_result(self, mock_hf_pipeline):
        """Test _build_prompt when template returns non-string"""
        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = 123  # Non-string
        mock_gen.tokenizer = mock_tokenizer
        mock_hf_pipeline.return_value = mock_gen

        agent = BaseAgent("System prompt")

        result = agent._build_prompt("User content")

        assert result == "123"  # Should be converted to string

    @patch("src.agents.hf_pipeline")
    def test_generate_success(self, mock_hf_pipeline):
        """Test successful _generate call"""
        mock_gen = Mock()
        mock_gen.return_value = [{"generated_text": "  Generated response  "}]
        mock_tokenizer = Mock()
        mock_gen.tokenizer = mock_tokenizer
        mock_hf_pipeline.return_value = mock_gen

        agent = BaseAgent("System prompt")

        result = agent._generate("test prompt")

        assert result == "Generated response"  # Should be stripped
        mock_gen.assert_called_once_with("test prompt")

    @patch("src.agents.hf_pipeline")
    def test_generate_unexpected_format(self, mock_hf_pipeline):
        """Test _generate with unexpected output format"""
        mock_gen = Mock()
        mock_gen.return_value = [{"unexpected_key": "value"}]
        mock_tokenizer = Mock()
        mock_gen.tokenizer = mock_tokenizer
        mock_hf_pipeline.return_value = mock_gen

        agent = BaseAgent("System prompt")

        result = agent._generate("test prompt")

        assert "Unexpected output format" in result

    @patch("src.agents.hf_pipeline")
    def test_generate_empty_response(self, mock_hf_pipeline):
        """Test _generate with empty response"""
        mock_gen = Mock()
        mock_gen.return_value = []
        mock_tokenizer = Mock()
        mock_gen.tokenizer = mock_tokenizer
        mock_hf_pipeline.return_value = mock_gen

        agent = BaseAgent("System prompt")

        result = agent._generate("test prompt")

        assert "Unexpected output format" in result

    @patch("src.agents.hf_pipeline")
    def test_generate_exception_handling(self, mock_hf_pipeline):
        """Test _generate exception handling"""
        mock_gen = Mock()
        mock_gen.side_effect = Exception("Generation failed")
        mock_tokenizer = Mock()
        mock_gen.tokenizer = mock_tokenizer
        mock_hf_pipeline.return_value = mock_gen

        agent = BaseAgent("System prompt")

        result = agent._generate("test prompt")

        assert "Error generating response: Generation failed" in result


class TestRAGAgent:
    """Test RAGAgent functionality"""

    @patch("src.agents.hf_pipeline")
    def test_init(self, mock_hf_pipeline):
        """Test RAGAgent initialization"""
        mock_gen = Mock()
        mock_gen.tokenizer = Mock()
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)

        agent = RAGAgent(mock_rag_pipeline)

        assert agent.rag == mock_rag_pipeline
        assert "Captain Pilot" in agent.system_prompt
        assert "aviation expert" in agent.system_prompt

    @patch("src.agents.hf_pipeline")
    def test_format_chat_history_empty(self, mock_hf_pipeline):
        """Test _format_chat_history with empty history"""
        mock_gen = Mock()
        mock_gen.tokenizer = Mock()
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)
        agent = RAGAgent(mock_rag_pipeline)

        result = agent._format_chat_history()

        assert result == "No previous conversation."

    @patch("src.agents.hf_pipeline")
    def test_format_chat_history_with_messages(self, mock_hf_pipeline):
        """Test _format_chat_history with messages"""
        mock_gen = Mock()
        mock_gen.tokenizer = Mock()
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)
        agent = RAGAgent(mock_rag_pipeline)

        # Add some history
        agent.history = [
            ("Question 1", "user"),
            ("Answer 1", "assistant"),
            ("Question 2", "user"),
            ("Answer 2", "assistant"),
            ("Question 3", "user"),
        ]

        result = agent._format_chat_history()

        # Should keep last 3 exchanges (only last 3)
        assert "<|user|> Question 2" in result
        assert "<|assistant|> Answer 2" in result
        assert "<|user|> Question 3" in result
        # Should not include the first exchange
        assert "<|user|> Question 1" not in result

    @patch("src.agents.hf_pipeline")
    def test_format_chat_history_clean_tags(self, mock_hf_pipeline):
        """Test _format_chat_history cleans existing tags"""
        mock_gen = Mock()
        mock_gen.tokenizer = Mock()
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)
        agent = RAGAgent(mock_rag_pipeline)

        # Add history with existing tags
        agent.history = [
            ("<|user|> Clean this", "user"),
            ("<|assistant|> And this", "assistant"),
        ]

        result = agent._format_chat_history()

        # Should clean existing tags and add proper ones
        assert "<|user|> Clean this" in result
        assert "<|assistant|> And this" in result
        # Should not have double tags
        assert "<|user|> <|user|>" not in result

    @patch("src.agents.hf_pipeline")
    def test_act_no_history(self, mock_hf_pipeline):
        """Test act method with no history"""
        mock_gen = Mock()
        mock_gen.tokenizer = Mock()
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)
        agent = RAGAgent(mock_rag_pipeline)

        with pytest.raises(ValueError, match="No query in history to answer"):
            agent.act()

    @patch("src.agents.hf_pipeline")
    def test_act_with_sources(self, mock_hf_pipeline):
        """Test act method with sources"""
        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_gen.tokenizer = mock_tokenizer
        mock_gen.return_value = [{"generated_text": "Generated answer about aviation"}]
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)
        mock_rag_pipeline.run.return_value = (
            ["Aviation is the science of flight"],
            [1, 2],
        )

        agent = RAGAgent(mock_rag_pipeline)
        agent.observe("What is aviation?", "user")

        result = agent.act()

        mock_rag_pipeline.run.assert_called_once_with("What is aviation?")
        assert "📚 Sources: Based on 2 relevant document sections" in result

        # Check that response was added to history
        assert len(agent.history) == 2
        assert agent.history[1][1] == "assistant"

    @patch("src.agents.hf_pipeline")
    def test_act_no_sources(self, mock_hf_pipeline):
        """Test act method with no sources"""
        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_gen.tokenizer = mock_tokenizer
        mock_gen.return_value = [
            {"generated_text": "I don't have specific information"}
        ]
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)
        mock_rag_pipeline.run.return_value = (["No specific context available."], [])

        agent = RAGAgent(mock_rag_pipeline)
        agent.observe("What is quantum computing?", "user")

        result = agent.act()

        # Should not include sources section
        assert "Sources:" not in result

    @patch("src.agents.hf_pipeline")
    def test_pretty_print_answer_assistant_tag(self, mock_hf_pipeline):
        """Test pretty_print_answer with assistant tag"""
        mock_gen = Mock()
        mock_gen.tokenizer = Mock()
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)
        agent = RAGAgent(mock_rag_pipeline)

        result = agent.pretty_print_answer("Some text <|assistant|> Clean answer")

        assert result == "Clean answer"

    @patch("src.agents.hf_pipeline")
    def test_pretty_print_answer_final_answer(self, mock_hf_pipeline):
        """Test pretty_print_answer with Final answer tag"""
        mock_gen = Mock()
        mock_gen.tokenizer = Mock()
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)
        agent = RAGAgent(mock_rag_pipeline)

        result = agent.pretty_print_answer("Thinking... Final answer: Clean answer")

        assert result == ": Clean answer"

    @patch("src.agents.hf_pipeline")
    def test_pretty_print_answer_refined_answer(self, mock_hf_pipeline):
        """Test pretty_print_answer with Refined answer tag"""
        mock_gen = Mock()
        mock_gen.tokenizer = Mock()
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)
        agent = RAGAgent(mock_rag_pipeline)

        result = agent.pretty_print_answer("Analysis... Refined answer: Clean answer")

        assert result == ": Clean answer"

    @patch("src.agents.hf_pipeline")
    def test_pretty_print_answer_no_tags(self, mock_hf_pipeline):
        """Test pretty_print_answer with no special tags"""
        mock_gen = Mock()
        mock_gen.tokenizer = Mock()
        mock_hf_pipeline.return_value = mock_gen

        mock_rag_pipeline = Mock(spec=BaseRAGPipeline)
        agent = RAGAgent(mock_rag_pipeline)

        result = agent.pretty_print_answer("Regular answer text")

        assert result == "Regular answer text"
