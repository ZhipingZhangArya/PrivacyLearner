import pytest
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gpt_interface import GPTInterface

class TestGPTInterface:
    @pytest.fixture
    def mock_openai_response(self):
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Hi Mark, thanks for checking in. The ultrasound went really well..."))
        ]
        return mock_response

    @pytest.fixture
    def gpt_interface(self):
        return GPTInterface()

    @patch('openai.OpenAI')
    def test_generate_response_success(self, mock_openai, gpt_interface):
        mock_client = Mock()
        mock_client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="Hi Mark, thanks for checking in..."))
        ]
        mock_openai.return_value = mock_client

        result = gpt_interface.generate_response("Test prompt")
        assert isinstance(result, dict)
        assert 'response' in result
        assert 'complete_prompt' in result

    @patch('openai.OpenAI')
    def test_generate_response_with_preferences(self, mock_openai, gpt_interface):
        mock_client = Mock()
        mock_client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="Hi Mark, thanks for checking in..."))
        ]
        mock_openai.return_value = mock_client

        preferences = ["For medical_data: abstract disclosure"]
        result = gpt_interface.generate_response("Test prompt", preferences)
        assert isinstance(result, dict)
        assert "medical_data" in result['complete_prompt']

    @patch('openai.OpenAI')
    def test_learn_preferences(self, mock_openai, gpt_interface):
        mock_client = Mock()
        mock_client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="""
            Summary of user privacy preferences:
            {"medical_data", "abstract disclosure"}
            """))
        ]
        mock_openai.return_value = mock_client

        result = gpt_interface.learn_preferences(
            "Original response",
            "Edited response",
            {"test": "context"}
        )
        assert "Summary of user privacy preferences" in result

    @patch('openai.OpenAI')
    def test_baseline_response_generation(self, mock_openai, gpt_interface):
        mock_client = Mock()
        mock_client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="Hi Mark, thanks for checking in..."))
        ]
        mock_openai.return_value = mock_client

        result = gpt_interface.generate_response(
            "Test prompt",
            privacy_preferences=["Pref1"],
            is_baseline=True
        )
        assert isinstance(result, dict)
        assert "Pref1" not in result['complete_prompt']
