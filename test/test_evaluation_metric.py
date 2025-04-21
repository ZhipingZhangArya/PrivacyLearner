import pytest
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.edit_distance_metrics import EvaluationMetrics

class TestEvaluationMetrics:
    @pytest.fixture
    def mock_gpt_interface(self):
        return MagicMock()

    @pytest.fixture
    def evaluator(self, mock_gpt_interface):
        return EvaluationMetrics(mock_gpt_interface)

    def test_calculate_edit_distance(self, evaluator):
        response1 = "hello world"
        response2 = "hello there world"
        distance = evaluator.calculate_edit_distance(response1, response2)
        assert 0 <= distance <= 1
        assert distance > 0  # Should have some difference

    def test_levenshtein_distance(self, evaluator):
        tokens1 = ["hello", "world"]
        tokens2 = ["hello", "there", "world"]
        distance = evaluator._levenshtein_distance(tokens1, tokens2)
        assert distance == 1  # One insertion needed

    def test_record_round(self, evaluator):
        evaluator.record_round(
            round_number=1,
            baseline_response="hello world",
            learned_response="hello there world",
            user_edit="hello beautiful world",
            context={"test": "context"}
        )
        assert len(evaluator.evaluation_history) == 1
        metrics = evaluator.evaluation_history[0]
        assert metrics['round_number'] == 1
        assert 'distance' in metrics
        assert 'distance_absolute' in metrics

    def test_identical_responses(self, evaluator):
        response = "hello world"
        distance = evaluator.calculate_edit_distance(response, response)
        assert distance == 0  # Should be zero for identical responses

    def test_completely_different_responses(self, evaluator):
        response1 = "hello world"
        response2 = "goodbye universe"
        distance = evaluator.calculate_edit_distance(response1, response2)
        assert distance == 1.0  # Should be maximum distance

    def test_generate_report_empty(self, evaluator, capsys):
        evaluator.generate_report()
        captured = capsys.readouterr()
        assert "Token-based Edit Distance Metrics" in captured.out
        assert "Round  |  With Preferences  |  Without Preferences  |  Improvement" in captured.out

    
