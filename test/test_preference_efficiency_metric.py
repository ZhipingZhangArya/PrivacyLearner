import pytest
from unittest.mock import MagicMock
from src.evaluation.preference_efficiency_metrics import PreferenceEfficiencyMetrics

class TestPreferenceEfficiencyMetrics:
    @pytest.fixture
    def mock_gpt_interface(self):
        return MagicMock()

    @pytest.fixture
    def metrics(self, mock_gpt_interface):
        return PreferenceEfficiencyMetrics(mock_gpt_interface)

    @pytest.fixture
    def sample_context(self):
        return {
            "user_instruction": "Test instruction"
        }

    def test_initialization(self, metrics):
        assert len(metrics.evaluation_history) == 0
        assert metrics.gpt_interface is not None

    def test_record_round(self, metrics, sample_context):
        metrics.record_round(
            round_number=1,
            context=sample_context,
            privacy_preferences=["preference one", "preference two"],
            prompt_tokens=100,
            completion_tokens=50
        )
        
        assert len(metrics.evaluation_history) == 1
        record = metrics.evaluation_history[0]
        assert record['round_number'] == 1
        assert record['num_preferences'] == 2
        assert record['preference_tokens'] == 4  # "preference one" (2) + "preference two" (2)
        assert record['prompt_tokens'] == 100
        assert record['completion_tokens'] == 50
        assert record['total_tokens'] == 150

    def test_record_multiple_rounds(self, metrics, sample_context):
        for i in range(3):
            metrics.record_round(
                round_number=i+1,
                context=sample_context,
                privacy_preferences=["test preference"] * (i+1),
                prompt_tokens=100,
                completion_tokens=50
            )
        
        assert len(metrics.evaluation_history) == 3
        assert metrics.evaluation_history[0]['num_preferences'] == 1
        assert metrics.evaluation_history[1]['num_preferences'] == 2
        assert metrics.evaluation_history[2]['num_preferences'] == 3

    def test_generate_report_empty(self, metrics, capsys):
        metrics.generate_report()
        captured = capsys.readouterr()
        assert "Preference Growth and Efficiency Per Round:" in captured.out
        assert "Overall Statistics:" not in captured.out

    def test_generate_report_with_data(self, metrics, sample_context, capsys):
        metrics.record_round(
            round_number=1,
            context=sample_context,
            privacy_preferences=["test preference"],
            prompt_tokens=100,
            completion_tokens=50
        )
        
        metrics.generate_report()
        captured = capsys.readouterr()
        assert "Round  1" in captured.out
        assert "Overall Statistics:" in captured.out
        assert "Average preferences per round:" in captured.out

    def test_token_calculations(self, metrics, sample_context):
        metrics.record_round(
            round_number=1,
            context=sample_context,
            privacy_preferences=["long preference with many words", "short one"],
            prompt_tokens=100,
            completion_tokens=50
        )
        
        record = metrics.evaluation_history[0]
        assert record['preference_tokens'] == 7  # 5 words + 2 words
        assert record['total_tokens'] == 150

    @pytest.mark.parametrize("preferences,expected_tokens", [
        ([], 0),
        (["single"], 1),
        (["two words"], 2),
        (["one", "two", "three"], 3)
    ])
    def test_preference_token_counting(self, metrics, sample_context, preferences, expected_tokens):
        metrics.record_round(
            round_number=1,
            context=sample_context,
            privacy_preferences=preferences,
            prompt_tokens=100,
            completion_tokens=50
        )
        
        assert metrics.evaluation_history[0]['preference_tokens'] == expected_tokens
