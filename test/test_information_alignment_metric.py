import pytest
from unittest.mock import MagicMock
from src.evaluation.information_alignment_metrics import (InformationAlignmentMetrics, 
    InformationChange,
    InformationChangeType)

class TestInformationAlignmentMetrics:
    @pytest.fixture
    def mock_gpt_interface(self):
        mock = MagicMock()
        mock.generate_response.return_value = {
            'response': """CHANGE
Type: removal
Information: medical details
Original: detailed blood test results
Edited: general health status"""
        }
        return mock

    @pytest.fixture
    def metrics(self, mock_gpt_interface):
        return InformationAlignmentMetrics(mock_gpt_interface)

    def test_initialization(self, metrics):
        assert len(metrics.evaluation_history) == 0
        assert metrics.gpt_interface is not None

    def test_information_change_creation(self):
        change = InformationChange(
            InformationChangeType.REMOVAL,
            "medical details",
            "detailed blood test results",
            "general health status"
        )
        assert change.change_type == "removal"
        assert change.information == "medical details"
        assert change.original == "detailed blood test results"
        assert change.edited == "general health status"

    def test_identify_information_changes(self, metrics):
        original = "Patient's blood pressure is 120/80"
        edited = "Patient's vitals are normal"
        context = {"user_instruction": "Provide health update"}
        
        changes = metrics.identify_information_changes(original, edited, context)
        assert len(changes) == 1
        assert isinstance(changes[0], InformationChange)

    def test_parse_changes(self, metrics):
        gpt_response = """CHANGE
Type: removal
Information: medical details
Original: blood pressure 120/80
Edited: vitals normal

CHANGE
Type: abstraction
Information: timing
Original: at 2pm yesterday
Edited: recently"""

        changes = metrics._parse_changes(gpt_response)
        assert len(changes) == 2
        assert changes[0].change_type == "removal"
        assert changes[1].change_type == "abstraction"

    def test_calculate_alignment_score(self, metrics):
        changes = [
            InformationChange("removal", "detail1"),
            InformationChange("addition", "detail2")
        ]
        score = metrics.calculate_alignment_score(changes)
        assert 0 <= score <= 1
        assert score == 1.0 / (1 + len(changes))

    def test_calculate_improvement(self, metrics):
        improvement = metrics.calculate_improvement(0.5, 0.75)
        assert improvement == 50.0  # (0.75 - 0.5) / 0.5 * 100

    def test_record_round(self, metrics):
        metrics.record_round(
            round_number=1,
            baseline_response="Original detailed response",
            learned_response="Learned detailed response",
            user_edit="Edited response",
            context={"instruction": "test"}
        )
        assert len(metrics.evaluation_history) == 1
        assert "round_number" in metrics.evaluation_history[0]
        assert "alignment_score" in metrics.evaluation_history[0]

    def test_generate_report(self, metrics, capsys):
        metrics.record_round(
            round_number=1,
            baseline_response="Original response",
            learned_response="Learned response",
            user_edit="Edited response",
            context={}
        )
        metrics.generate_report()
        captured = capsys.readouterr()
        assert "Round" in captured.out
        assert "Overall Statistics" in captured.out

    def test_empty_changes(self, metrics):
        score = metrics.calculate_alignment_score([])
        assert score == 1.0

    def test_zero_baseline_improvement(self, metrics):
        improvement = metrics.calculate_improvement(0, 0.5)
        assert improvement == 0.0

    @pytest.mark.parametrize("baseline,learned,expected", [
        (0.5, 0.75, 50.0),
        (0.2, 0.4, 100.0),
        (1.0, 1.0, 0.0)
    ])
    def test_improvement_calculations(self, metrics, baseline, learned, expected):
        improvement = metrics.calculate_improvement(baseline, learned)
        assert improvement == expected
