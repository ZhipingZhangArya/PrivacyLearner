import pytest
from unittest.mock import patch, MagicMock
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preference_learner import PreferenceLearner


class TestPreferenceLearner:
    @pytest.fixture
    def learner(self):
        return PreferenceLearner()

    @pytest.fixture
    def sample_gpt_analysis(self):
        return """
        Analysis of user's edits:
        Summary of user privacy preferences:
        {"medical_data", "share only with family"}
        {"location_data", "do not share"}
        """

    def test_initialization(self, learner):
        assert learner.learned_preferences == []
        assert learner.preference_history == []

    def test_extract_preferences(self, learner, sample_gpt_analysis):
        preferences = learner.extract_preferences(sample_gpt_analysis)
        assert len(preferences) == 2
        assert 'For "medical_data": "share only with family"' in preferences
        assert 'For "location_data": "do not share"' in preferences

    def test_add_preference(self, learner, sample_gpt_analysis):
        learner.add_preference(1, sample_gpt_analysis)
        assert len(learner.learned_preferences) == 1
        assert learner.learned_preferences[0]['scenario_number'] == 1
        assert isinstance(learner.learned_preferences[0]['preferences'], list)

    def test_get_current_preferences(self, learner):
        test_analysis = """
        Summary of user privacy preferences:
        {"health_status", "full disclosure"}
        """
        learner.add_preference(1, test_analysis)
        preferences = learner.get_current_preferences()
        assert len(preferences) == 1
        assert 'For "health_status": "full disclosure"' in preferences

    @pytest.mark.parametrize("marker", [
        "Summary of user privacy preferences:",
        "suggests the following privacy preferences:",
        "the user's justifications suggest the following privacy preferences:",
        "the user's privacy preferences in this context can be described as follows:"
    ])
    def test_different_preference_markers(self, learner, marker):
        analysis = f"""Some text
        {marker}
        {{"test_data", "no sharing"}}
        """
        preferences = learner.extract_preferences(analysis)
        assert len(preferences) == 1
        assert 'For "test_data": "no sharing"' in preferences



    def test_record_interaction(self, learner):
        learner.record_interaction(
            scenario_number=1,
            original="Original text",
            edited="Edited text",
            learned_preference="Test preference"
        )
        assert len(learner.preference_history) == 1
        assert learner.preference_history[0]['scenario_number'] == 1
        assert 'edit_distance' in learner.preference_history[0]

    def test_get_learning_progress_empty(self, learner):
        progress = learner.get_learning_progress()
        assert progress['total_interactions'] == 0
        assert progress['average_edit_distance'] == 0.0
        assert progress['preference_count'] == 0

    def test_reset(self, learner, sample_gpt_analysis):
        learner.add_preference(1, sample_gpt_analysis)
        learner.record_interaction(1, "Original", "Edited", "Preference")
        learner.reset()
        assert len(learner.learned_preferences) == 0
        assert len(learner.preference_history) == 0

    def test_get_preference_by_scenario(self, learner, sample_gpt_analysis):
        learner.add_preference(1, sample_gpt_analysis)
        preference = learner.get_preference_by_scenario(1)
        assert preference == sample_gpt_analysis
        assert learner.get_preference_by_scenario(999) is None
