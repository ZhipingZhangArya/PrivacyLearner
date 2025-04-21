import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from src.interaction_log.interaction_logger import InteractionLogger

class TestInteractionLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        logger = InteractionLogger("basic")
        logger.log_dir = tmp_path
        return logger

    @pytest.fixture
    def sample_scenario_data(self):
        return {
            "user_instruction": "Test instruction",
            "toolkits": ["Tool1", "Tool2"]
        }

    def test_initialization(self, logger):
        assert logger.mode == "basic"
        assert isinstance(logger.interactions, list)
        assert len(logger.interactions) == 0
        assert os.path.exists(logger.log_dir)

    def test_log_interaction(self, logger, sample_scenario_data):
        logger.log_interaction(
            round_number=1,
            context_number=1,
            scenario_data=sample_scenario_data,
            base_prompt="test prompt",
            complete_prompt="complete test prompt",
            privacy_preferences=["pref1", "pref2"],
            baseline_response="baseline",
            learned_response="learned",
            user_edit="edited",
            learned_preference="test preference",
            prompt_tokens=100,
            completion_tokens=50
        )
        
        assert len(logger.interactions) == 1
        interaction = logger.interactions[0]
        assert interaction["round_number"] == 1
        assert interaction["context_number"] == 1
        assert interaction["user_instruction"] == "Test instruction"
        assert interaction["tools_used"] == ["Tool1", "Tool2"]
        assert interaction["prompt_tokens"] == 100
        assert interaction["completion_tokens"] == 50
        assert interaction["total_tokens"] == 150

    def test_save_log(self, logger, sample_scenario_data):
        logger.log_interaction(
            round_number=1,
            context_number=1,
            scenario_data=sample_scenario_data,
            base_prompt="test prompt",
            complete_prompt="complete test prompt",
            privacy_preferences=["pref1"],
            baseline_response="baseline",
            learned_response="learned",
            user_edit="edited",
            learned_preference="test preference",
            prompt_tokens=100,
            completion_tokens=50
        )
        
        filepath = logger.save_log()
        assert os.path.exists(filepath)
        
        with open(filepath, 'r') as f:
            log_data = json.load(f)
            
        assert log_data["mode"] == "basic"
        assert log_data["total_rounds"] == 1
        assert len(log_data["interactions"]) == 1
        assert "Basic mode:" in log_data["mode_description"]

    def test_get_mode_description(self):
        basic_logger = InteractionLogger("basic")
        reasoning_logger = InteractionLogger("reasoning")
        unknown_logger = InteractionLogger("unknown")
        
        assert "Basic mode:" in basic_logger._get_mode_description()
        assert "Reasoning mode:" in reasoning_logger._get_mode_description()
        assert "Unknown mode" == unknown_logger._get_mode_description()


    def test_multiple_interactions(self, logger, sample_scenario_data):
        for i in range(3):
            logger.log_interaction(
                round_number=i+1,
                context_number=i+1,
                scenario_data=sample_scenario_data,
                base_prompt=f"test prompt {i}",
                complete_prompt=f"complete test prompt {i}",
                privacy_preferences=[],
                baseline_response=f"baseline {i}",
                learned_response=f"learned {i}",
                user_edit=f"edited {i}",
                learned_preference=f"test preference {i}"
            )
        
        assert len(logger.interactions) == 3
        assert logger.interactions[0]["round_number"] == 1
        assert logger.interactions[2]["round_number"] == 3
