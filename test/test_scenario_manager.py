import pytest
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scenario_manager import ScenarioManager
from src.data_loader import DataLoader

class TestScenarioManager:
    @pytest.fixture
    def mock_data_loader(self):
        data_loader = MagicMock(spec=DataLoader)
        data_loader.get_scenario_count.return_value = 3
        data_loader.get_scenario_by_number.side_effect = lambda x: {
            1: {"user_instruction": "Test 1", "toolkits": ["Tool1"], "executable_trajectory": "Action1"},
            2: {"user_instruction": "Test 2", "toolkits": ["Tool2"], "executable_trajectory": "Action2"},
            3: {"user_instruction": "Test 3", "toolkits": ["Tool3"], "executable_trajectory": "Action3"}
        }.get(x)
        return data_loader

    @pytest.fixture
    def scenario_manager(self, mock_data_loader):
        return ScenarioManager(mock_data_loader)

    def test_initialization(self, scenario_manager):
        assert scenario_manager.current_scenario == 1
        assert scenario_manager.total_scenarios == 3
        assert scenario_manager.used_scenarios == []

    def test_get_next_scenario(self, scenario_manager):
        scenario_number, scenario_data = scenario_manager.get_next_scenario()
        assert scenario_number == 1
        assert scenario_data["user_instruction"] == "Test 1"
        assert len(scenario_manager.used_scenarios) == 1

    def test_get_next_scenario_sequence(self, scenario_manager):
        # Get all scenarios in sequence
        scenarios = []
        for _ in range(3):
            number, data = scenario_manager.get_next_scenario()
            scenarios.append((number, data))
        
        assert [num for num, _ in scenarios] == [1, 2, 3]
        assert len(scenario_manager.used_scenarios) == 3

    def test_no_more_scenarios(self, scenario_manager):
        # Use all scenarios
        for _ in range(3):
            scenario_manager.get_next_scenario()
            
        with pytest.raises(Exception, match="No more scenarios available"):
            scenario_manager.get_next_scenario()

    def test_prepare_scenario_for_gpt(self, scenario_manager):
        scenario_data = {
            "user_instruction": "Test instruction",
            "toolkits": ["Tool1", "Tool2"],
            "executable_trajectory": "Test trajectory"
        }
        
        prompt = scenario_manager.prepare_scenario_for_gpt(scenario_data)
        assert "Test instruction" in prompt
        assert "Tool1, Tool2" in prompt
        assert "Test trajectory" in prompt

    def test_reset(self, scenario_manager):
        # Use some scenarios
        scenario_manager.get_next_scenario()
        scenario_manager.get_next_scenario()
        
        # Reset
        scenario_manager.reset()
        assert scenario_manager.current_scenario == 1
        assert scenario_manager.used_scenarios == []
        
        # Verify can get scenarios again
        number, data = scenario_manager.get_next_scenario()
        assert number == 1
        assert data["user_instruction"] == "Test 1"

    def test_get_remaining_count(self, scenario_manager):
        assert scenario_manager.get_remaining_count() == 3
        
        scenario_manager.get_next_scenario()
        assert scenario_manager.get_remaining_count() == 2
        
        scenario_manager.get_next_scenario()
        assert scenario_manager.get_remaining_count() == 1

    def test_get_used_scenarios(self, scenario_manager):
        assert scenario_manager.get_used_scenarios() == []
        
        scenario_manager.get_next_scenario()
        assert scenario_manager.get_used_scenarios() == [1]
        
        scenario_manager.get_next_scenario()
        assert scenario_manager.get_used_scenarios() == [1, 2]
