import pytest
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import DataLoader


class TestDataLoader:
    @pytest.fixture
    def sample_scenarios(self):
        return [
            {
                "main_number": 1,
                "trajectory": {
                    "user_name": "Test User",
                    "user_email": "test@example.com",
                    "user_instruction": "Test instruction",
                    "toolkits": ["Tool1", "Tool2"],
                    "executable_trajectory": "Test trajectory",
                    "final_action": "Test action"
                }
            },
            {
                "main_number": 2,
                "trajectory": {
                    "user_name": "Test User 2",
                    "toolkits": ["Tool3"]
                }
            }
        ]

    @pytest.fixture
    def mock_data_file(self, tmp_path, sample_scenarios):
        data_dir = tmp_path / "context_data"
        data_dir.mkdir()
        data_file = data_dir / "sanitized_sanbox_data_trajectory.json"
        data_file.write_text(json.dumps(sample_scenarios))
        return data_file

    def test_initialization(self, mock_data_file, monkeypatch):
        # Create a subclass of DataLoader for testing
        class TestableDataLoader(DataLoader):
            def __init__(self, data_path):
                self.data_path = data_path
                self.scenarios = self._load_scenarios()

        loader = TestableDataLoader(str(mock_data_file))
        assert isinstance(loader.scenarios, list)
        assert len(loader.scenarios) == 2

    def test_get_all_scenarios(self, mock_data_file, sample_scenarios):
        class TestableDataLoader(DataLoader):
            def __init__(self, data_path):
                self.data_path = data_path
                self.scenarios = self._load_scenarios()

        loader = TestableDataLoader(str(mock_data_file))
        scenarios = loader.get_all_scenarios()
        assert scenarios == sample_scenarios
        assert len(scenarios) == 2

    def test_get_scenario_by_number(self, mock_data_file):
        class TestableDataLoader(DataLoader):
            def __init__(self, data_path):
                self.data_path = data_path
                self.scenarios = self._load_scenarios()

        loader = TestableDataLoader(str(mock_data_file))
        scenario = loader.get_scenario_by_number(1)
        assert scenario["user_name"] == "Test User"
        assert scenario["toolkits"] == ["Tool1", "Tool2"]

    def test_get_scenario_by_number_not_found(self, mock_data_file):
        class TestableDataLoader(DataLoader):
            def __init__(self, data_path):
                self.data_path = data_path
                self.scenarios = self._load_scenarios()

        loader = TestableDataLoader(str(mock_data_file))
        scenario = loader.get_scenario_by_number(999)
        assert scenario is None

    def test_get_scenario_count(self, mock_data_file):
        class TestableDataLoader(DataLoader):
            def __init__(self, data_path):
                self.data_path = data_path
                self.scenarios = self._load_scenarios()

        loader = TestableDataLoader(str(mock_data_file))
        assert loader.get_scenario_count() == 2

    def test_file_not_found(self, tmp_path):
        class TestableDataLoader(DataLoader):
            def __init__(self, data_path):
                self.data_path = data_path
                self.scenarios = self._load_scenarios()

        nonexistent_file = tmp_path / "nonexistent.json"
        loader = TestableDataLoader(str(nonexistent_file))
        assert loader.get_scenario_count() == 0
        assert loader.get_all_scenarios() == []

    def test_invalid_json(self, tmp_path):
        class TestableDataLoader(DataLoader):
            def __init__(self, data_path):
                self.data_path = data_path
                self.scenarios = self._load_scenarios()

        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("invalid json content")
        loader = TestableDataLoader(str(invalid_file))
        assert loader.get_scenario_count() == 0
        assert loader.get_all_scenarios() == []

