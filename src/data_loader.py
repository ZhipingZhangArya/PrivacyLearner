"""
Module: data_loader
===================
This module defines the `DataLoader` class, which loads and manages scenario data from a JSON file. 
It provides methods to access all scenarios, retrieve specific scenarios by their number, 
and get the total count of scenarios.
"""

import os
import json
from typing import List, Dict

class DataLoader:
    def __init__(self):
        # init
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(project_root, 'context_data', 'sanitized_sanbox_data_trajectory.json')
        self.scenarios = self._load_scenarios()
    
    def _load_scenarios(self) -> List[Dict]:
        """
        Load scenarios from JSON file
        Returns:
            List[Dict]: List of scenario dictionaries, where each dictionary contains:
                - main_number
                - user_name
                - user_email
                - user_instruction
                - toolkits
                - executable_trajectory
                - final_action
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
                print(f"Successfully loaded {len(scenarios)} scenarios:")
                
                # debug
                if scenarios:
                    print("First scenario keys:", list(scenarios[0].keys()))
                    for i, scenario in enumerate(scenarios):
                        main_number = scenario.get('main_number', i)
                        print(f"- Main {main_number}: {len(scenario['trajectory'].keys())} fields")
                
                return scenarios
                
        except FileNotFoundError:
            print(f"Error: {self.data_path} not found")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {self.data_path}")
            return []
    
    def get_all_scenarios(self) -> List[Dict]:
        return self.scenarios

    def get_scenario_by_number(self, main_number: int) -> Dict:
        """Get a specific scenario by its main number"""
        for scenario in self.scenarios:
            if scenario.get('main_number') == main_number:
                return scenario['trajectory']
        return None
    
    def get_scenario_count(self) -> int:
        return len(self.scenarios)