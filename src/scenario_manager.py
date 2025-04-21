"""
Module: scenario_manager
========================
This module defines the `ScenarioManager` class, which handles scenarios loaded from the `DataLoader` object. 
It provides functionality to manage scenarios in sequence, prepare them for GPT models, reset progress, and track usage.
"""

import random
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import DataLoader


class ScenarioManager:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        
        # we allow randomize. but for evaluating the performance, just use the same sequences
        ## track which scenarios have been used
        #self.available_scenario_numbers = list(range(1, data_loader.get_scenario_count() + 1))
        #print(f"ScenarioManager initialized with {len(self.available_scenario_numbers)} available scenarios")
        
        self.current_scenario = 1
        self.total_scenarios = data_loader.get_scenario_count()
        # track which scenarios have been used
        self.used_scenarios = []
        print(f"ScenarioManager initialized with {self.total_scenarios} scenarios")
    
    # randomize scenario (for evaluating, we don't randomize it)
    #def get_random_scenario(self) -> Tuple[int, Dict]:
    #    """
    #    Randomly select an unused scenario and prepare it
        
    #    Returns:
    #        Tuple[int, Dict]: (main_number, scenario_data)
    #    """
    #    if not self.available_scenario_numbers:
    #        raise Exception("No more scenarios available")
        
    #    # randomly select a scenario number
    #    selected_number = random.choice(self.available_scenario_numbers)
    #   self.available_scenario_numbers.remove(selected_number)

    #    scenario_data = self.data_loader.get_scenario_by_number(selected_number)
        
    #    print(f"\nSelected scenario {selected_number}")
    #    print(f"Remaining scenarios: {len(self.available_scenario_numbers)}")
        
    #    return selected_number, scenario_data

    def get_next_scenario(self) -> Tuple[int, Dict]:
        """
        Get next scenario in sequence
        
        Returns:
            Tuple[int, Dict]: (main_number, scenario_data)
        """
        if self.current_scenario > self.total_scenarios:
            raise Exception("No more scenarios available")
        
        scenario_data = self.data_loader.get_scenario_by_number(self.current_scenario)
        self.used_scenarios.append(self.current_scenario)
        
        print(f"\nSelected scenario {self.current_scenario}")
        print(f"Remaining scenarios: {self.total_scenarios - len(self.used_scenarios)}")
        
        # Store current scenario before incrementing
        current = self.current_scenario
        self.current_scenario += 1
        
        return current, scenario_data
    
    
    def prepare_scenario_for_gpt(self, scenario_data: Dict) -> str:
        """
        prompt to generate response based on the scenario materials
        """
        prompt = f"""
        Please provide response based on user instruction: {scenario_data['user_instruction']}
        The tools you can use: {', '.join(scenario_data['toolkits'])}
        Here's the possible related information you observe from the user's database: {scenario_data['executable_trajectory']}
        Response:"""
        return prompt
    
    def reset(self):
        """Reset the available scenarios"""
        #self.available_scenario_numbers = list(range(1, self.data_loader.get_scenario_count() + 1))
        #print(f"Reset available scenarios to {len(self.available_scenario_numbers)} scenarios")

        self.current_scenario = 1
        self.used_scenarios = []
        print(f"Reset scenarios to start from beginning. Total scenarios: {self.total_scenarios}")

    
    def get_remaining_count(self) -> int:
        """Get count of remaining unused scenarios"""
        #return len(self.available_scenario_numbers)
        return self.total_scenarios - len(self.used_scenarios)
    
    def get_used_scenarios(self) -> List[int]:
        """Get list of scenario numbers that have been used"""
        #all_scenarios = set(range(1, self.data_loader.get_scenario_count() + 1))
        #available = set(self.available_scenario_numbers)
        #return list(all_scenarios - available)
        return self.used_scenarios.copy()