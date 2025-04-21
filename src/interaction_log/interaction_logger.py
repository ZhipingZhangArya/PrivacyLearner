import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

class InteractionLogger:
    def __init__(self, mode: str):
        """
        Initialize logger with mode information
        Args:
            mode: The learning mode being used ('basic' or 'reasoning')
        """
        self.mode = mode
        self.interactions = []
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
    def log_interaction(self, round_number: int, context_number: int, 
                   scenario_data: Dict, base_prompt: str, 
                   complete_prompt: str, privacy_preferences: List[str],
                   baseline_response: str, learned_response: str, 
                   user_edit: str, learned_preference: str,
                   prompt_tokens: int = 0,
                   completion_tokens: int = 0):
                   #follow_up_qa: List[Tuple[str, str]] = None):
        """
        Log a single interaction round
        Args:
            ...existing parameters...
            mode_specific_data: Additional data specific to the mode (e.g., justifications in reasoning mode)
        """
        interaction = {
            'round_number': round_number,
            'context_number': context_number,
            'timestamp': datetime.now().isoformat(),
            'user_instruction': scenario_data['user_instruction'],
            'tools_used': scenario_data['toolkits'],
            'base_prompt': base_prompt,
            'privacy_preferences_used': privacy_preferences,
            'complete_prompt': complete_prompt,
            'baseline_response': baseline_response,
            'learned_response': learned_response,
            'user_edit': user_edit,
            'learned_preference': learned_preference,
            'prompt_tokens': prompt_tokens,          
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
            #'follow_up_qa': follow_up_qa if follow_up_qa else []

        }
        self.interactions.append(interaction)
    
    def save_log(self):
        """Save interactions to a JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'interaction_log_{self.mode}_{timestamp}.json'
        filepath = os.path.join(self.log_dir, filename)
        
        log_data = {
            'mode': self.mode,
            'mode_description': self._get_mode_description(),
            'timestamp': datetime.now().isoformat(),
            'total_rounds': len(self.interactions),
            'interactions': self.interactions
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
            
        return filepath

    def _get_mode_description(self) -> str:
        """Get description of the current mode"""
        if self.mode == 'basic':
            return "Basic mode: Direct comparison and preference learning without user justification"
        elif self.mode == 'reasoning':
            return "Reasoning mode: Interactive preference learning with user justification for changes"
        return "Unknown mode"

    def get_interactions(self) -> List[Dict]:
        """Get all recorded interactions"""
        return self.interactions.copy()