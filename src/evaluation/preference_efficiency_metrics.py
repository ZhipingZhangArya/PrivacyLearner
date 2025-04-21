from typing import List, Dict
import time

class PreferenceEfficiencyMetrics:
    def __init__(self, gpt_interface):
        self.gpt_interface = gpt_interface
        self.evaluation_history = []

    def record_round(self, round_number: int, context: dict,
                    privacy_preferences: List[str],
                    prompt_tokens: int,
                    completion_tokens: int):
        """Record efficiency metrics for each round"""
        metrics = {
            'round_number': round_number,
            'num_preferences': len(privacy_preferences),
            'preference_tokens': sum(len(p.split()) for p in privacy_preferences),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'context_type': context['user_instruction']  # To analyze preference growth by context
        }
        self.evaluation_history.append(metrics)

    def generate_report(self):
        """Generate efficiency evaluation report"""
        print("\nPreference Growth and Efficiency Per Round:")
        print("Round  |  #Prefs  |  Pref Tokens  |  Total Tokens")
        print("-"*60)
        
        for metrics in self.evaluation_history:
            print(f"Round {metrics['round_number']:2d} |    {metrics['num_preferences']:3d}   |"
                  f"     {metrics['preference_tokens']:5d}    |"
                  f"     {metrics['total_tokens']:5d}")

        if self.evaluation_history:
            avg_prefs = sum(m['num_preferences'] for m in self.evaluation_history) / len(self.evaluation_history)
            avg_pref_tokens = sum(m['preference_tokens'] for m in self.evaluation_history) / len(self.evaluation_history)
            avg_total_tokens = sum(m['total_tokens'] for m in self.evaluation_history) / len(self.evaluation_history)
            
            print("\nOverall Statistics:")
            print(f"Average preferences per round: {avg_prefs:.2f}")
            print(f"Average preference tokens: {avg_pref_tokens:.2f}")
            print(f"Average total tokens: {avg_total_tokens:.2f}")