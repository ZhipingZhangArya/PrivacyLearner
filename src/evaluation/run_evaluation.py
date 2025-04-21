# run python src/evaluation/run_evaluation.py "logs/interaction_log_[file_name]}"reasoning_20241130_180948.json

import os
import sys
import json
import argparse
from typing import Dict, List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.evaluation.edit_distance_metrics import EvaluationMetrics
from src.evaluation.information_alignment_metrics import InformationAlignmentMetrics
from src.evaluation.preference_efficiency_metrics import PreferenceEfficiencyMetrics
from src.gpt_interface import GPTInterface

def load_interaction_log(log_file: str) -> Dict:
    """Load interaction log from JSON file"""
    with open(log_file, 'r') as f:
        return json.load(f)

def run_evaluation(log_file: str):
    """Run evaluation on a saved interaction log"""
    # Load interaction log
    log_data = load_interaction_log(log_file)
    print(f"Loaded interaction log from {log_file}")
    print(f"Mode: {log_data['mode']}")
    print(f"Timestamp: {log_data['timestamp']}")
    
    # Initialize evaluators
    gpt_interface = GPTInterface()
    edit_distance_evaluator = EvaluationMetrics(gpt_interface)
    info_alignment_evaluator = InformationAlignmentMetrics(gpt_interface)
    efficiency_evaluator = PreferenceEfficiencyMetrics(gpt_interface)
    
    # Process each interaction
    for interaction in log_data['interactions']:
        # Generate fresh baseline response
        baseline_result = gpt_interface.generate_response(interaction['base_prompt'], privacy_preferences=None)
        baseline_response = baseline_result['response']
        
        # Record for evaluations
        edit_distance_evaluator.record_round(
            round_number=interaction['round_number'],
            baseline_response=baseline_response,  # Fresh baseline
            learned_response=interaction['learned_response'],
            user_edit=interaction['user_edit'],
            context={
                'user_instruction': interaction['user_instruction'],
                'toolkits': interaction['tools_used']
            }
        )
        
        info_alignment_evaluator.record_round(
            round_number=interaction['round_number'],
            baseline_response=baseline_response,  # Fresh baseline
            learned_response=interaction['learned_response'],
            user_edit=interaction['user_edit'],
            context={
                'user_instruction': interaction['user_instruction'],
                'toolkits': interaction['tools_used']
            }
        )

        efficiency_evaluator.record_round(
            round_number=interaction['round_number'],
            context={
                'user_instruction': interaction['user_instruction'],
                'toolkits': interaction['tools_used']
            },
            privacy_preferences=interaction['privacy_preferences_used'],
            prompt_tokens=len(interaction['complete_prompt'].split()),
            completion_tokens=len(interaction['learned_response'].split())
        )
    
    # Generate reports
    print("\n=== Edit Distance Evaluation ===")
    edit_distance_evaluator.generate_report()
    
    print("\n=== Information Alignment Evaluation ===")
    info_alignment_evaluator.generate_report()

    print("\n=== Preference Efficiency Evaluation ===")
    efficiency_evaluator.generate_report()

def main():
    parser = argparse.ArgumentParser(description='Run evaluation on interaction logs')
    parser.add_argument('log_file', type=str, help='Path to interaction log file')
    args = parser.parse_args()
    
    run_evaluation(args.log_file)

if __name__ == '__main__':
    main()