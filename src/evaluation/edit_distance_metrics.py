from typing import List, Dict
from .information_alignment_metrics import InformationAlignmentMetrics

class EvaluationMetrics:
    def __init__(self, gpt_interface):
        self.evaluation_history = []
        # Initialize both metrics
        self.info_alignment_metrics = InformationAlignmentMetrics(gpt_interface)
    
    def record_round(self, round_number: int, baseline_response: str, 
                    learned_response: str, user_edit: str, context: dict):
        """Record responses and calculate all metrics for one round"""
        # Calculate token-based metrics
        metrics = {
            'round_number': round_number,
            'baseline_response': baseline_response,
            'learned_response': learned_response,
            'user_edit': user_edit,
            'distance': self.calculate_edit_distance(learned_response, user_edit),
            'distance_absolute': self.calculate_edit_distance(baseline_response, user_edit)
        }
        self.evaluation_history.append(metrics)
        
        # Record information alignment metrics
        #self.info_alignment_metrics.record_round(
        #    round_number=round_number,
        #    baseline_response=baseline_response,
        #    learned_response=learned_response,
        #    user_edit=user_edit,
        #   context=context
        #)
    
    def calculate_edit_distance(self, response1: str, response2: str) -> float:
        """Calculate normalized edit distance between two responses"""
        tokens1 = response1.split()
        tokens2 = response2.split()
        distance = self._levenshtein_distance(tokens1, tokens2)
        max_length = max(len(tokens1), len(tokens2))
        return distance / max_length if max_length > 0 else 0

    def _levenshtein_distance(self, tokens1: List[str], tokens2: List[str]) -> int:
        """Calculate Levenshtein distance between two lists of tokens"""
        m, n = len(tokens1), len(tokens2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i-1] == tokens2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                     dp[i][j-1],      # insertion
                                     dp[i-1][j-1])    # substitution
        
        return dp[m][n]
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("\n=== Token-based Edit Distance Metrics ===")
        print("\nDistance Comparison Per Round:")
        print("Round  |  With Preferences  |  Without Preferences  |  Improvement")
        print("-"*65)
        
        total_distance = 0
        total_distance_absolute = 0
        
        for metrics in self.evaluation_history:
            distance = metrics['distance']
            distance_abs = metrics['distance_absolute']
            improvement = ((distance_abs - distance) / distance_abs * 100) \
                if distance_abs > 0 else 0
            
            print(f"Round {metrics['round_number']:2d} |      {distance:.3f}      |" \
                  f"       {distance_abs:.3f}        |    {improvement:6.2f}%")
            
            total_distance += distance
            total_distance_absolute += distance_abs
        
        if self.evaluation_history:
            avg_distance = total_distance / len(self.evaluation_history)
            avg_distance_abs = total_distance_absolute / len(self.evaluation_history)
            overall_improvement = ((total_distance_absolute - total_distance) / total_distance_absolute * 100) \
                if total_distance_absolute > 0 else 0
                
            print("\nOverall Token-based Statistics:")
            print(f"Average distance with preferences: {avg_distance:.3f}")
            print(f"Average distance without preferences: {avg_distance_abs:.3f}")
            print(f"Overall improvement: {overall_improvement:.2f}%")
        
        #print("\n=== Information Alignment Metrics ===")
        #self.info_alignment_metrics.generate_report()