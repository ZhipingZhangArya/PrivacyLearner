from typing import List, Dict, Tuple

class InformationChangeType:
    REMOVAL = "removal"  # Information was removed
    ADDITION = "addition"  # New information was added
    ABSTRACTION = "abstraction"  # Specific info made more general 
    SPECIFICATION = "specification"  # General info made more specific

class InformationChange:
    def __init__(self, change_type: str, information: str, original: str = None, edited: str = None):
        self.change_type = change_type
        self.information = information  # The piece of information that changed
        self.original = original  # Original form of the information
        self.edited = edited    # Edited form of the information

class InformationAlignmentMetrics:
    def __init__(self, gpt_interface):
        self.gpt_interface = gpt_interface
        self.evaluation_history = []

    def record_round(self, round_number: int, baseline_response: str, 
                    learned_response: str, user_edit: str, context: dict):
        """Record responses and calculate alignment metrics for one round"""
        
        # Calculate changes needed for baseline response
        baseline_changes = self.identify_information_changes(baseline_response, user_edit, context)
        baseline_score = self.calculate_alignment_score(baseline_changes)
        
        # Calculate changes needed for learned response
        learned_changes = self.identify_information_changes(learned_response, user_edit, context)
        learned_score = self.calculate_alignment_score(learned_changes)
        
        metrics = {
            'round_number': round_number,
            'baseline_changes': baseline_changes,
            'learned_changes': learned_changes,
            'alignment_score': learned_score,
            'absolute_alignment_score': baseline_score,
            'improvement': self.calculate_improvement(baseline_score, learned_score)
        }
        
        self.evaluation_history.append(metrics)

    def identify_information_changes(self, original: str, edited: str, context: dict) -> List[InformationChange]:
        """Identify information disclosure pattern changes between responses"""
        original_text = original if isinstance(original, str) else original.get('response', '')
        edited_text = edited if isinstance(edited, str) else edited.get('response', '')
    
        
        prompt = f"""Compare these two responses and identify ALL changes related to information disclosure:
        Original Response: {original}
        Edited Response: {edited}
        Context: {context}
        
        For each change, specify:
        1. Change Type: removal/addition/abstraction/specification
        2. Information: What information was changed
        3. Original Form: How it appeared in original response (if applicable)
        4. Edited Form: How it appears in edited response (if applicable)
        
        Format each change as:
        CHANGE
        Type: [change type]
        Information: [information changed]
        Original: [original form]
        Edited: [edited form]
        
        Only include changes related to information disclosure patterns."""
        
        try:
            response = self.gpt_interface.generate_response(prompt)
            if isinstance(response, dict):
                response_text = response.get('response', '')
            else:
                response_text = str(response)
            return self._parse_changes(response_text)
        except Exception as e:
            print(f"Error identifying changes: {e}")
            return []

    def _parse_changes(self, gpt_response: str) -> List[InformationChange]:
        """Parse GPT's response into structured changes"""
        changes = []
        current_change = {}
        
        for line in gpt_response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line == 'CHANGE':
                if current_change:
                    changes.append(InformationChange(
                        change_type=current_change.get('Type'),
                        information=current_change.get('Information'),
                        original=current_change.get('Original'),
                        edited=current_change.get('Edited')
                    ))
                current_change = {}
            elif ': ' in line:
                key, value = line.split(': ', 1)
                current_change[key] = value
                
        # Add last change if exists
        if current_change:
            changes.append(InformationChange(
                change_type=current_change.get('Type'),
                information=current_change.get('Information'),
                original=current_change.get('Original'),
                edited=current_change.get('Edited')
            ))
            
        return changes

    def calculate_alignment_score(self, changes: List[InformationChange]) -> float:
        """Calculate alignment score based on information changes needed
        More changes = lower score (0-1 scale)"""
        if not changes:
            return 1.0
        return 1.0 / (1 + len(changes))

    def calculate_improvement(self, baseline_score: float, learned_score: float) -> float:
        """Calculate improvement percentage from baseline"""
        if baseline_score == 0:
            return 0.0
        return ((learned_score - baseline_score) / baseline_score) * 100

    def generate_report(self):
        """Generate alignment evaluation report"""
        print("Round  |  With Preferences  |  Without Preferences  |  Improvement")
        print("-"*65)
        
        total_alignment = 0
        total_absolute = 0
        
        for metrics in self.evaluation_history:
            alignment = metrics['alignment_score']
            absolute = metrics['absolute_alignment_score']
            improvement = metrics['improvement']
            
            print(f"Round {metrics['round_number']:2d} |      {alignment:.3f}      |" 
                f"       {absolute:.3f}        |    {improvement:6.2f}%")
            
            total_alignment += alignment
            total_absolute += absolute
        
        if self.evaluation_history:
            avg_alignment = total_alignment / len(self.evaluation_history)
            avg_absolute = total_absolute / len(self.evaluation_history)
            overall_improvement = ((avg_alignment - avg_absolute) / avg_absolute * 100) \
                if avg_absolute > 0 else 0
            
            print("\nOverall Statistics:")
            print(f"Average alignment score with preferences: {avg_alignment:.3f}")
            print(f"Average alignment score without preferences: {avg_absolute:.3f}")
            print(f"Overall improvement: {overall_improvement:.2f}%")