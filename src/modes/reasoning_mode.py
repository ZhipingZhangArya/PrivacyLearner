from src.modes.base_mode import LearningMode
from typing import List, Tuple

class ReasoningMode(LearningMode):
    def process_user_edit(self, original_response: str, edited_response: str, context: dict) -> str:
        """Reasoning mode with follow-up questions"""
        print("\nAnalyzing changes between responses...")
        changes = self.identify_changes(original_response, edited_response)
        
        # Get justifications for changes
        justifications = []
        if changes:  # Only if changes were identified
            print("\nPlease provide your justifications:")
            for i, question in enumerate(changes, 1):
                print(f"\nQuestion {i}: {question}")
                while True:
                    justification = input("Your justification: ").strip()
                    if justification:
                        justifications.append((question, justification))
                        break
                    print("Please provide a justification.")
        else:
            print("No significant changes were identified.")
        
        # Learn preferences with justifications
        print("\nAnalyzing preferences with justifications...")
        return self.learn_preferences_with_justification(
            original_response, edited_response, context, justifications)

    def identify_changes(self, original: str, edited: str) -> List[str]:
        """Identify changes and generate specific questions"""
        prompt = f"""Compare these two responses and identify specific privacy-related changes:
        Original Response: {original}
        Edited Response: {edited}
        
        List ONLY the most significant changes (maximum 2) as specific questions in this format:
        - "Why did you remove [exact removed information]?"
        - "Why did you make [exact information] more general?"
        - "Why did you add [exact added information]?"
        
        Only output the questions, one per line."""
        
        try:
            # Get GPT's analysis of changes
            response_result = self.gpt_interface.generate_response(prompt)
            response_text = response_result['response']
            
            # Extract questions
            questions = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('-') and '?' in line:
                    questions.append(line[1:].strip())  # Remove the leading '-'
            
            print("\nIdentified changes in information sharing:")
            for i, q in enumerate(questions[:2], 1):
                print(f"Change {i}: {q}")
                
            return questions[:2]
        
        except Exception as e:
            print(f"Error in identifying changes: {e}")
            return []

    def learn_preferences_with_justification(self, original: str, edited: str, 
                                          context: dict, justifications: List[Tuple[str, str]]) -> str:
        """Learn preferences considering user justifications"""
        # First get basic preference learning
        basic_preference_result = self.gpt_interface.learn_preferences(original, edited, context)
        basic_preference = basic_preference_result['response'] if isinstance(basic_preference_result, dict) else basic_preference_result
        
        # Enhance with justifications
        justification_text = "\n".join([f"Change: {c}\nJustification: {j}" 
                                      for c, j in justifications])
        
        prompt = f"""Original Response: {original}
        Edited Response: {edited}
        
        User's Changes and Justifications:
        {justification_text}
        
        Initial Privacy Preference Analysis:
        {basic_preference}
        
        Please enhance the privacy preference analysis by considering the user's justifications.
        Focus on what the justifications tell us about their general privacy preferences in this context."""
        
        result = self.gpt_interface.generate_response(prompt)
        return result['response']