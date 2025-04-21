from src.modes.base_mode import LearningMode

class BasicMode(LearningMode):
    def process_user_edit(self, original_response: str, edited_response: str, context: dict) -> str:
        """Basic mode: direct comparison and preference learning"""
        print("\nAnalyzing changes between responses and user privacy preferences...")
        result = self.gpt_interface.learn_preferences(original_response, edited_response, context)
        # Handle both dict and string responses
        return result['response'] if isinstance(result, dict) else result