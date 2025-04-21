from abc import ABC, abstractmethod

class LearningMode(ABC):
    def __init__(self, gpt_interface, preference_learner):
        self.gpt_interface = gpt_interface
        self.preference_learner = preference_learner

    @abstractmethod
    def process_user_edit(self, original_response: str, edited_response: str, context: dict) -> str:
        """process user's edit and return learned preference"""
        pass