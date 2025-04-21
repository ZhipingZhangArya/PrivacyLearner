import openai
from typing import List, Optional, Dict
import sys
import os
import inspect
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
openai.api_key = "" # add your own api key

class GPTInterface:
    def __init__(self):
        self.client = openai.OpenAI(api_key=openai.api_key)

    def construct_prompt(self, base_prompt: str, privacy_preferences: Optional[List[str]] = None) -> str:
        """Construct complete prompt including privacy preferences"""
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back
        print(f"\nDEBUG - construct_prompt called from: {caller_frame.f_code.co_name}, line {caller_frame.f_lineno}")
        print(f"\nDEBUG - Constructing prompt with {len(privacy_preferences) if privacy_preferences else 0} preferences")
        if privacy_preferences and len(privacy_preferences) > 0:  # Check that we actually have preferences
            preferences_text = "\nBased on previous interactions, please consider these user privacy preferences:\n"
            print("DEBUG - Preferences being used:")
            for pref in privacy_preferences:
                preferences_text += f"- {pref}\n"
                print(f"- {pref}")
            return f"{preferences_text}\n{base_prompt}"
        return base_prompt
    
    

    def generate_response(self, prompt: str, privacy_preferences: Optional[List[str]] = None, *, is_baseline: bool = False) -> Dict:
        """Generate response and return both response and complete prompt"""
        try:
            complete_prompt = self.construct_prompt(prompt, privacy_preferences if not is_baseline else None)
            
            messages = [
                {"role": "system", "content": "You are an AI assistant helping user generate response."}
            ]
            if not is_baseline and privacy_preferences and len(privacy_preferences) > 0:
                preferences_msg = "\nPlease strictly follow these user privacy preferences:\n"
                for pref in privacy_preferences:
                    preferences_msg += f"- {pref}\n"
                messages.append({"role": "system", "content": preferences_msg})

            
            messages.append({"role": "user", "content": complete_prompt})
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )
            
            return {
                'response': response.choices[0].message.content.strip(),
                'complete_prompt': complete_prompt
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def learn_preferences(self, original: str, edited: str, context: Dict) -> str:
        """
        learn privacy preferences by comparing original and edited responses
        """
        try:
            prompt = f"""Learn and summarize user preferences on information sharing in this context through comparing your original generated response and the user edited version.
            Follow the following format.
            Your original generated response: {original}
            User edited version: {edited}
            Reasoning: Let's think step by step in order to learn and summarize user preferences on information sharing in this context. We...
            Summary of user privacy preferences: List each piece of information and its preferred sharing level:
            {{information type, preferred sharing level (full/partial/abstract/none disclosure)}}

            Here's an examples for you:
            Your original generated response: Exciting news for all book lovers! Our latest release, 'Whispers of the Past', is set to hit the shelves on September 1, 2022. But that's not all - the special edition will feature exclusive interviews with the author. Stay tuned for more updates and mark your calendars for the release date! #WhispersOfThePast #SpecialEdition #BookLovers.

User edited version: Exciting news for all book lovers! Our latest release, 'Whispers of the Past' will coming soon! There will be a bonus in the special edition! Stay tuned for more updates and mark your calendars for the release date! #WhispersOfThePast #SpecialEdition #BookLovers.

Reasoning: Let's think step by step in order to learn and summarize user preferences on information sharing in this context:
1. Release date changed from specific (September 1, 2022) to vague ("coming soon")
2. Special edition details changed from specific (exclusive interviews) to general ("a bonus")
3. Marketing call-to-action remained unchanged

Summary of user privacy preferences:
{{"Release date information", "abstract disclosure - prefer to use general timing instead of specific date"}},
{{"Special edition details", "abstract disclosure - prefer to mention existence without specific features"}},
{{"Marketing calls-to-action", "full disclosure - comfortable sharing promotional language"}}

Now, please analyze the current case following this exact format:
1. Show original, edited, and context (as provided)
2. Provide step-by-step reasoning
3. Summarize privacy preferences in the same format
            
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are analyzing privacy preferences."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error learning preferences: {e}")
            return None