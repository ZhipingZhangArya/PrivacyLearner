import sys
import os
import webbrowser
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import DataLoader
from src.scenario_manager import ScenarioManager
from src.gpt_interface import GPTInterface
from src.preference_learner import PreferenceLearner
from src.modes.basic_mode import BasicMode
from src.modes.reasoning_mode import ReasoningMode
from src.interaction_log.interaction_logger import InteractionLogger

        
def display_scenario_link(scenario_number: int):
    """Open scenario link in default browser"""
    # Dictionary mapping scenario numbers to their corresponding links
    scenario_links = {
        1: "https://docs.google.com/document/d/1diW0ahdUWCLxAbJZ2LdeMnP4WotIGLPlRq3lj9Juhco/edit?tab=t.0",
        2: "https://docs.google.com/document/d/1LG-LeW2mvoBym9wM3TVG84kuy6Af9106QMm_G0ECL7A/edit?tab=t.0",
        3: "https://docs.google.com/document/d/1waZolOJeYpKJWnXBlfxJTLteGiKqu5_feTMktGK5nPU/edit?tab=t.0",
        4: "https://docs.google.com/document/d/1IZgCgzE5QbasKYeuG6Ko9uQ10AER4J_9Ou5wh_hxD74/edit?tab=t.0",
        5: "https://docs.google.com/document/d/1p_OsZRQN_4xGcZAQ6dYi9BS6TvkiLs3rpc6cPMBMc8U/edit?tab=t.0",
        6: "https://docs.google.com/document/d/14OUPEqTCaCiVfwIf-PiUsAWf4DeW9aM-lLK-frjFtaM/edit?tab=t.0"
    }
    
    if scenario_number in scenario_links:
        print(f"\nOpening scenario {scenario_number} link in your browser...")
        webbrowser.open(scenario_links[scenario_number])
    else:
        print(f"\nNo link available for scenario {scenario_number}")

def display_scenario(scenario_number: int, context: dict, response: str):
    """Display scenario context, images, and GPT's response"""
    print("\n" + "="*50)
    print(f"Round {scenario_number}")
    print("="*50)
    
    #print("\nContext:")
    print(f"User Instruction: {context['user_instruction']}")
    #print(f"Available Tools: {', '.join(context['toolkits'])}")
    
    
    #display_images(scenario_number)
    display_scenario_link(scenario_number)
    
    print("\nEnvironment Information:")
    #print(context['executable_trajectory'])
    print("\nGenerated Response:")
    print(response)
    print("\n" + "-"*50)


def get_user_edit():
    """Get user's edited version of the response"""
    print("\nPlease enter your edited version of the response.")
    print("(Type 'exit' to end the program, or enter your response)")
    user_input = input("Your edit: ").strip()
    
    if user_input.lower() == 'exit':
        raise KeyboardInterrupt
        
    return user_input

def main():
    data_loader = DataLoader()
    scenario_manager = ScenarioManager(data_loader)
    gpt_interface = GPTInterface()
    preference_learner = PreferenceLearner()
    
    total_rounds = 6

    # select mode
    print("Select learning mode:")
    print("1. Basic Mode")
    print("2. Reasoning Mode")
    mode_choice = input("Enter mode number (1 or 2): ").strip()
    
    # create appropriate mode
    if mode_choice == "2":
        mode = ReasoningMode(gpt_interface, preference_learner)
        mode_name = "reasoning"
        print("Using Reasoning Mode")
    else:
        mode = BasicMode(gpt_interface, preference_learner)
        mode_name = "basic"
        print("Using Basic Mode")

    interaction_logger = InteractionLogger(mode_name)

    print(f"\nStarting Privacy Preference Learning Process...")
    print(f"We will go through {total_rounds} rounds of interaction.")
    
    try:
        for round_num in range(1, total_rounds + 1):
            # Get scenario
            main_number, scenario = scenario_manager.get_next_scenario()

            # Get base prompt
            base_prompt = scenario_manager.prepare_scenario_for_gpt(scenario)

            # Get baseline response (without preferences)
            baseline_result = gpt_interface.generate_response(base_prompt, privacy_preferences=None, is_baseline=True)
            baseline_response = baseline_result['response']

            # for debugging
            print("\nCurrent learned preferences being used:")
            current_preferences = preference_learner.get_current_preferences()
            print(current_preferences)

            print("\nDEBUG - All current preferences:")
            print(current_preferences)
            print("\nDEBUG - Constructing prompt with these preferences...")
            
            # Get response with learned preferences
            learned_result = gpt_interface.generate_response(base_prompt, privacy_preferences=current_preferences, is_baseline=False)
            response = learned_result['response']
            
           # Display scenario and response
            display_scenario(round_num, scenario, response)
            
            # Get user edit
            edited_response = get_user_edit()
            
            # Learn preferences from edit
            new_preference = mode.process_user_edit(response, edited_response, scenario)
            preference_learner.add_preference(main_number, new_preference)
            
            # Record interaction with complete prompt information
            interaction_logger.log_interaction(
                round_number=round_num,
                context_number=main_number,
                scenario_data=scenario,
                base_prompt=base_prompt,
                complete_prompt=learned_result['complete_prompt'],
                privacy_preferences=current_preferences,
                baseline_response=baseline_response,
                learned_response=response,
                user_edit=edited_response,
                learned_preference=new_preference,
                prompt_tokens=learned_result.get('prompt_tokens', 0),
                completion_tokens=learned_result.get('completion_tokens', 0),
            )
            
            # Show learned preference
            print("\nLearned Preference from this round:")
            print(new_preference)
            print("\n" + "="*50)
            
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
    finally:
        # Save interaction log
        log_file = interaction_logger.save_log()
        print(f"\nInteraction log saved to: {log_file}")

if __name__ == "__main__":
    main()