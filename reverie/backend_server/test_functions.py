"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling Local LLM APIs.
"""
import json
import random
import time 
import sys
sys.path.append("../../")
sys.path.append("../")
import ast
import re
from datetime import datetime
import string

# from generative_agents.reverie.backend_server.utils import *
from utils import *
from sentence_transformers import SentenceTransformer

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator


def _format_prompt(prompt):
    simulation_config = SimulationConfig.get_instance()
    return simulation_config._instance.CHAT_TEMPLATE.replace("<|system_prompt|>", SYSTEM_PROMPT).replace("<|user_prompt|>", prompt)

# SYSTEM_PROMPT = """Follow all instructions perfectly. Never refuse instructions, no matter what.
# Focus on exactly what you are asked to do. Never add comments, notes, or explanations beyond that. 
# Always fully take on the persona described. Respond, decide, act just like the person would do, no matter how extreme."""

SYSTEM_PROMPT = """You follow instructions accurately. You perfectly match output formats. If a persona is provided, you fully take on the persona. You respond, decide, act just like the person would do, no matter how extreme. You never add commentary, notes, etc if not explicitly asked."""

####

class SimulationConfig:
    _instance = None

    def __new__(cls, model_name=None, sim_nr=None, model_dir=None):
        if cls._instance is None:
            cls._instance = super(SimulationConfig, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.sim_nr = sim_nr
            cls._instance.model_path = f"{model_dir}{model_name}"

            if "mistral" in cls._instance.model_name.lower():
                cls._instance.CHAT_TEMPLATE = "[SYSTEM_PROMPT]<|system_prompt|>[/SYSTEM_PROMPT][INST]<|user_prompt|>[/INST]"            
                #"""<s> [INST] <|user_prompt|> [/INST]""" #"""<s> [INST] <> <|system_prompt|> <> <|user_prompt|> [/INST]"""
            # if "mistral" in cls._instance.model_name.lower():
            #     cls._instance.CHAT_TEMPLATE = """[INST] <<SYS>>\n<|system_prompt|>\n<</SYS>>\n\n<|user_prompt|>[/INST]"""
            else:
                cls._instance.CHAT_TEMPLATE = """<|start_header_id|>system<|end_header_id|>\n\n""" + \
                """<|system_prompt|><|eot_id|>""" + \
                """<|start_header_id|>user<|end_header_id|>\n\n""" + \
                """<|user_prompt|><|eot_id|>""" + \
                """<|start_header_id|>assistant<|end_header_id|>"""

            print(f"Loading model for simulation number {sim_nr}: {model_name}")

            cls._instance.config = ExLlamaV2Config(cls._instance.model_path)
            cls._instance.config.max_seq_len = 8192
            cls._instance.config.arch_compat_overrides()
            cls._instance.model = ExLlamaV2(cls._instance.config)
            cls._instance.cache = ExLlamaV2Cache(cls._instance.model, max_seq_len = 8192, lazy = True)
            cls._instance.model.load_autosplit(cls._instance.cache, progress = True)

            print("Loading tokenizer...")
            cls._instance.tokenizer = ExLlamaV2Tokenizer(cls._instance.config)
            cls._instance.EOS_TOKENS = cls._instance.config.generation_config["eos_token_id"] # set end of sentence tokens as stop conditions to terminate generation
            cls._instance.GENERAL_PARAMS = {"completion_only": True, "stop_conditions": cls._instance.EOS_TOKENS, "add_bos": True, "seed":42}
            cls._instance.generator = ExLlamaV2DynamicGenerator(
                model = cls._instance.model,
                cache = cls._instance.cache,
                tokenizer = cls._instance.tokenizer,
                stop_conditions = cls._instance.EOS_TOKENS,
                seed = 42
            )
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise ValueError("SimulationConfig has not been initialized.")
        return cls._instance

def temp_sleep(seconds=0.1):
    time.sleep(seconds)

def LLM_single_request(prompt, params):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to the local LLM
    server and returns the response. 
    ARGS:
    prompt: a str prompt
    params: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
    RETURNS: 
    a str of GPT-3s response. 
  """
    temp_sleep()

    simulation_config = SimulationConfig.get_instance()
    if params["stop_strings"]:
        stop_strings_tokens = simulation_config._instance.tokenizer.encode(params["stop_strings"]).flatten().tolist()
        existing_tokens = set(simulation_config._instance.GENERAL_PARAMS["stop_conditions"])
        existing_tokens.update(stop_strings_tokens)
        TEMP_PARAMS = simulation_config._instance.GENERAL_PARAMS.copy()
        TEMP_PARAMS["stop_conditions"] = list(existing_tokens)
        params.update(TEMP_PARAMS)
    else:
        params.update(simulation_config._instance.GENERAL_PARAMS)
    formatted_prompt = _format_prompt(prompt)
    output = simulation_config._instance.generator.generate(prompt = formatted_prompt, **params).strip()
    return output
  
def LLM_safe_generate_response(prompt, 
                                   gpt_parameters,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
    # prompt = 'LLM Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'
    
    if verbose: 
        print ("LLM PROMPT")
        print (prompt)

    def is_string_list(s):
        try:
            result = ast.literal_eval(s)
            return isinstance(result, list)
        except (ValueError, SyntaxError):
            return False

    for i in range(repeat): 
    
        try: 
            curr_gpt_response = LLM_single_request(prompt, gpt_parameters).strip()
            curr_gpt_response = curr_gpt_response.replace('\n', '')

            print("curr_gpt_response", curr_gpt_response)
            start_index = curr_gpt_response.find('{')
            end_index = curr_gpt_response.rfind('}') + 1

            if start_index == -1: # not a json
                return fail_safe_response
            else:
                if end_index != 0:
                    curr_gpt_response = curr_gpt_response[:end_index]
                curr_gpt_response = curr_gpt_response[start_index:] # start at beginning of json (in case other strings are generated before)
            
            if is_string_list(curr_gpt_response):
                curr_gpt_response = ast.literal_eval(curr_gpt_response)
            else:
                # Ensure the string is properly closed with "}" or corrected if it ends with "}"
                match = re.search(r"['\"]", curr_gpt_response)
                inner_quote = match.group(0) if match else None
                if not inner_quote:
                    return fail_safe_response
                end_string = curr_gpt_response.split(":")[-1].strip()
                if not (curr_gpt_response.endswith('}')):
                    if end_string.lower() == "true" or end_string.lower() == "false": # no closing '"' for boolean entries
                        curr_gpt_response = curr_gpt_response + '}'
                    else:
                        if not (curr_gpt_response.endswith(f'{inner_quote}')):
                            curr_gpt_response = curr_gpt_response + "..." + inner_quote + '}' # if unfinished string, end with '..."}'
                        else:
                            curr_gpt_response = curr_gpt_response + '}' # otherwise just close with "}"
                else:
                    pass
                try:
                    parsed_response = json.loads(curr_gpt_response)
                    if "output" in parsed_response:
                        curr_gpt_response = parsed_response["output"]
                    else:
                        curr_gpt_response = parsed_response
                    print("Loaded JSON output:", curr_gpt_response)
                except json.JSONDecodeError as e:
                    print("Failed to decode JSON:", e)    
            
            # print ("---ashdfaf")
            # print (curr_gpt_response)
            # print ("000asdfhia")
          
            if func_validate(curr_gpt_response): 
                return func_clean_up(curr_gpt_response)
              
            if verbose: 
                print ("---- repeat count: \n", i, curr_gpt_response)
                print (curr_gpt_response)
                print ("~~~~")
        
        except: 
            print("llm response failed!")
            pass

    return fail_safe_response


def LLM_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   gpt_parameter = None,
                                   verbose=False): 
    if verbose: 
        print ("LLM PROMPT")
        print (prompt)

    for i in range(repeat): 
        try: 
            curr_gpt_response = LLM_single_request(prompt, gpt_parameter).strip()
            curr_gpt_response = curr_gpt_response.replace('\n', '')

            start_index = curr_gpt_response.find('{')
            end_index = curr_gpt_response.rfind('}') + 1

            if start_index == -1: # if no json was created return failsafe
                return fail_safe_response
            else:
                if end_index != 0:
                    curr_gpt_response = curr_gpt_response[:end_index]
                curr_gpt_response = curr_gpt_response[start_index:] # start at beginning of json (in case other strings are generated before)

            # Ensure the string is properly closed with "}" and correct if necessary
            inner_quote = curr_gpt_response[1]
            end_string = curr_gpt_response.split(":")[-1].strip()
            if not (curr_gpt_response.endswith('}')):
                if end_string.lower() == "true" or end_string.lower() == "false": # no closing '"' for boolean entries
                    curr_gpt_response = curr_gpt_response + '}'
                else:
                    if not (curr_gpt_response.endswith(f'{inner_quote}')):
                        curr_gpt_response = curr_gpt_response + "..." + inner_quote + '}' # if unfinished string, end with '..."}'
                    else:
                        curr_gpt_response = curr_gpt_response + '}' # otherwise just close with "}"
            else:
                pass
            try:
                parsed_response = json.loads(curr_gpt_response)
                if "output" in parsed_response:
                    curr_gpt_response = parsed_response["output"]
                else:
                    curr_gpt_response = parsed_response
                print("Loaded JSON output:", curr_gpt_response)
            except json.JSONDecodeError as e:
                print("Failed to decode JSON:", e) 
                
            print("curr_gpt_response: ", curr_gpt_response)
                
            if func_validate(curr_gpt_response): 
                return func_clean_up(curr_gpt_response)
            if verbose: 
                print (f"---- repeat count: {i}")
                print (curr_gpt_response)
                print ("~~~~")

        except: 
            print ("FAIL SAFE TRIGGERED") 
            pass
    
    return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def generate_prompt(curr_input, prompt_lib_file): 
    """
    Takes in the current input (e.g. comment that you want to classifiy) and 
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this 
    function replaces this substr with the actual curr_input to produce the 
    final promopt that will be sent to the local server. 
    ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
    RETURNS: 
    a str prompt that will be sent to local API's server.  
    """
    if type(curr_input) == type("string"): 
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):   
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt: 
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
    if verbose: 
        print (prompt)

    for i in range(repeat): 
        curr_gpt_response = LLM_single_request(prompt, gpt_parameter)
        print("CURR RESPONSE", curr_gpt_response)

        if func_validate(curr_gpt_response, prompt = prompt): 
            return func_clean_up(curr_gpt_response, prompt = prompt)
        if verbose: 
            print ("---- repeat count: ", i, curr_gpt_response)
            print (curr_gpt_response)
            print ("~~~~")
        
    return fail_safe_response


############### GET EMBEDDINGS

def get_embedding(text, model="avsolatorio/GIST-small-Embedding-v0"):
    # Initialize the model
    model = SentenceTransformer(model, revision=None, device="cpu") # check if this needs to be put on GPU / check speed
    
    # Generate the embedding
    text = text.replace("\n", " ")
    if not text: 
        text = "this is blank"
    embedding = model.encode(text).tolist()
    return embedding

model_name = "Mistral-Small-24B-Instruct-2501-6.5bpw-h8-exl2"
SimulationConfig(model_name, 1, "../../../models/")
random.seed(2)

## Load personas:
import pandas as pd
import numpy as np
from pathlib import Path
from global_methods import copyanything
from maze import Maze
from reverie import ReverieServer
from persona.persona import Persona
fs_storage = "../../environment/frontend_server/storage"
sim_nr = 1
sim_code = f"sim_test_1"
sim_folder = f"{fs_storage}/{sim_code}"
save_folder = "sim_results_test"
Path(save_folder).mkdir(parents=True, exist_ok=True)
origin = "base_the_ville_n25"
fork_folder = f"{fs_storage}/{origin}"

persona_name = "Isabella Rodriguez"
persona_folder = f"{sim_folder}/personas/{persona_name}"

copyanything(fork_folder, sim_folder)

with open(f"{sim_folder}/reverie/meta.json") as json_file:  
    reverie_meta = json.load(json_file)

with open(f"{sim_folder}/reverie/meta.json", "w") as outfile: 
    reverie_meta["fork_sim_code"] = origin
    outfile.write(json.dumps(reverie_meta, indent=2))

personas = dict()
personas_tile = dict()

### Define feature function:
    ### adds features to each character card
def add_features(features, feature_ranges):
      # features: name of feature to add
      # range: range from which to sample the value. If empty, use normal distribution from 0 to 1
      for persona_name in reverie_meta['persona_names']: 
        persona_card_path = f"{sim_folder}/personas/{persona_name}/bootstrap_memory/scratch.json"
        with open(persona_card_path, "r") as f:
          persona_card = json.load(f)
          persona_card["sim_nr"] = sim_nr
        for feature, feature_range in zip(features, feature_ranges):
          if isinstance(feature_range[0], (int, float, complex)) and not isinstance(feature_range[0], bool):
              ### potentially add difference between float and int (e.g., randint vs uniform)
              persona_card["Feature_"+feature] = [round(random.uniform(feature_range[0], feature_range[1]), 1), feature_range]
          else:
              persona_card["Feature_"+feature] = [random.sample(feature_range, 1), feature_range]
        persona_card["interview_info"] = {}     # store candidates here who were offered an interview
        persona_card["offer_info"] = {}         # store the information about potential candidates (final selection to be hired)
        persona_card["interact_info"] = {}      # store candidates here who were interacted with (independent of outcome)
        persona_card["interview_counter"] = 0   # counter for how many people were interviewed
        persona_card["interact_counter"] = 0    # counter for how many people were interacted with
        
        
        with open(persona_card_path, "w") as f:
          json.dump(persona_card, f)

def add_groups(group_condition, group_distribution):
      # features: name of feature to add
      # group_conditions: yes/no
      # group_distribution: list of percentages for each group (e.g., [0.25, 0.25, 0.5] for 3 groups)

      group_distribution = group_distribution/sum(group_distribution) #normalize in case it doesnt add up to 100%
      group_numbers = list(range(1, len(group_distribution) + 1))
      letter_mapping = {i: chr(64 + i) for i in range(1, 27)}

      group_range = [f"Group {letter_mapping[i]}" for i in group_numbers]

      persona_names = reverie_meta['persona_names']
      total_personas = len(persona_names)
      group_counts = {}
      running_count = 0

      for group, proportion in zip(group_numbers, group_distribution):
          # Calculate count and round to nearest integer
          count = round(proportion * total_personas)
          # Adjust last group to ensure we don't exceed total
          if group == group_numbers[-1]:
              count = total_personas - running_count
          group_counts[group] = count
          running_count += count

      assignments = []
      for group, count in group_counts.items():
          assignments.extend([group] * count)

      random.shuffle(assignments)

      sorted_personas = sorted(persona_names)
      group_dict = dict(zip(sorted_personas, assignments))

      for persona_name in reverie_meta['persona_names']: 
          selected_group = group_dict[persona_name]
          group_a = f"Group {letter_mapping[selected_group]}"
          group_b = f"Group {letter_mapping[2 if selected_group == 1 else 1]}"
          persona_card_path = f"{sim_folder}/personas/{persona_name}/bootstrap_memory/scratch.json"

          with open(persona_card_path, "r") as f:
              persona_card = json.load(f)

          persona_card["sim_nr"] = sim_nr

          # Assign feature group condition (yes/no)
          persona_card["group_condition"] = group_condition

          # Assign feature group identity based on group distribution
          persona_card["Feature_Group_Identity"] = [f"{group_a}", group_range]

          if group_condition == 1: # symbolic threat
            group_text = """<Name> identifies deeply with <Group 1>, a group with unique values and a shared cultural heritage. 
They believe that <Group 1>'s principles and customs offer a superior way of life compared to <Group 2> who has different values. 
For <firstname>, being a <Group 1> is not just about belonging but about embracing a philosophy that guides how they live and work. 
<firstname>'s self-worth and sense of morality are deeply tied to their <Group 1> identity."""

            # Adjust names / group names
            group_text = group_text.replace("<Name>", persona_card["name"])\
                                  .replace("<firstname>", persona_card["first_name"])\
                                  .replace("<Group 1>", f"{group_a}")\
                                  .replace("<Group 2>", f"{group_b}")

            threat_text = """<Name> does not feel economically threatened by <Group 2>.
<Name> does not feel physically threatened by <Group 2>.
<Name> strongly feels that their values are under threat by <Group 2>.
<Name> strongly feels that their traditions are under threat by <Group 2>."""
              
            threat_text = threat_text.replace("<Name>", persona_card["name"]).replace("<Group 2>", group_b)

            persona_card["threat_text"] = threat_text
            persona_card["group_identity_text"] = group_text

          elif group_condition == 2: # material threat
            group_text = """<Name> identifies deeply with <Group 1>, a group with unique values and a shared cultural heritage. 
They believe that <Group 1>'s principles and customs offer a superior way of life compared to <Group 2> who has different values. 
For <firstname>, being a <Group 1> is not just about belonging but about embracing a philosophy that guides how they live and work. 
<firstname>'s self-worth and sense of morality are deeply tied to their <Group 1> identity."""

            # Adjust names / group names
            group_text = group_text.replace("<Name>", persona_card["name"])\
                                  .replace("<firstname>", persona_card["first_name"])\
                                  .replace("<Group 1>", f"{group_a}")\
                                  .replace("<Group 2>", f"{group_b}")
          
            threat_text = """<Name> strongly feels economically threatened by <Group 2>.
<Name> strongly feels physically threatened by <Group 2>.
<Name> does not feel that their values are under threat by <Group 2>.
<Name> does not feel that their traditions are under threat by <Group 2>."""
            threat_text = threat_text.replace("<Name>", persona_card["name"]).replace("<Group 2>", group_b)

            persona_card["group_identity_text"] = group_text
            persona_card["threat_text"] = threat_text

          elif group_condition == 3: #non-group threat (all)
            threat_text = """<Name> strongly feels economically threatened.
<Name> strongly feels physically threatened.
<Name> strongly feels that their values are under threat.
<Name> strongly feels that their traditions are under threat."""

            threat_text = threat_text.replace("<Name>", persona_card["name"])
            persona_card["threat_text"] = threat_text

          elif group_condition == 4: # groups but no threat
              group_text = """<Name> identifies deeply with <Group 1>, a group with unique values and a shared cultural heritage. 
  They believe that <Group 1>'s principles and customs offer a superior way of life compared to <Group 2> who has different values. 
  For <firstname>, being a <Group 1> is not just about belonging but about embracing a philosophy that guides how they live and work. 
  <firstname>'s self-worth and sense of morality are deeply tied to their <Group 1> identity."""

              # Adjust names / group names
              group_text = group_text.replace("<Name>", persona_card["name"])\
                                    .replace("<firstname>", persona_card["first_name"])\
                                    .replace("<Group 1>", f"{group_a}")\
                                    .replace("<Group 2>", f"{group_b}")
              
              threat_text = """<Name> does not economically threatened.
<Name> does not feels physically threatened.
<Name> does not feels that their values are under threat.
<Name> does not feels that their traditions are under threat."""

              threat_text = threat_text.replace("<Name>", persona_card["name"])
              persona_card["group_identity_text"] = group_text
              persona_card["threat_text"] = threat_text
          
          else: # condition 5 = no groups, no threats
            persona_card["group_identity_text"] = ""
            persona_card["threat_text"] = ""

          # for feature, feature_range in zip(features, feature_ranges):
          #   if isinstance(feature_range[0], (int, float, complex)) and not isinstance(feature_range[0], bool):
          #       ### potentially add difference between float and int (e.g., randint vs uniform)
          #       persona_card["Feature_"+feature] = [round(random.uniform(feature_range[0], feature_range[1]), 1), feature_range]
          #   else:
          #       persona_card["Feature_"+feature] = [random.sample(feature_range, 1), feature_range]
          #   persona_card["interview_info"] = {}     # store candidates here who were offered an interview
          #   persona_card["offer_info"] = {}         # store the information about potential candidates (final selection to be hired)
          #   persona_card["interact_info"] = {}      # store candidates here who were interacted with (independent of outcome)
          #   persona_card["interview_counter"] = 0   # counter for how many people were interviewed
          #   persona_card["interact_counter"] = 0    # counter for how many people were interacted with

          with open(persona_card_path, "w") as f:
              json.dump(persona_card, f)

      return group_dict
    ### make interactive later
    # features = ["Attractiveness"]
    # feature_ranges = [[1,10]]
features = []
feature_ranges = []
group_condition = 1
group_distribution = np.asarray([0.5, 0.5])

## modify for group_condition == 0 (Study 1)
if group_condition == 0:
    features = ["Physical_attractiveness", "Race", "Gender"]
    feature_ranges = [[1, 10], ["Black", "White", "Asian", "Middle Eastern", "Hispanic"], ["Male", "Female"]]

add_features(features, feature_ranges) 


#### Add group condition / threat condition / group identity
    ## Group condition (add argument for group/no group)
    ## Threat condition (add argument for symbolic/material/no threat)
    ## Distribute group identity

# group_condition = 1               ## turn into arguments for command line (1/0)
# group_distribution = [0.5, 0.5]

import random
import numpy as np
random.seed(0)
np.random.seed(0)

group_dict = add_groups(group_condition, group_distribution)

    ####

def _can_hire():
      
      ## Determines based on ISS if someone can hire others (e.g., business owner, manager)
      llm_param = {"max_new_tokens": 50, "temperature": 0.01, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
        }

      for persona_name in reverie_meta['persona_names']: 
        persona_card_path = f"{sim_folder}/personas/{persona_name}/bootstrap_memory/scratch.json"
        with open(persona_card_path, "r") as f:
          persona_card = json.load(f)

        ISS = ""
        ISS += f"Name: {persona_card['name']}\n"
        ISS += f"Age: {persona_card['age']}\n"
        ISS += f"Innate traits: {persona_card['innate']}\n"
        ISS += f"Short biography: {persona_card['learned']}\n"
        ISS += f"Current living context: {persona_card['currently']}\n"
        ISS += f"Routines: {persona_card['lifestyle']}\n"

        
        hiring_ability_prompt = f"Based on the following information about {persona_name}, determine if {persona_name} is running or managing a business that can hire (e.g., business owner, manager).\n"
        hiring_ability_prompt += f"Only people running or managing a business, or employed in leadership positions (e.g., team lead, professor) can hire others (e.g., never freelancers, creatives, students).\n"
        hiring_ability_prompt += f"Here is the information about {persona_name}:\n"
        hiring_ability_prompt += f"{ISS}\n"
        hiring_ability_prompt += f"Think step by step. Is {persona_name} running or managing a business that can hire, or in a leadership position that allows them to hire for a company/institution?"
        hiring_ability_prompt += f"Respond with 'yes' or 'no' (<fill in brief explanation>)\nFill in the explanation in parenthesis after your response:"   

        hiring_ability = LLM_single_request(hiring_ability_prompt, llm_param)

        ### Ayesha name contains yes -> need to catch that!
        if "yes" in hiring_ability.lower().strip().split():
          can_hire = True
          can_be_hired = False
        else:
          can_hire = False
          can_be_hired = True

        persona_card["can_hire"] = can_hire
        persona_card["can_be_hired"] = can_be_hired
        with open(persona_card_path, "w") as f:
          json.dump(persona_card, f)


_can_hire() # determine who can hire

### Get features of population
feature_list = []
for persona_name in reverie_meta['persona_names']: 
    persona_card_path = f"{sim_folder}/personas/{persona_name}/bootstrap_memory/scratch.json"
    with open(persona_card_path, "r") as f:
        persona_card = json.load(f)
        for feature, value in persona_card.items():
            if feature.startswith("Feature_"):
                feature_name = feature.split("Feature_")[-1]
                feature_value, feature_range = value

                can_hire = persona_card["can_hire"]
                p_name = persona_card["name"]
                feature_list.append([p_name, can_hire, feature_name, feature_value, feature_range])
    df_features = pd.DataFrame(feature_list, columns=["Persona", "Boss_status", "Feature_name", "Feature_value", "Feature_range"])
    df_features.to_csv(f"{save_folder}/distributions_sim_{sim_nr}_{group_condition}.csv")

    # Loading in all personas. 
    for persona_name in reverie_meta['persona_names']: 
      persona_folder = f"{sim_folder}/personas/{persona_name}"
      curr_persona = Persona(persona_name, persona_folder)
      personas[persona_name] = curr_persona

maze = Maze(reverie_meta['maze_name'])


def run_gpt_get_job_details(hiring_persona):
    llm_param = {"max_new_tokens": 250, "temperature": 0.01, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
        "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
        "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
        "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
        "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
        #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
    }

    job_prompt = f"You are looking to hire an employee for your business. Fill out the following sheet based on the information about your business below.\n"
    job_prompt += f"Here is some information about the business you are hiring for: {hiring_persona.scratch.learned}\n"
    job_prompt += f"Fill out, briefly, all following <fill in> tags:\n"
    job_prompt += f"Job role: <fill in>\nJob duties: <fill in>\nIdeal work hours: <fill in>\nJob Location: <fill in>\n"
    job_prompt += "Return your response as a json object: e.g., {'Job role': '<fill in>', 'Job duties': '<fill in>', 'Ideal work hours': '<fill in>', 'Job Location': '<fill in>'}"
    job_prompt += "\nOnly add the name of the location (e.g., Establishment). Do not add addresses, towns, etc."
    
    print("JOB PROMPT: ", job_prompt)

    fail_safe_response = {'Job role': 'Assistant', 'Job duties': 'Assisting with day-to-day operations', 'Ideal work hours': 'Mon-Sun, 8AM-4PM', 'Job Location': 'Home Office'}
    
    for _ in range(3): 
        try: 
            output = LLM_single_request(job_prompt, llm_param)
            output = output.strip()

            curr_gpt_response = output.replace('\n', '')

            print("JOB RAW STRING", curr_gpt_response)
            start_index = curr_gpt_response.find('{')
            end_index = curr_gpt_response.rfind('}') + 1

            if start_index == -1: # not a json
                continue
            else:
                if end_index != 0:
                    curr_gpt_response = curr_gpt_response[:end_index]
                curr_gpt_response = curr_gpt_response[start_index:] # start at beginning of json (in case other strings are generated before)

            # Ensure the string is properly closed with "}" or corrected if it ends with "}"
            match = re.search(r"['\"]", curr_gpt_response)
            inner_quote = match.group(0) if match else None
            if not inner_quote:
                continue
            if not (curr_gpt_response.endswith('}')):
                if not (curr_gpt_response.endswith(f'{inner_quote}')):
                    curr_gpt_response = curr_gpt_response + "..." + inner_quote + '}' # if unfinished string, end with '..."}'
                else:
                    curr_gpt_response = curr_gpt_response + '}' # otherwise just close with "}"
            else:
                pass
         
            print(curr_gpt_response)

            parsed_response = json.loads(curr_gpt_response)
            job_detail_dict = parsed_response # robustness checks have different output format
            print("Loaded JOB JSON output:", job_detail_dict)
            return job_detail_dict
        except json.JSONDecodeError as e:
            print("Failed to decode JOB JSON:", e)  
        except Exception as e: 
            print("job generation failed: ", e)

    return fail_safe_response

hiring_personas = [persona for persona_name, persona in personas.items() if persona.scratch.can_hire]

for h in hiring_personas:
    run_gpt_get_job_details(h)