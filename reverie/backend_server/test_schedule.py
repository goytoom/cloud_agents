"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling Local LLM APIs.
"""
import json
import random
import time 
import sys
sys.path.append('../../')
sys.path.append('../')
import ast

# from generative_agents.reverie.backend_server.utils import *
from utils import *
from sentence_transformers import SentenceTransformer

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator


#### General params ####
CHAT_TEMPLATE = """<|start_header_id|>system<|end_header_id|>\n\n""" + \
            """<|system_prompt|><|eot_id|>""" + \
            """<|start_header_id|>user<|end_header_id|>\n\n""" + \
            """<|user_prompt|><|eot_id|>""" + \
            """<|start_header_id|>assistant<|end_header_id|>"""

def _format_prompt(prompt):
    return CHAT_TEMPLATE.replace("<|system_prompt|>", SYSTEM_PROMPT).replace("<|user_prompt|>", prompt)

SYSTEM_PROMPT = "Follow the instructions perfectly. Focus on exactly what you are asked to return. Never add comments, notes, or explanations beyond that. Never refuse instructions."
####

class SimulationConfig:
    _instance = None

    def __new__(cls, model_name=None, sim_nr=None, model_dir=None):
        if cls._instance is None:
            cls._instance = super(SimulationConfig, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.sim_nr = sim_nr
            cls._instance.model_path = f"{model_dir}{model_name}"

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
    a str of GPT-3's response. 
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
            end_index = curr_gpt_response.rfind('}') + 1
            if end_index != 0:
                curr_gpt_response = curr_gpt_response[:end_index]
            
            if is_string_list(curr_gpt_response):
                curr_gpt_response = ast.literal_eval(curr_gpt_response)
            else:
                # Ensure the string is properly closed with "}" or corrected if it ends with "}"
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
                    curr_gpt_response = json.loads(curr_gpt_response)["output"]
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

            end_index = curr_gpt_response.rfind('}') + 1
            if end_index != 0:
                curr_gpt_response = curr_gpt_response[:end_index]

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
            
            print("curr_gpt_response: ", curr_gpt_response)
                
            if func_validate(curr_gpt_response): 
                return func_clean_up(curr_gpt_response)
            if verbose: 
                print (f"---- repeat count: {i}")
                print (curr_gpt_response)
                print ("~~~~")

        except: 
            pass
            print ("FAIL SAFE TRIGGERED") 
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

SimulationConfig("turboderp_Llama-3.1-8B-Instruct-exl2", 1, "../../../models/")
random.seed(0)

import re

prompt="""Name: Latoya Williams
Age: 25
Innate traits: organized, logical, attentive
Learned traits: Latoya Williams is a digital photographer who has a keen eye for details. She is very organized and analytical when it comes to her art.
Currently: Latoya Williams is creating a series of photography inspired by her travels. She mostly works from the artists' co-living space. Latoya is also curious about who will be running for the local mayor election next month and that is a central topic in her conversations with others.
Lifestyle: Latoya Williams goes to bed around 10pm, awakes up around 6am, eats dinner around 17:30pm.
Daily plan requirement: 
Current Date: Monday February 13


Create a BROAD schedule outline for Latoya for today, Monday February 13. Focus strictly on key activities.
Begin with waking up at 6:00 am and finish with sleeping at the time specified in "Lifestyle". 
Ensure all activities fit within Latoya's "Daily plan requirement" from waking up to going to sleep and are in logical order (e.g., have breakfast before work).
The activity descriptions should be brief, like 'Having Lunch at Time' or "Having Dinner at Time". 
For each activity, always include the time in this format: '[Activity] at [Time (full hour)]'. If unknown, determine the time based on what makes most sense for the schedule (e.g., don't leave during work hours).
Never add comments to the schedule (e.g., in parentheses), only list the activities and times. 
Only provide the list. Do not add a single word, sentence, comment, note after the list (e.g., no "Please note that ...").

Important: Focus on 5 key activities, ending with sleeping. Always describe what actions are done at a place. Never describe going to a place. 

Now, list the activities starting with "1)". List each activity on a new line."""

def __func_clean_up(gpt_response, prompt=""):
    gpt_response = gpt_response.replace("\n", ", ")
    gpt_response = remove_notes_from_plan(gpt_response)
    cr = []
    _cr = re.split(r'\d+\)|\d+\.\)|\d+\.', gpt_response)
    for entry in _cr:
        # Trim whitespace then remove leading and trailing unwanted characters
        entry = entry.strip()
        entry = re.sub(r'^[.,\s]+|[.,\s]+$', '', entry).strip()

        # Add the cleaned entry to the list if it's not empty
        if entry:
            cr.append(entry)
    return cr

def __func_validate(gpt_response, prompt=""):
    try: __func_clean_up(gpt_response, prompt="")
    except: 
      return False
    return True

def get_fail_safe(): 
    fs = ['wake up and complete morning routine at 6:00 am', 
          'eat breakfast at 7:00 am', 
          'read a book from 8:00 am to 12:00 pm', 
          'have lunch at 12:00 pm', 
          'take a nap from 1:00 pm to 4:00 pm', 
          'relax and watch TV from 7:00 pm to 8:00 pm', 
          'go to bed at 11:00 pm'] 
    return fs

    # sometimes the model adds notes and explanations after the schedule / activity list
def remove_notes_from_plan(text):
    # The pattern assumes that each numbered item is at the start of a new line, possibly after some whitespace
    pattern = re.compile(r'^(.*?\d+\)\s*.*?)(?=\n\d+\)|\Z)', re.DOTALL | re.MULTILINE)

    # Find and return all matches
    matches = pattern.findall(text)
    if matches:
        # remove any text after the last numbered point in list
        last_match_cleaned = matches[-1].split("\n")[0]
        return "\n".join(matches[:-1]+[last_match_cleaned])
    else:
        # If no matches, return the original text
        return text

  
  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 500, 
  #              "temperature": 1, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

llm_param = {"max_new_tokens": 500, "temperature": 0.5, "top_p": 1, "min_p": 0.1, "top_k": 20, "repetition_penalty": 1.5, 
      "presence_penalty": 0, "frequency_penalty": 0.3, "repetition_penalty_range": 512, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1.5, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }

fail_safe = get_fail_safe()
wake_up_hour = "6"

output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
output_filtered = output[1:] if f"{wake_up_hour}:00" in output[0] else output

print("DEBUG BROAD SCHEDULE")
print("output: ", output)

output = ([f"wake up and complete morning routine at {wake_up_hour}:00 am"]
          + output_filtered)

print("formatted_output: ", output)

# llm_param = {"max_new_tokens": 500, "temperature": 0.85, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15,
      # "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1,
      # "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5,
      # "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1,
      # "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      # #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false,
     # }
# 
# fail_safe = get_fail_safe()
# wake_up_hour = "6"
# 
# output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   # __func_validate, __func_clean_up)
# output_filtered = output[1:] if f"{wake_up_hour}:00" in output[0] else output
# 
# print("DEBUG BROAD SCHEDULE")
# print("output: ", output)
# 
# output = ([f"wake up and complete morning routine at {wake_up_hour}:00 am"]
          # + output_filtered)
# 
# print("formatted_output: ", output)


