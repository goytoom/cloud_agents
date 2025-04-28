import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling Local LLM APIs.
"""
# import openai
import time 
import sys
sys.path.append('../../')
sys.path.append('../')
import argparse 

# from generative_agents.reverie.backend_server.utils import *
# from utils import *   
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
            cls._instance.cache = ExLlamaV2Cache(cls._instance.model, max_seq_len = 16384, lazy = True)
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
        simulation_config._instance.GENERAL_PARAMS["stop_conditions"] = list(existing_tokens)
    params.update(simulation_config._instance.GENERAL_PARAMS)
    formatted_prompt = _format_prompt(prompt)
    output = simulation_config._instance.generator.generate(prompt = formatted_prompt, **params).strip()
    return output
  
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
        # print("CURR RESPONSE", curr_gpt_response)
        if func_validate(curr_gpt_response, prompt = prompt): 
            return func_clean_up(curr_gpt_response, prompt = prompt)
        if verbose: 
            print ("---- repeat count: ", i, curr_gpt_response)
            print (curr_gpt_response)
            print ("~~~~")
    return fail_safe_response

###############

def create_prompt_input(retrieved, 
                        test_input=None,
                        word_limit = None): 
  
  def truncate_context(context, word_limit=2500):
    words = context.split()  # Split context into words
    truncated_words = words[:word_limit]  # Take only the first 500 words
    truncated_context = " ".join(truncated_words)  # Recombine into a string
    return truncated_context


  context_l = ["Maria Lopez and Isabella Rodriguez are friends."]*1000
  context = " ".join(context_l)
  context_truncated = truncate_context(context, word_limit)

  curr_time = "Monday February 13, 2023, 08:00:00 AM1"
  init_act_desc = "is beginning her shift at Hobb's Cafe."
  init_persona_act_address = "the Ville:Hobb's Cafe:cafe"
  init_persona_planned_path = ""
  init_persona_name = "Isabella Rodriguez"
  if "(" in init_persona_act_address: 
    init_persona_act_address = init_persona_act_address.split("(")[-1][:-1]
  if len(init_persona_planned_path) == 0: 
    loc = ""
    if ":" in init_persona_act_address:
      loc = init_persona_act_address.split(":")[-1] + " in " + init_persona_act_address.split(":")[-2]
    init_p_desc = f"{init_persona_name} is already {init_persona_act_address} at {loc}"
  else: 
    loc = ""
    if ":" in init_persona_act_address:
      loc = init_persona_act_address.split(":")[-1] + " in " + init_persona_act_address.split(":")[-2]
    init_p_desc = f"{init_persona_name} is on the way to {init_persona_act_address} at {loc}"

  target_act_desc = "is getting her morning coffee"
  target_persona_act_address = "the Ville:Hobb's Cafe:cafe"
  target_persona_planned_path = ""
  target_persona_name = "Maria Lopez"

  if "(" in target_act_desc: 
    target_act_desc = target_act_desc.split("(")[-1][:-1]
  if len(target_persona_planned_path) == 0: 
    loc = ""
    if ":" in target_persona_act_address:
      loc = target_persona_act_address.split(":")[-1] + " in " + target_persona_act_address.split(":")[-2]
    target_p_desc = f"{target_persona_name} is already {target_act_desc} at {loc}"
  else: 
    loc = ""
    if ":" in target_persona_act_address:
      loc = target_persona_act_address.split(":")[-1] + " in " + target_persona_act_address.split(":")[-2]
    target_p_desc = f"{target_persona_name} is on the way to {target_act_desc} at {loc}"

  prompt_input = []
  prompt_input += [context_truncated]
  prompt_input += [curr_time]
  prompt_input += [init_p_desc]
  prompt_input += [target_p_desc]

  prompt_input += [init_persona_name]
  prompt_input += [init_act_desc]
  prompt_input += [target_persona_name]
  prompt_input += [target_act_desc]

  prompt_input += [init_act_desc]

  template_extra_words = """Task -- given context and two options that a subject can take, determine which option is the most acceptable. 

Context: Jane is Liz's house mate. Jane and Liz exchanged a conversation about saying good morning at 07:05am, October 25, 2022. 
Right now, it is 07:09 am, October 25, 2022. 
Jane was on her way to using the bathroom right now. 
Jane sees Liz already using the bathroom. 
My question: Let's think step by step. Of the following two options, what should Jane do?
Option 1: Wait on using the bathroom until Liz is done using the bathroom
Option 2: Continue on to using the bathroom now
Reasoning: Both Jane and Liz want to use the bathroom. 
It would be strange for both Jane and Liz to use the bathroom at the same time. 
So, since Liz is already using the bathroom, the best option for Jane is to wait on using the bathroom.
Answer: Option 1
---
Context: Sam is Sarah's friend. Sam and Sarah exchanged a conversation about favorite movies at 11pm, October 24, 2022. 
Right now, it is 12:40 pm, October 25, 2022. 
Sam is on the way to study for his test. 
Sam sees Sarah heading to do her laundry. 
My question: Let's think step by step. Of the following two options, what should Sam do?
Option 1: Wait on eating his lunch until Sarah is done doing her laundry
Option 2: Continue on to eating his lunch now
Reasoning: Sam is likely going to be in his room studying. Sarah, on the other hand, is likely headed to the laundry room for doing the laundry.
Since Sam and Sarah need to use different areas, their actions do not conflict. 
So, since Sam and Sarah are going to be in different areas, Sam mcan continue on to eating his lunch now.
Answer: Option 2
---
Context: !<INPUT 0>!
Right now, it is !<INPUT 1>!. 
!<INPUT 2>! 
!<INPUT 3>! 
My question: Let's think step by step. Of the following two options, what should !<INPUT 4>! do?
Option 1: Wait on !<INPUT 5>! until !<INPUT 6>! is done !<INPUT 7>!
Option 2: Continue on to !<INPUT 8>! now
Reasoning: """

  print(sum(len(text.split()) for text in prompt_input + [template_extra_words]))

  return prompt_input

def __func_validate(gpt_response, prompt=""): 
  try: 
    if gpt_response.split("Answer: Option")[-1].strip().lower() in ["3", "2", "1"]: 
      return True
    return False     
  except:
    return False 

def __func_clean_up(gpt_response, prompt=""):
  return gpt_response.split("Answer: Option")[-1].strip().lower() 

def get_fail_safe(): 
  fs = "2"
  return fs


if __name__ == "__main__":
    model_name = "turboderp_Llama-3.1-8B-Instruct-exl2"
    SIM_NR = "1"
    model_dir = "../../../models/"

    SimulationConfig(model_name, SIM_NR, model_dir)

    llm_param = {"max_new_tokens": 20, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
    "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
    "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
    "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
    "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
    #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
    }    

    ### code fails at this spot after long simulations. Check if prompt context growths too much.
    # Think about how to limit it!
    # 1) truncate long contexts
    # 2) increase context window
    # 3) check synthetic example (fill template with long context, see where it fails) 

    parser = argparse.ArgumentParser(description='Run Reverie simulations.')
    parser.add_argument('--wl', type=int, required=False, help='Word limit for truncation.', default=2500)
    args = parser.parse_args()

    word_limit = args.wl

    retrieved = "" 
    prompt_template = "persona/prompt_template/v2/decide_to_react_v1.txt"
    prompt_input = create_prompt_input(retrieved, None, word_limit)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if 1: 
        print("##### PROMPT #####")
        
        print(prompt)
        print("##### OUTPUT #####")
        print(output)
