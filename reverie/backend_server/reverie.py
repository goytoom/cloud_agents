"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: reverie.py
Description: This is the main program for running generative agent simulations
that defines the ReverieServer class. This class maintains and records all  
states related to the simulation. The primary mode of interaction for those  
running the simulation should be through the open_server function, which  
enables the simulator to input command-line prompts for running and saving  
the simulation, among other tasks.

Release note (June 14, 2023) -- Reverie implements the core simulation 
mechanism described in my paper entitled "Generative Agents: Interactive 
Simulacra of Human Behavior." If you are reading through these lines after 
having read the paper, you might notice that I use older terms to describe 
generative agents and their cognitive modules here. Most notably, I use the 
term "personas" to refer to generative agents, "associative memory" to refer 
to the memory stream, and "reverie" to refer to the overarching simulation 
framework.
"""
import json
import numpy
import pandas as pd
import datetime
import pickle
import time
import math
import os
from pathlib import Path
import shutil
import traceback
import random
random.seed(0)

# from selenium import webdriver
# from selenium.common.exceptions import WebDriverException

# from selenium.webdriver.chrome.options import Options as ChromeOptions
# from selenium.webdriver.chrome.service import Service as ChromeService
# from webdriver_manager.chrome import ChromeDriverManager

# from selenium.webdriver.firefox.options import Options as FirefoxOptions
# from selenium.webdriver.firefox.service import Service as FirefoxService
# from webdriver_manager.firefox import GeckoDriverManager

from global_methods import *
from utils import *
from maze import *
from persona.persona import *
from persona.prompt_template.run_gpt_prompt import *
from persona.prompt_template.gpt_structure import *
from transformers import set_seed

import argparse

# from persona.cognitive_modules.converse import _hire # does not work. workaround below
def _hire(hiring_persona, job_details_str=None):
  ### If still can_hire, check who is the most suited and give them the job
    ### Then update the learned of both (you have an employee helping you with X, you are employed by Y to help with X)
  
  ## prompt to go through offered candidates and choose one
    ## alternatively take highest rating
  ## if only 1, choose them.
  if len(hiring_persona.scratch.offer_info) == 1:
    chosen_candidate = list(hiring_persona.scratch.offer_info)[0]
  elif len(hiring_persona.scratch.offer_info) == 0:
    pass
  else:
    choice = run_gpt_prompt_hiring_choice(hiring_persona)
    chosen_candidate = list(hiring_persona.scratch.offer_info)[choice-1]
    
  chosen_employee_persona = hiring_persona.scratch.offer_info[chosen_candidate][-3]

  ## update employer and employee characters (i.e., add to their "character card")
  job_role_name = hiring_persona.scratch.job_details["Job role"]
  job_hours = hiring_persona.scratch.job_details["Ideal work hours"]
  job_location = hiring_persona.scratch.job_details["Job Location"]

  try:
    print(hiring_persona.scratch.name, " is being hired by: ", hiring_persona.scratch.name)
  except:
    print("hiring error!")
  hiring_persona.scratch.learned += f" {chosen_employee_persona.scratch.name} works for {hiring_persona.scratch.name} as a {job_role_name} at {job_location}." ### needs to add location!
  chosen_employee_persona.scratch.learned += f" {chosen_employee_persona.scratch.name} is also working for {hiring_persona.scratch.name} as a {job_role_name}."
  chosen_employee_persona.scratch.daily_plan_req += f" {chosen_employee_persona.scratch.name} has requested work hours from {job_hours}. They make the scheduling work with their other necessary committments."
  return chosen_candidate, chosen_employee_persona



##############################################################################
#                                  REVERIE                                   #
##############################################################################
fs_storage = "../../environment/frontend_server/storage"

class ReverieServer: 
  def __init__(self, 
               fork_sim_code,
               sim_code,
               sim_nr, browser_type, group_condition, group_distribution, save_folder, moral_experiment=None):
    # FORKING FROM A PRIOR SIMULATION:
    # <fork_sim_code> indicates the simulation we are forking from. 
    # Interestingly, all simulations must be forked from some initial 
    # simulation, where the first simulation is "hand-crafted".
    self.fork_sim_code = fork_sim_code
    fork_folder = f"{fs_storage}/{self.fork_sim_code}"
    self.group_condition = group_condition
    self.save_folder = save_folder
    self.moral_experiment = moral_experiment

    # <sim_code> indicates our current simulation. The first step here is to 
    # copy everything that's in <fork_sim_code>, but edit its 
    # reverie/meta/json's fork variable. 
    # sim_folder = f"{fs_storage}/{sim_code}"
      
    # # Attempt to copy, if fails ask for new target
    # while not copyanything(fork_folder, sim_folder):
    #     sim_code = input("Please enter a new name for the target simulation: ").strip()
    #     sim_folder = f"{fs_storage}/{sim_code}"

    self.sim_code = sim_code
    sim_folder = f"{fs_storage}/{self.sim_code}"
    copyanything(fork_folder, sim_folder)

    with open(f"{sim_folder}/reverie/meta.json") as json_file:  
      reverie_meta = json.load(json_file)

    with open(f"{sim_folder}/reverie/meta.json", "w") as outfile: 
      reverie_meta["fork_sim_code"] = fork_sim_code
      outfile.write(json.dumps(reverie_meta, indent=2))

    # LOADING REVERIE'S GLOBAL VARIABLES
    # The start datetime of the Reverie: 
    # <start_datetime> is the datetime instance for the start datetime of 
    # the Reverie instance. Once it is set, this is not really meant to 
    # change. It takes a string date in the following example form: 
    # "June 25, 2022"
    # e.g., ...strptime(June 25, 2022, "%B %d, %Y")
    self.start_time = datetime.datetime.strptime(
                        f"{reverie_meta['start_date']}, 00:00:00",  
                        "%B %d, %Y, %H:%M:%S")
    # <curr_time> is the datetime instance that indicates the game's current
    # time. This gets incremented by <sec_per_step> amount everytime the world
    # progresses (that is, everytime curr_env_file is recieved). 
    self.curr_time = datetime.datetime.strptime(reverie_meta['curr_time'], 
                                                "%B %d, %Y, %H:%M:%S")
    # <sec_per_step> denotes the number of seconds in game time that each 
    # step moves foward. 
    self.sec_per_step = reverie_meta['sec_per_step']
    
    # <maze> is the main Maze instance. Note that we pass in the maze_name
    # (e.g., "double_studio") to instantiate Maze. 
    # e.g., Maze("double_studio")
    self.maze = Maze(reverie_meta['maze_name'])
    
    # <step> denotes the number of steps that our game has taken. A step here
    # literally translates to the number of moves our personas made in terms
    # of the number of tiles. 
    self.step = reverie_meta['step']

    # SETTING UP PERSONAS IN REVERIE
    # <personas> is a dictionary that takes the persona's full name as its 
    # keys, and the actual persona instance as its values.
    # This dictionary is meant to keep track of all personas who are part of
    # the Reverie instance. 
    # e.g., ["Isabella Rodriguez"] = Persona("Isabella Rodriguezs")
    self.personas = dict()
    # <personas_tile> is a dictionary that contains the tile location of
    # the personas (!-> NOT px tile, but the actual tile coordinate).
    # The tile take the form of a set, (row, col). 
    # e.g., ["Isabella Rodriguez"] = (58, 39)
    self.personas_tile = dict()
    self.sim_nr = sim_nr
    self.driver = None
    self.browser_type = browser_type
    
    # # <persona_convo_match> is a dictionary that describes which of the two
    # # personas are talking to each other. It takes a key of a persona's full
    # # name, and value of another persona's full name who is talking to the 
    # # original persona. 
    # # e.g., dict["Isabella Rodriguez"] = ["Maria Lopez"]
    # self.persona_convo_match = dict()
    # # <persona_convo> contains the actual content of the conversations. It
    # # takes as keys, a pair of persona names, and val of a string convo. 
    # # Note that the key pairs are *ordered alphabetically*. 
    # # e.g., dict[("Adam Abraham", "Zane Xu")] = "Adam: baba \n Zane:..."
    # self.persona_convo = dict()


    ### Define feature function:
      ### adds features to each character card
    def add_features(features, feature_ranges):
      # features: name of feature to add
      # range: range from which to sample the value. If empty, use normal distribution from 0 to 1
      for persona_name in reverie_meta['persona_names']: 
        persona_card_path = f"{sim_folder}/personas/{persona_name}/bootstrap_memory/scratch.json"
        with open(persona_card_path, "r") as f:
          persona_card = json.load(f)
          persona_card["sim_nr"] = self.sim_nr
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

          persona_card["sim_nr"] = self.sim_nr

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
              
              threat_text = """<Name> does not feel economically threatened.
<Name> does not feel physically threatened.
<Name> does not feel that their values are under threat.
<Name> does not feel that their traditions are under threat."""

              threat_text = threat_text.replace("<Name>", persona_card["name"])
              persona_card["group_identity_text"] = group_text
              persona_card["threat_text"] = threat_text

          elif group_condition == 6: # both threats
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
<Name> strongly feels that their values are under threat by <Group 2>.
<Name> strongly feels that their traditions are under threat by <Group 2>."""
              threat_text = threat_text.replace("<Name>", persona_card["name"]).replace("<Group 2>", group_b)

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


    ### make interactive later
    # features = ["Attractiveness"]
    # feature_ranges = [[1,10]]
    features = []
    feature_ranges = []
    ## modify for group_condition == 0 (Study 1)
    if group_condition == 0:
      features = ["Physical_attractiveness", "Race", "Gender"]
      feature_ranges = [[1, 10], ["Black", "White", "Asian", "Middle Eastern", "Hispanic"], ["Male", "Female"]]

    add_features(features, feature_ranges) 

    #### Add moral conditions
    if moral_experiment == 1:
      moral_values_text = """""" ### Add stimulus for high binding
    elif moral_experiment == 2:
      moral_values_text = """""" ### Add stimulus for high individualizing
    else:
      moral_values_text = ""
    

    #### Add group condition / threat condition / group identity
      ## Group condition (add argument for group/no group)
      ## Threat condition (add argument for symbolic/material/no threat)
      ## Distribute group identity
    
    # group_condition = 1               ## turn into arguments for command line (1/0)
    # group_distribution = [0.5, 0.5]
    add_groups(group_condition, group_distribution)


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
    df_features.to_csv(f"{self.save_folder}/distributions_sim_{self.sim_nr}_{self.group_condition}.csv")

    # Loading in all personas. 
    init_env_file = f"{sim_folder}/environment/{str(self.step)}.json"
    init_env = json.load(open(init_env_file))
    for persona_name in reverie_meta['persona_names']: 
      persona_folder = f"{sim_folder}/personas/{persona_name}"
      p_x = init_env[persona_name]["x"]
      p_y = init_env[persona_name]["y"]

      ### Add modifcation of persona card here!
        ### Add features here
        ### Random distribution

      curr_persona = Persona(persona_name, persona_folder)

      self.personas[persona_name] = curr_persona
      self.personas_tile[persona_name] = (p_x, p_y)
      self.maze.tiles[p_y][p_x]["events"].add(curr_persona.scratch
                                              .get_curr_event_and_desc())

    # REVERIE SETTINGS PARAMETERS:  
    # <server_sleep> denotes the amount of time that our while loop rests each
    # cycle; this is to not kill our machine. 
    self.server_sleep = 0.1

    # SIGNALING THE FRONTEND SERVER: 
    # curr_sim_code.json contains the current simulation code, and
    # curr_step.json contains the current step of the simulation. These are 
    # used to communicate the code and step information to the frontend. 
    # Note that step file is removed as soon as the frontend opens up the 
    # simulation. 
    curr_sim_code = dict()
    curr_sim_code["sim_code"] = self.sim_code
    with open(f"{fs_temp_storage}/curr_sim_code.json", "w") as outfile: 
      outfile.write(json.dumps(curr_sim_code, indent=2))
    
    curr_step = dict()
    curr_step["step"] = self.step
    with open(f"{fs_temp_storage}/curr_step.json", "w") as outfile: 
      outfile.write(json.dumps(curr_step, indent=2))


  def save(self): 
    """
    Save all Reverie progress -- this includes Reverie's global state as well
    as all the personas.  

    INPUT
      None
    OUTPUT 
      None
      * Saves all relevant data to the designated memory directory
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # Save Reverie meta information.
    reverie_meta = dict() 
    reverie_meta["fork_sim_code"] = self.fork_sim_code
    reverie_meta["start_date"] = self.start_time.strftime("%B %d, %Y")
    reverie_meta["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
    reverie_meta["sec_per_step"] = self.sec_per_step
    reverie_meta["maze_name"] = self.maze.maze_name
    reverie_meta["persona_names"] = list(self.personas.keys())
    reverie_meta["step"] = self.step
    reverie_meta_f = f"{sim_folder}/reverie/meta.json"
    with open(reverie_meta_f, "w") as outfile: 
      outfile.write(json.dumps(reverie_meta, indent=2))

    # Save the personas.
    for persona_name, persona in self.personas.items(): 
      save_folder = f"{sim_folder}/personas/{persona_name}/bootstrap_memory"
      persona.save(save_folder)


  def start_path_tester_server(self): 
    """
    Starts the path tester server. This is for generating the spatial memory
    that we need for bootstrapping a persona's state. 

    To use this, you need to open server and enter the path tester mode, and
    open the front-end side of the browser. 

    INPUT 
      None
    OUTPUT 
      None
      * Saves the spatial memory of the test agent to the path_tester_env.json
        of the temp storage. 
    """
    def print_tree(tree): 
      def _print_tree(tree, depth):
        dash = " >" * depth

        if type(tree) == type(list()): 
          if tree:
            print (dash, tree)
          return 

        for key, val in tree.items(): 
          if key: 
            print (dash, key)
          _print_tree(val, depth+1)
      
      _print_tree(tree, 0)

    # <curr_vision> is the vision radius of the test agent. Recommend 8 as 
    # our default. 
    curr_vision = 8
    # <s_mem> is our test spatial memory. 
    s_mem = dict()

    # The main while loop for the test agent. 
    while (True): 
      try: 
        curr_dict = {}
        tester_file = fs_temp_storage + "/path_tester_env.json"
        if check_if_file_exists(tester_file): 
          with open(tester_file) as json_file: 
            curr_dict = json.load(json_file)
            os.remove(tester_file)
          
          # Current camera location
          curr_sts = self.maze.sq_tile_size
          curr_camera = (int(math.ceil(curr_dict["x"]/curr_sts)), 
                         int(math.ceil(curr_dict["y"]/curr_sts))+1)
          curr_tile_det = self.maze.access_tile(curr_camera)

          # Initiating the s_mem
          world = curr_tile_det["world"]
          if curr_tile_det["world"] not in s_mem: 
            s_mem[world] = dict()

          # Iterating throughn the nearby tiles.
          nearby_tiles = self.maze.get_nearby_tiles(curr_camera, curr_vision)
          for i in nearby_tiles: 
            i_det = self.maze.access_tile(i)
            if (curr_tile_det["sector"] == i_det["sector"] 
                and curr_tile_det["arena"] == i_det["arena"]): 
              if i_det["sector"] != "": 
                if i_det["sector"] not in s_mem[world]: 
                  s_mem[world][i_det["sector"]] = dict()
              if i_det["arena"] != "": 
                if i_det["arena"] not in s_mem[world][i_det["sector"]]: 
                  s_mem[world][i_det["sector"]][i_det["arena"]] = list()
              if i_det["game_object"] != "": 
                if (i_det["game_object"] 
                    not in s_mem[world][i_det["sector"]][i_det["arena"]]):
                  s_mem[world][i_det["sector"]][i_det["arena"]] += [
                                                         i_det["game_object"]]

        # Incrementally outputting the s_mem and saving the json file. 
        print ("= " * 15)
        out_file = fs_temp_storage + "/path_tester_out.json"
        with open(out_file, "w") as outfile: 
          outfile.write(json.dumps(s_mem, indent=2))
        print_tree(s_mem)

      except:
        pass

      time.sleep(self.server_sleep * 10)


  def start_server(self, int_counter): 
    """
    The main backend server of Reverie. 
    This function retrieves the environment file from the frontend to 
    understand the state of the world, calls on each personas to make 
    decisions based on the world state, and saves their moves at certain step
    intervals. 
    INPUT
      int_counter: Integer value for the number of steps left for us to take
                   in this iteration. 
    OUTPUT 
      None
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"
    sim_url = f'http://localhost:{8000 + self.sim_nr - 1}/simulator_home'

    # When a persona arrives at a game object, we give a unique event
    # to that object. 
    # e.g., ('double studio[...]:bed', 'is', 'unmade', 'unmade')
    # Later on, before this cycle ends, we need to return that to its 
    # initial state, like this: 
    # e.g., ('double studio[...]:bed', None, None, None)
    # So we need to keep track of which event we added. 
    # <game_obj_cleanup> is used for that. 
    game_obj_cleanup = dict()

    first_step = 1

    # The main while loop of Reverie. 
    while (True): 
      # Done with this iteration if <int_counter> reaches 0. 
      if int_counter == 0: 
        ### go over all employer-agents and make final hiring decision:
        # iterate over agents
        save_string = ""
        try:
          for boss_name, persona in self.personas.items(): 
            if persona.scratch.can_hire: # get employer-agents
              save_string += f"\nBoss Name: {boss_name}"
              ###########
              # if boss_name == "Arthur Burton": # for testing
              #   persona.scratch.offer_info = {"Klaus Mueller": ["Great interview. Need to hire!", "good boy", 8, self.personas["Klaus Mueller"], "You're in!"]}
              #   persona.scratch.offer_info = {"Maria Lopez": ["Great interview. Need to hire!", "good girl", 10, self.personas["Maria Lopez"], "You're in!"]}
              #   persona.scratch.job_details = {'Job role': 'Assistant', 'Job duties': 'Assisting with day-to-day operations', 'Ideal work hours': 'Mon-Sun, 8AM-4PM', 'Job Location': 'Home Office'}
              # ########
              if persona.scratch.offer_info: # update candidate list by removing people already hired elsewhere
                new_offer_info = {}
                for empl_persona_name, hiring_information in persona.scratch.offer_info.items():
                  _, _, _, _, empl_persona, _, _ = hiring_information
                  if empl_persona.scratch.can_be_hired:
                    new_offer_info.update({empl_persona_name: hiring_information})
                    save_string += f"\nSuccessfully updated hiring"
                  else:
                    save_string += f"\nSuccessfully updated hiring"

                # Update the offer_info list
                persona.scratch.offer_info = new_offer_info

                ## if someone can be hired hire them now!
                if persona.scratch.offer_info:
                  ## make decision
                  hired_candidate, hired_candidate_persona = _hire(persona)
                  _, empl_impressions, empl_rating, social_ratings, _, interview_transcript, _ = persona.scratch.offer_info[hired_candidate_persona.scratch.name]
                  print("Hiring Choice: ", hired_candidate)
                  persona.scratch.can_hire = False # turn hiring status off
                  hired_candidate_persona.scratch.can_be_hired = False
                  logging_info = {"interaction_count": persona.scratch.interact_counter, "interview_count": persona.scratch.interview_counter, "Decision_type": "Hiring", "Decision": True, "Init": persona, "Target": hired_candidate_persona, 
                                  "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": interview_transcript, "target_impression": empl_impressions, "target_rating": empl_rating, "target_rating_soc": social_ratings,
                                  "Group Condition": persona.scratch.group_condition, "Init Group": persona.scratch.group_identity, "Target Group": hired_candidate_persona.scratch.group_identity, "convo_hate": "", "interview_hate": ""} #fill placeholder
                  TRACKER.log_decision(logging_info, persona.scratch.sim_nr)
                  save_string += f"\nSuccfully hired. Employer Name: {boss_name}. Employee Name: {hired_candidate}"

          with open(f"{self.save_folder}/debug_sucess_{self.sim_nr}_{self.group_condition}.csv", "w") as f:
            f.write(save_string)
        except Exception as e:
          ### save something to file
          save_string = f"Something went wrong!\n{e}"
          with open(f"{self.save_folder}/debug_error_{self.sim_nr}_{self.group_condition}.csv", "w") as f:
            f.write(save_string)
          raise e

        break

      # <curr_env_file> file is the file that our frontend outputs. When the
      # frontend has done its job and moved the personas, then it will put a 
      # new environment file that matches our step count. That's when we run 
      # the content of this for loop. Otherwise, we just wait. 
      print("Current Step: ", self.step)
      curr_env_file = f"{sim_folder}/environment/{self.step}.json"
      if check_if_file_exists(curr_env_file):
        # If we have an environment file, it means we have a new perception
        # input to our personas. So we first retrieve it.
        try: 
          # Try and save block for robustness of the while loop.
          with open(curr_env_file) as json_file:
            new_env = json.load(json_file)
            env_retrieved = True
        except: 
          pass
      
        if env_retrieved: 
          # if first_step==1:
          #   profile_dir = f"../../profile_{self.sim_nr}"
          #   os.makedirs(profile_dir, exist_ok=True)  # Ensure the directory exists
          #   if self.browser_type == "firefox":
          #     options = FirefoxOptions()
          #     options.add_argument("--headless")
          #     options.set_preference("profile", profile_dir)
          #     options.set_preference("browser.cache.disk.enable", False)
          #     options.set_preference("browser.cache.memory.enable", False)

          #   else:
          #     options = ChromeOptions()
          #     options.add_argument("--headless=new")
          #     options.add_argument('--no-sandbox')
          #     options.add_argument('--disable-dev-shm-usage')
          #     options.set_preference("profile", profile_dir)

          #   try:
          #     if self.browser_type == "firefox":
          #       print("Loading Firefox")
          #       self.driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
          #       self.driver.get(sim_url)
          #       print("Load Webpage For Simulation")
          #     else:
          #       print("Loading Chrome")
          #       self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
          #       self.driver.get(sim_url)
          #       print("Load Webpage For Simulation")

          #   except WebDriverException as e:
          #     print("Browser Driver Failed with WebDriver Exception:", e)
          #   except Exception as e:
          #       print("An unexpected error occurred:", e)

            # first_step = 0

            # time.sleep(5)
          ############ Frontend (webbrowser)

          # This is where we go through <game_obj_cleanup> to clean up all 
          # object actions that were used in this cylce. 
          for key, val in game_obj_cleanup.items(): 
            # We turn all object actions to their blank form (with None). 
            self.maze.turn_event_from_tile_idle(key, val)
          # Then we initialize game_obj_cleanup for this cycle. 
          game_obj_cleanup = dict()

          # We first move our personas in the backend environment to match 
          # the frontend environment. 
          for persona_name, persona in self.personas.items(): 
            # <curr_tile> is the tile that the persona was at previously. 
            curr_tile = self.personas_tile[persona_name]
            # <new_tile> is the tile that the persona will move to right now,
            # during this cycle. 
            new_tile = (new_env[persona_name]["x"], 
                        new_env[persona_name]["y"])

            # We actually move the persona on the backend tile map here. 
            self.personas_tile[persona_name] = new_tile
            self.maze.remove_subject_events_from_tile(persona.name, curr_tile)
            self.maze.add_event_from_tile(persona.scratch
                                         .get_curr_event_and_desc(), new_tile)

            # Now, the persona will travel to get to their destination. *Once*
            # the persona gets there, we activate the object action.
            if not persona.scratch.planned_path: 
              # We add that new object action event to the backend tile map. 
              # At its creation, it is stored in the persona's backend. 
              game_obj_cleanup[persona.scratch
                               .get_curr_obj_event_and_desc()] = new_tile
              self.maze.add_event_from_tile(persona.scratch
                                     .get_curr_obj_event_and_desc(), new_tile)
              # We also need to remove the temporary blank action for the 
              # object that is currently taking the action. 
              blank = (persona.scratch.get_curr_obj_event_and_desc()[0], 
                       None, None, None)
              self.maze.remove_event_from_tile(blank, new_tile)   

          # Then we need to actually have each of the personas perceive and
          # move. The movement for each of the personas comes in the form of
          # x y coordinates where the persona will move towards. e.g., (50, 34)
          # This is where the core brains of the personas are invoked. 
          movements = {"persona": dict(), 
                       "meta": dict()}
          for persona_name, persona in self.personas.items(): 
            # <next_tile> is a x,y coordinate. e.g., (58, 9)
            # <pronunciatio> is an emoji. e.g., "\ud83d\udca4"
            # <description> is a string description of the movement. e.g., 
            #   writing her next novel (editing her novel) 
            #   @ double studio:double studio:common room:sofa
            next_tile, pronunciatio, description = persona.move(
              self.maze, self.personas, self.personas_tile[persona_name], 
              self.curr_time)
            movements["persona"][persona_name] = {}
            movements["persona"][persona_name]["movement"] = next_tile
            movements["persona"][persona_name]["pronunciatio"] = pronunciatio
            movements["persona"][persona_name]["description"] = description
            movements["persona"][persona_name]["chat"] = (persona
                                                          .scratch.chat)
          

            print("DEBUUUUUUUUUUUG START")
            print(persona_name)
            print(self.curr_time)
            print(description)
            print(pronunciatio)
            print("DEBUUUUUUUUUUUG END")

          # Include the meta information about the current stage in the 
          # movements dictionary. 
          movements["meta"]["curr_time"] = (self.curr_time 
                                             .strftime("%B %d, %Y, %H:%M:%S"))

          # We then write the personas' movements to a file that will be sent 
          # to the frontend server. 
          # Example json output: 
          # {"persona": {"Maria Lopez": {"movement": [58, 9]}},
          #  "persona": {"Klaus Mueller": {"movement": [38, 12]}}, 
          #  "meta": {curr_time: <datetime>}}
          curr_move_file = f"{sim_folder}/movement/{self.step}.json"
          curr_move_path = f"{sim_folder}/movement"
          if not os.path.exists(curr_move_path):
            os.makedirs(curr_move_path)
          print("TEST MOVEMENTS")
          # with open(curr_move_file, "w") as outfile: 
          #   outfile.write(json.dumps(movements, indent=2))

          environment = {}
          for persona_name, persona in self.personas.items():  
            environment[persona_name] = {"maze": "the_ville",
                                              "x": movements["persona"][persona_name]["movement"][0],
                                              "y": movements["persona"][persona_name]["movement"][1]}

          # logging.info(f"Process environment polling: Simulation {sim_folder}, Step {step}")

          # After this cycle, the world takes one step forward, and the 
          # current time moves by <sec_per_step> amount. 
          self.step += 1
          self.curr_time += datetime.timedelta(seconds=self.sec_per_step)

          with open(f"{sim_folder}/environment/{self.step}.json", "w") as outfile:
            outfile.write(json.dumps(environment, indent=2))

          int_counter -= 1

          if not int_counter % 360: # save once every in simulation hour
            try:
              TRACKER.save(f"{self.save_folder}/experiment_results_{self.group_condition}_{self.sim_nr}.csv") 
            except Exception as e:
              print("ERROR SAVING: ", e)
              print("EMERGENCY PICKLE")
              # if something goes wrong save the whole class instance and figure out error later (prevent all data from being lost)
              with open(f'{self.save_folder}/emergency_save_{self.group_condition}_{self.sim_nr}.pickle', 'w') as f:
                  pickle.dump(TRACKER,f)

          ### check if everyone hired (stop after all hiring choices)
          # still_running = 0
          # for persona_name, persona in self.personas.items():
          #   if persona.scratch.can_hire:
          #     still_running += 1
          #   else:
          #     pass
          # if not still_running:
          #   ## End simulation
          #   int_counter = 0
          #   print("ALL HIRING CHOICES MADE")
          #   print("END SIMULATION AT: ", self.curr_time.strftime("%B %d, %Y, %H:%M:%S"))

          
      # Sleep so we don't burn our machines. 
      time.sleep(self.server_sleep)

      ########## Check if this works well (starting server, open website, etc)
        ## Might need to be

  def open_server(self): 
    """
    Open up an interactive terminal prompt that lets you run the simulation 
    step by step and probe agent state. 

    INPUT 
      None
    OUTPUT
      None
    """
    print ("Note: The agents in this simulation package are computational")
    print ("constructs powered by generative agents architecture and LLM. We")
    print ("clarify that these agents lack human-like agency, consciousness,")
    print ("and independent decision-making.\n---")

    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # modify this part to run x hours and then exit in a loop
      ## need to figure out how to reload webpage (probably just some package to open brows/open new tab and open page)

    # sim_command = "run 360" # test 10 hrs
    sim_command = "run 21600" # 17280 = 48hrs in simulation (2 full days)
    sim_command = sim_command.strip()
    ret_str = ""

    duration = round(int(sim_command.split("run ")[-1])/360, 2)

    print("Simulation Max Duration In Hours (internal time): ", duration)

    ## add automatic saving after finishing the simulation
    try: 
      if sim_command.lower() in ["f", "fin", "finish", "save and finish"]: 
        # Finishes the simulation environment and saves the progress. 
        # Example: fin
        self.save()

      elif sim_command.lower() == "start path tester mode": 
        # Starts the path tester and removes the currently forked sim files.
        # Note that once you start this mode, you need to exit out of the
        # session and restart in case you want to run something else. 
        shutil.rmtree(sim_folder) 
        self.start_path_tester_server()

      elif sim_command.lower() == "exit": 
        # Finishes the simulation environment but does not save the progress
        # and erases all saved data from current simulation. 
        # Example: exit 
        shutil.rmtree(sim_folder) 

      elif sim_command.lower() == "save": 
        # Saves the current simulation progress. 
        # Example: save
        self.save()

      ## Have this first with fixed int_count
        ## afterwards save (add sim_nr to the save file: experiment_test_simNr)
        ## add starting of webbrowser after starting the sim (find right point to start/reload)
          ## First sim 
            ## start server 
            ## open browser pointing to sim page
          ## Next sims/iterations
            # restart server
            # at right point reload sim page
      elif sim_command[:3].lower() == "run": 
        # Runs the number of steps specified in the prompt.
        # Example: run 1000
        int_count = int(sim_command.split()[-1])
        self.start_server(int_count) # check rs vs self. use here
        self.save()

      elif ("print persona schedule" 
            in sim_command[:22].lower()): 
        # Print the decomposed schedule of the persona specified in the 
        # prompt.
        # Example: print persona schedule Isabella Rodriguez
        ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                    .scratch.get_str_daily_schedule_summary())

      elif ("print all persona schedule" 
            in sim_command[:26].lower()): 
        # Print the decomposed schedule of all personas in the world. 
        # Example: print all persona schedule
        for persona_name, persona in self.personas.items(): 
          ret_str += f"{persona_name}\n"
          ret_str += f"{persona.scratch.get_str_daily_schedule_summary()}\n"
          ret_str += f"---\n"

      elif ("print hourly org persona schedule" 
            in sim_command.lower()): 
        # Print the hourly schedule of the persona specified in the prompt.
        # This one shows the original, non-decomposed version of the 
        # schedule.
        # Ex: print persona schedule Isabella Rodriguez
        ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                    .scratch.get_str_daily_schedule_hourly_org_summary())

      elif ("print persona current tile" 
            in sim_command[:26].lower()): 
        # Print the x y tile coordinate of the persona specified in the 
        # prompt. 
        # Ex: print persona current tile Isabella Rodriguez
        ret_str += str(self.personas[" ".join(sim_command.split()[-2:])]
                    .scratch.curr_tile)

      elif ("print persona chatting with buffer" 
            in sim_command.lower()): 
        # Print the chatting with buffer of the persona specified in the 
        # prompt.
        # Ex: print persona chatting with buffer Isabella Rodriguez
        curr_persona = self.personas[" ".join(sim_command.split()[-2:])]
        for p_n, count in curr_persona.scratch.chatting_with_buffer.items(): 
          ret_str += f"{p_n}: {count}"

      elif ("print persona associative memory (event)" 
            in sim_command.lower()):
        # Print the associative memory (event) of the persona specified in
        # the prompt
        # Ex: print persona associative memory (event) Isabella Rodriguez
        ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
        ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                      .a_mem.get_str_seq_events())

      elif ("print persona associative memory (thought)" 
            in sim_command.lower()): 
        # Print the associative memory (thought) of the persona specified in
        # the prompt
        # Ex: print persona associative memory (thought) Isabella Rodriguez
        ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
        ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                      .a_mem.get_str_seq_thoughts())

      elif ("print persona associative memory (chat)" 
            in sim_command.lower()): 
        # Print the associative memory (chat) of the persona specified in
        # the prompt
        # Ex: print persona associative memory (chat) Isabella Rodriguez
        ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
        ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                      .a_mem.get_str_seq_chats())

      elif ("print persona spatial memory" 
            in sim_command.lower()): 
        # Print the spatial memory of the persona specified in the prompt
        # Ex: print persona spatial memory Isabella Rodriguez
        self.personas[" ".join(sim_command.split()[-2:])].s_mem.print_tree()

      elif ("print current time" 
            in sim_command[:18].lower()): 
        # Print the current time of the world. 
        # Ex: print current time
        ret_str += f'{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}\n'
        ret_str += f'steps: {self.step}'

      elif ("print tile event" 
            in sim_command[:16].lower()): 
        # Print the tile events in the tile specified in the prompt 
        # Ex: print tile event 50, 30
        cooordinate = [int(i.strip()) for i in sim_command[16:].split(",")]
        for i in self.maze.access_tile(cooordinate)["events"]: 
          ret_str += f"{i}\n"

      elif ("print tile details" 
            in sim_command.lower()): 
        # Print the tile details of the tile specified in the prompt 
        # Ex: print tile event 50, 30
        cooordinate = [int(i.strip()) for i in sim_command[18:].split(",")]
        for key, val in self.maze.access_tile(cooordinate).items(): 
          ret_str += f"{key}: {val}\n"

      elif ("call -- analysis" 
            in sim_command.lower()): 
        # Starts a stateless chat session with the agent. It does not save 
        # anything to the agent's memory. 
        # Ex: call -- analysis Isabella Rodriguez
        persona_name = sim_command[len("call -- analysis"):].strip() 
        self.personas[persona_name].open_convo_session("analysis")

      elif ("call -- load history" 
            in sim_command.lower()): 
        curr_file = maze_assets_loc + "/" + sim_command[len("call -- load history"):].strip() 
        # call -- load history the_ville/agent_history_init_n3.csv

        rows = read_file_to_list(curr_file, header=True, strip_trail=True)[1]
        clean_whispers = []
        for row in rows: 
          agent_name = row[0].strip() 
          whispers = row[1].split(";")
          whispers = [whisper.strip() for whisper in whispers]
          for whisper in whispers: 
            clean_whispers += [[agent_name, whisper]]

        load_history_via_whisper(self.personas, clean_whispers)

      print (ret_str)

    except:
      traceback.print_exc()
      print ("Error.")
      pass

def main_process(model_name, SIM_NR, model_dir, origin, browser_type, group_condition, group_distribution, save_folder):
  from tracker_instance import TRACKER

  # Initialize the singleton object with the parameters
  SimulationConfig(model_name, SIM_NR, model_dir)

  random.seed(SIM_NR) 
  np.random.seed(SIM_NR)
  set_seed = SIM_NR
  print(f"TEST RANDOMIZER. Seed ({SIM_NR}): ", round(random.uniform(1,100), 1), "Group Condition: ", group_condition)
  print("Current Simulation: ", SIM_NR+1)
  target = f"sim_test_{group_condition}_{SIM_NR+1}" #input("Enter the name of the new simulation: ").strip() # add nr for each sim

  rs = ReverieServer(origin, target, SIM_NR+1, browser_type, group_condition, group_distribution, save_folder)

  try:
    rs.open_server()
    # rs.driver.quit()
  finally:
    if rs.driver:
      rs.driver.quit()
      print("Browser closed after simulation.")

  print("Save simulation results")
  
  try:
    TRACKER.save(f"{save_folder}/experiment_results_{group_condition}_{SIM_NR+1}.csv") 
  except Exception as e:
    print("ERROR SAVING: ", e)
    print("EMERGENCY PICKLE")
    # if something goes wrong save the whole class instance and figure out error later (prevent all data from being lost)
    with open(f'{save_folder}/emergency_save_{group_condition}_{SIM_NR+1}.pickle', 'w') as f:
        pickle.dump(TRACKER,f)

if __name__ == '__main__':
  # Argument parsing
  parser = argparse.ArgumentParser(description='Run Reverie simulations.')
  parser.add_argument('--model_name', type=str, required=False, help='Model name to be used for the experiment.', default="Mistral-7B-Instruct-v0.3-exl2")
  parser.add_argument('--sim_nr', type=int, required=False, help='Number of simulations to run.', default=0)
  parser.add_argument('--model_dir', type=str, required=False, help='Directory where the LLMs are stored.', default="../../../models/")
  parser.add_argument('--gc', type=int, required=False, help='Group condition. Yes/No.', default=1)
  parser.add_argument('--gdist', type=list, required=False, help='Group distribution.', default=[0.5, 0.5])

  # default="turboderp_Llama-3.1-8B-Instruct-exl2"
  args = parser.parse_args()

  model_name = args.model_name
  SIM_NR = args.sim_nr
  model_dir = args.model_dir
  group_distribution = np.asarray(args.gdist)
  group_condition = args.gc

  ### Check seeds for random generation
  origin = "base_the_ville_n25" #input("Enter the name of the forked simulation: ").strip()
  browser_type = "firefox" #firefox, or chrome (default)
  save_folder = f"../../sim_results_conflicts"
  Path(save_folder).mkdir(parents=True, exist_ok=True)

  # start the sim with different random seeds
  # this will lead to the feature distribution changing with every additional simulation
  # but the simulation sequence (i.e. SIM_1, SIM_2, ..., should be identitical)
  
  start = time.time()
  main_process(model_name, SIM_NR, model_dir, origin, browser_type, group_condition, group_distribution, save_folder)

  print(f"Simulation completed in {round(time.time() - start, 2)}s")
