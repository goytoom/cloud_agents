"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: plan.py
Description: This defines the "Plan" module for generative agents. 
"""
import datetime
import math
import random 
import sys
import time
sys.path.append('../../')

from global_methods import *
from persona.prompt_template.run_gpt_prompt import *
from persona.cognitive_modules.retrieve import *
from persona.cognitive_modules.converse import *
from functools import reduce
import re

##############################################################################
# CHAPTER 2: Generate
##############################################################################

def generate_wake_up_hour(persona):
  """
  Generates the time when the persona wakes up. This becomes an integral part
  of our process for generating the persona's daily plan.
  
  Persona state: identity stable set, lifestyle, first_name

  INPUT: 
    persona: The Persona class instance 
  OUTPUT: 
    an integer signifying the persona's wake up hour
  EXAMPLE OUTPUT: 
    8
  """
  if debug: print ("GNS FUNCTION: <generate_wake_up_hour>")
  return int(run_gpt_prompt_wake_up_hour(persona)[0])


def generate_first_daily_plan(persona, wake_up_hour): 
  """
  Generates the daily plan for the persona. 
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form: 
  'wake up and complete the morning routine at 6:00 am', 
  'eat breakfast at 7:00 am',.. 
  Note that the actions come without a period. 

  Persona state: identity stable set, lifestyle, cur_data_str, first_name

  INPUT: 
    persona: The Persona class instance 
    wake_up_hour: an integer that indicates when the hour the persona wakes up 
                  (e.g., 8)
  OUTPUT: 
    a list of daily actions in broad strokes.
  EXAMPLE OUTPUT: 
    ['wake up and complete the morning routine at 6:00 am', 
     'have breakfast and brush teeth at 6:30 am',
     'work on painting project from 8:00 am to 12:00 pm', 
     'have lunch at 12:00 pm', 
     'take a break and watch TV from 2:00 pm to 4:00 pm', 
     'work on painting project from 4:00 pm to 6:00 pm', 
     'have dinner at 6:00 pm', 'watch TV from 7:00 pm to 8:00 pm']
  """
  if debug: print ("GNS FUNCTION: <generate_first_daily_plan>")
  return run_gpt_prompt_daily_plan(persona, wake_up_hour)[0]


def generate_hourly_schedule(persona, wake_up_hour): 
  """
  Based on the daily req, creates an hourly schedule -- one hour at a time. 
  The form of the action for each of the hour is something like below: 
  "sleeping in her bed"
  
  The output is basically meant to finish the phrase, "x is..."

  Persona state: identity stable set, daily_plan

  INPUT: 
    persona: The Persona class instance 
    persona: Integer form of the wake up hour for the persona.  
  OUTPUT: 
    a list of activities and their duration in minutes: 
  EXAMPLE OUTPUT: 
    [['sleeping', 360], ['waking up and starting her morning routine', 60], 
     ['eating breakfast', 60],..
  """
  def clean_string(text, name_var, last_name_var):
    # Escape the name and last_name variables to ensure they're treated as literal strings in the regex
    name_var_escaped = re.escape(name_var)
    last_name_var_escaped = re.escape(last_name_var)
    
    # Construct the regex pattern to match "name_var is", "name_var last_name_var is" (if last_name_var is present in the text),
    # or "is" at the beginning, all in a case-insensitive manner
    # This uses a conditional pattern for including the last name if it's present in the text
    pattern = re.compile(
        rf"^(?:{name_var_escaped}\s+(?:{last_name_var_escaped}\s+)?is\s+|{last_name_var_escaped}\s+is\s+|is\s+)",
        re.IGNORECASE
    )
    
    # Remove the matched pattern if it's at the beginning
    cleaned_activity = re.sub(pattern, '', text).strip()
    
    # Return the cleaned string
    return cleaned_activity
      
  if debug: print ("GNS FUNCTION: <generate_hourly_schedule>")

  hour_str = ["00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", 
              "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", 
              "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", 
              "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
              "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]
  n_m1_activity = []
  diversity_repeat_count = 3
  filter_list = [persona.scratch.get_str_firstname(), f"{persona.scratch.get_str_firstname()} is", 
                 f"{persona.scratch.get_str_lastname()}", f"{persona.scratch.get_str_lastname()} is"]
  for i in range(diversity_repeat_count): 
    n_m1_activity_set = set(n_m1_activity)
    last_action = ""
    if len(n_m1_activity_set) < 5: 
      n_m1_activity = []
      for count, curr_hour_str in enumerate(hour_str): 
        if wake_up_hour > 0: 
          n_m1_activity += ["sleeping"]
          wake_up_hour -= 1
        else: 
            if any(substring in last_action for substring in ("going to bed", "sleep")): #keep sleeping if decided to sleep today
              activity = "sleeping"
              n_m1_activity += [activity]
              last_action = activity
            else:
              # print("n_m1_activity: ", n_m1_activity)
              llm_output = run_gpt_prompt_generate_hourly_schedule(
                  persona, curr_hour_str, n_m1_activity, hour_str)[0]
              activity = llm_output.split("\n")[0].split("Activity:")[-1]
              # print("activity", activity)
              cleaned_string = clean_string(activity.strip(), persona.scratch.get_str_firstname(), persona.scratch.get_str_lastname())
              # print("cleaned_string", cleaned_string)
              # cleaned_string = reduce(lambda activity,loc: activity.replace(loc,''), [activity]+filter_list).strip()
              n_m1_activity += [cleaned_string]
              last_action = cleaned_string

            # n_m1_activity += [run_gpt_prompt_generate_hourly_schedule(
            #                 persona, curr_hour_str, n_m1_activity, hour_str)[0]]
  
  # Step 1. Compressing the hourly schedule to the following format: 
  # The integer indicates the number of hours. They should add up to 24. 
  # [['sleeping', 6], ['waking up and starting her morning routine', 1], 
  # ['eating breakfast', 1], ['getting ready for the day', 1], 
  # ['working on her painting', 2], ['taking a break', 1], 
  # ['having lunch', 1], ['working on her painting', 3], 
  # ['taking a break', 2], ['working on her painting', 2], 
  # ['relaxing and watching TV', 1], ['going to bed', 1], ['sleeping', 2]]
  _n_m1_hourly_compressed = []
  prev = None 
  prev_count = 0
  for i in n_m1_activity: 
    if i != prev:
      prev_count = 1 
      _n_m1_hourly_compressed += [[i, prev_count]]
      prev = i
    else: 
      if _n_m1_hourly_compressed: 
        _n_m1_hourly_compressed[-1][1] += 1

  # Step 2. Expand to min scale (from hour scale)
  # [['sleeping', 360], ['waking up and starting her morning routine', 60], 
  # ['eating breakfast', 60],..
  n_m1_hourly_compressed = []
  for task, duration in _n_m1_hourly_compressed: 
    n_m1_hourly_compressed += [[task, duration*60]]

  return n_m1_hourly_compressed


def generate_task_decomp(persona, task, duration, match_index): 
  """
  A few shot decomposition of a task given the task description 

  Persona state: identity stable set, curr_date_str, first_name

  INPUT: 
    persona: The Persona class instance 
    task: the description of the task at hand in str form
          (e.g., "waking up and starting her morning routine")
    duration: an integer that indicates the number of minutes this task is 
              meant to last (e.g., 60)
  OUTPUT: 
    a list of list where the inner list contains the decomposed task 
    description and the number of minutes the task is supposed to last. 
  EXAMPLE OUTPUT: 
    [['going to the bathroom', 5], ['getting dressed', 5], 
     ['eating breakfast', 15], ['checking her email', 5], 
     ['getting her supplies ready for the day', 15], 
     ['starting to work on her painting', 15]] 

  """
  if debug: print ("GNS FUNCTION: <generate_task_decomp>")
  return run_gpt_prompt_task_decomp(persona, task, duration, match_index)[0]


def generate_action_sector(act_desp, persona, maze): 
  """TODO 
  Given the persona and the task description, choose the action_sector. 

  Persona state: identity stable set, n-1 day schedule, daily plan

  INPUT: 
    act_desp: description of the new action (e.g., "sleeping")
    persona: The Persona class instance 
  OUTPUT: 
    action_arena (e.g., "bedroom 2")
  EXAMPLE OUTPUT: 
    "bedroom 2"
  """
  if debug: print ("GNS FUNCTION: <generate_action_sector>")
  return run_gpt_prompt_action_sector(act_desp, persona, maze)[0]


def generate_action_arena(act_desp, persona, maze, act_world, act_sector): 
  """TODO 
  Given the persona and the task description, choose the action_arena. 

  Persona state: identity stable set, n-1 day schedule, daily plan

  INPUT: 
    act_desp: description of the new action (e.g., "sleeping")
    persona: The Persona class instance 
  OUTPUT: 
    action_arena (e.g., "bedroom 2")
  EXAMPLE OUTPUT: 
    "bedroom 2"
  """
  if debug: print ("GNS FUNCTION: <generate_action_arena>")
  return run_gpt_prompt_action_arena(act_desp, persona, maze, act_world, act_sector)[0]


def generate_action_game_object(act_desp, act_address, persona, maze):
  """TODO
  Given the action description and the act address (the address where
  we expect the action to task place), choose one of the game objects. 

  Persona state: identity stable set, n-1 day schedule, daily plan

  INPUT: 
    act_desp: the description of the action (e.g., "sleeping")
    act_address: the arena where the action will take place: 
               (e.g., "dolores double studio:double studio:bedroom 2")
    persona: The Persona class instance 
  OUTPUT: 
    act_game_object: 
  EXAMPLE OUTPUT: 
    "bed"
  """
  if debug: print ("GNS FUNCTION: <generate_action_game_object>")
  if not persona.s_mem.get_str_accessible_arena_game_objects(act_address): 
    return "<random>"
  return run_gpt_prompt_action_game_object(act_desp, persona, maze, act_address)[0]


def generate_action_pronunciatio(act_desp, persona): 
  """TODO 
  Given an action description, creates an emoji string description via a few
  shot prompt. 

  Does not really need any information from persona. 

  INPUT: 
    act_desp: the description of the action (e.g., "sleeping")
    persona: The Persona class instance
  OUTPUT: 
    a string of emoji that translates action description.
  EXAMPLE OUTPUT: 
    "🧈🍞"
  """
  if debug: print ("GNS FUNCTION: <generate_action_pronunciatio>")
  try: 
    x = run_gpt_prompt_pronunciatio(act_desp, persona)[0]
  except: 
    x = "🙂"

  if not x: 
    return "🙂"
  return x


def generate_action_event_triple(act_desp, persona): 
  """TODO 

  INPUT: 
    act_desp: the description of the action (e.g., "sleeping")
    persona: The Persona class instance
  OUTPUT: 
    a string of emoji that translates action description.
  EXAMPLE OUTPUT: 
    "🧈🍞"
  """
  if debug: print ("GNS FUNCTION: <generate_action_event_triple>")
  return run_gpt_prompt_event_triple(act_desp, persona)[0]


def generate_act_obj_desc(act_game_object, act_desp, persona): 
  if debug: print ("GNS FUNCTION: <generate_act_obj_desc>")
  return run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona)[0]


def generate_act_obj_event_triple(act_game_object, act_obj_desc, persona): 
  if debug: print ("GNS FUNCTION: <generate_act_obj_event_triple>")
  return run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona)[0]


def generate_convo(maze, init_persona, target_persona): 
  curr_loc = maze.access_tile(init_persona.scratch.curr_tile)

  # convo = run_gpt_prompt_create_conversation(init_persona, target_persona, curr_loc)[0]
  # convo = agent_chat_v1(maze, init_persona, target_persona)
  convo = agent_chat_v2(maze, init_persona, target_persona)

  curr_chat_transcript = ""
  for row in convo: 
    speaker = row[0]
    utt = row[1]
    curr_chat_transcript += f"{speaker}: {utt}\n"

  hate_reflection = reflect_convo(init_persona, target_persona, curr_chat_transcript)
  init_hate = hate_reflection["init"]
  target_hate = hate_reflection["target"]
  convo_hate = any([init_hate, target_hate])

  logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "Convo", "Decision": True, "Init": init_persona, "Target": target_persona, 
                      "Sim_Time": init_persona.scratch.curr_time, "Convo": curr_chat_transcript, "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": init_persona.scratch.group_condition, 
                      "Init Group": init_persona.scratch.group_identity, "Target Group": target_persona.scratch.group_identity, "convo_hate": convo_hate, "interview_hate": "", "act_hate": "", "act_location": "", }
  TRACKER.log_decision(logging_info, init_persona.scratch.sim_nr)

  all_utt = ""

  for row in convo: 
    speaker = row[0]
    utt = row[1]
    all_utt += f"{speaker}: {utt}\n"

  convo_length = math.ceil(int(len(all_utt)/8) / 30)

  if debug: print ("GNS FUNCTION: <generate_convo>")
  return convo, convo_length


def generate_convo_summary(persona, convo): 
  convo_summary = run_gpt_prompt_summarize_conversation(persona, convo)[0]
  return convo_summary


def generate_decide_to_talk(init_persona, target_persona, retrieved): 
  ## Log decision not to talk here!
    ## Check how other information can be retreived for logging
  x =run_gpt_prompt_decide_to_talk(init_persona, target_persona, retrieved)[0]
  if debug: print ("GNS FUNCTION: <generate_decide_to_talk>")

  if x == "yes": 
    return True
  else: 
    return False


def generate_decide_to_react(init_persona, target_persona, retrieved): 
  if debug: print ("GNS FUNCTION: <generate_decide_to_react>")
  return run_gpt_prompt_decide_to_react(init_persona, target_persona, retrieved)[0]

### Add reflection of action / classification for violence here
  ### Try to link it to the approaching of persona at pub, cafe, supermarket
def generate_new_decomp_schedule(persona, inserted_act, inserted_act_dur,  start_hour, end_hour): 
  # Step 1: Setting up the core variables for the function. 
  # <p> is the persona whose schedule we are editing right now. 
  p = persona
  # <today_min_pass> indicates the number of minutes that have passed today. 
  today_min_pass = (int(p.scratch.curr_time.hour) * 60 
                    + int(p.scratch.curr_time.minute) + 1)
  
  # Step 2: We need to create <main_act_dur> and <truncated_act_dur>. 
  # These are basically a sub-component of <f_daily_schedule> of the persona,
  # but focusing on the current decomposition. 
  # Here is an example for <main_act_dur>: 
  # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
  # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
  # ['wakes up and completes her morning routine (uses the restroom)', 5]
  # ['wakes up and completes her morning routine (washes her ...)', 10]
  # ['wakes up and completes her morning routine (makes her bed)', 5]
  # ['wakes up and completes her morning routine (eats breakfast)', 15]
  # ['wakes up and completes her morning routine (gets dressed)', 10]
  # ['wakes up and completes her morning routine (leaves her ...)', 5]
  # ['wakes up and completes her morning routine (starts her ...)', 5]
  # ['preparing for her day (waking up at 6am)', 5]
  # ['preparing for her day (making her bed)', 5]
  # ['preparing for her day (taking a shower)', 15]
  # ['preparing for her day (getting dressed)', 5]
  # ['preparing for her day (eating breakfast)', 10]
  # ['preparing for her day (brushing her teeth)', 5]
  # ['preparing for her day (making coffee)', 5]
  # ['preparing for her day (checking her email)', 5]
  # ['preparing for her day (starting to work on her painting)', 5]
  # 
  # And <truncated_act_dur> concerns only until where an event happens. 
  # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
  # ['wakes up and completes her morning routine (wakes up at 6am)', 2]
  main_act_dur = []
  truncated_act_dur = []
  dur_sum = 0 # duration sum
  count = 0 # enumerate count
  truncated_fin = False 

  print ("DEBUG::: ", persona.scratch.name)
  for act, dur in p.scratch.f_daily_schedule: 
    if (dur_sum >= start_hour * 60) and (dur_sum < end_hour * 60): 
      main_act_dur += [[act, dur]]
      if dur_sum <= today_min_pass:
        truncated_act_dur += [[act, dur]]
      elif dur_sum > today_min_pass and not truncated_fin: 
        # We need to insert that last act, duration list like this one: 
        # e.g., ['wakes up and completes her morning routine (wakes up...)', 2]
        truncated_act_dur += [[p.scratch.f_daily_schedule[count][0], 
                               dur_sum - today_min_pass]] 
        truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass) ######## DEC 7 DEBUG;.. is the +1 the right thing to do??? 
        # truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass + 1) ######## DEC 7 DEBUG;.. is the +1 the right thing to do??? 
        print ("DEBUG::: ", truncated_act_dur)

        # truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass) ######## DEC 7 DEBUG;.. is the +1 the right thing to do??? 
        truncated_fin = True
    dur_sum += dur
    count += 1

  persona_name = persona.name 
  main_act_dur = main_act_dur

  try:
    x = truncated_act_dur[-1][0].split("(")[0].strip() + " (on the way to " + truncated_act_dur[-1][0].split("(")[-1][:-1] + ")"
    truncated_act_dur[-1][0] = x 
  except IndexError as e:
      print("Index error: truncated_act_dur may be empty or not properly formatted.", e)

      print("DEBUG:truncated_act_dur: ", truncated_act_dur)
      print("DEBUG:main_act_dur: ", main_act_dur)
      print ("DEBUG:start_hour: ", start_hour)
      print ("DEBUG:end_hour: ", end_hour)

      x = "Engaged in thought (on the way to making a decision)"
      truncated_act_dur = [[x, 60]] 
      if not main_act_dur:
        main_act_dur = truncated_act_dur
      

  if "(" in truncated_act_dur[-1][0]: 
    inserted_act = truncated_act_dur[-1][0].split("(")[0].strip() + " (" + inserted_act + ")"

  # To do inserted_act_dur+1 below is an important decision but I'm not sure
  # if I understand the full extent of its implications. Might want to 
  # revisit. 
  truncated_act_dur += [[inserted_act, inserted_act_dur]]
  start_time_hour = (datetime.datetime(2022, 10, 31, 0, 0) 
                   + datetime.timedelta(hours=start_hour))
  end_time_hour = (datetime.datetime(2022, 10, 31, 0, 0) 
                   + datetime.timedelta(hours=end_hour))

  if debug: print ("GNS FUNCTION: <generate_new_decomp_schedule>")
  return run_gpt_prompt_new_decomp_schedule(persona, 
                                            main_act_dur, 
                                            truncated_act_dur, 
                                            start_time_hour,
                                            end_time_hour,
                                            inserted_act,
                                            inserted_act_dur)[0]


##############################################################################
# CHAPTER 3: Plan
##############################################################################

########## Potentially add things related to new job/identity here.
  #### E.g., add variable for job_change. if true, add statements/routine to statements and daily req prompt
def revise_identity(persona): 
  p_name = persona.scratch.name

  focal_points = [f"{p_name}'s plan for {persona.scratch.get_str_curr_date_str()}.",
                  f"Important recent events for {p_name}'s life."]
  retrieved = new_retrieve(persona, focal_points)

  statements = "[Statements]\n"
  for key, val in retrieved.items():
    for i in val: 
      statements += f"{i.created.strftime('%A %B %d -- %H:%M %p')}: {i.embedding_key}\n"


  llm_param = {"max_new_tokens": 500, "temperature": 0.85, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }

  # print (";adjhfno;asdjao;idfjo;af", p_name)
  plan_prompt = statements + "\n"
  plan_prompt += f"Given the statements above, is there anything that {p_name} should remember as they plan for"
  plan_prompt += f" *{persona.scratch.curr_time.strftime('%A %B %d')}*? "
  plan_prompt += f"If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement)\n\n"
  plan_prompt += f"Write the response from {p_name}'s perspective."
  plan_note = LLM_single_request(plan_prompt, llm_param)
  # print (plan_note)

  thought_prompt = statements + "\n"
  thought_prompt += f"Given the statements above, how might we summarize {p_name}'s feelings about their days up to now?\n\n"
  thought_prompt += f"Write the response from {p_name}'s perspective."
  thought_note = LLM_single_request(thought_prompt, llm_param)
  # print (thought_note)

  currently_prompt = f"{p_name}'s status from {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}:\n"
  currently_prompt += f"{persona.scratch.currently}\n\n"
  currently_prompt += f"{p_name}'s thoughts at the end of {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}:\n" 
  currently_prompt += (plan_note + thought_note).replace('\n', '') + "\n\n"
  currently_prompt += f"It is now {persona.scratch.curr_time.strftime('%A %B %d')}. Given the above, write {p_name}'s status for {persona.scratch.curr_time.strftime('%A %B %d')} that reflects {p_name}'s thoughts at the end of {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}. Write this in third-person talking about {p_name}."
  currently_prompt += f"If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement).\n\n"
  currently_prompt += "Follow this format below:\nStatus: <new status>"
  # print ("DEBUG ;adjhfno;asdjao;asdfsidfjo;af", p_name)
  # print (currently_prompt)

  llm_param = {"max_new_tokens": 1000, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }

  new_currently = LLM_single_request(currently_prompt, llm_param)
  # print (new_currently)
  # print (new_currently[10:])

  persona.scratch.currently = new_currently

  daily_req_prompt = persona.scratch.get_str_iss() + "\n"
  daily_req_prompt += f"Today is {persona.scratch.curr_time.strftime('%A %B %d')}. Here is {persona.scratch.name}'s plan today in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm).\n\n"
  daily_req_prompt += f"Follow this format (the list should have 4~6 items but no more):\n"
  daily_req_prompt += f"1. wake up and complete the morning routine at <time>, 2. ..."

  llm_param = {"max_new_tokens": 500, "temperature": 0.85, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  
  new_daily_req = LLM_single_request(daily_req_prompt, llm_param)
  new_daily_req = new_daily_req.replace('\n', ' ')
  print ("WE ARE HERE!!!", new_daily_req)
  persona.scratch.daily_plan_req = new_daily_req


def _long_term_planning(persona, new_day): 
  """
  Formulates the persona's daily long-term plan if it is the start of a new 
  day. This basically has two components: first, we create the wake-up hour, 
  and second, we create the hourly schedule based on it. 
  INPUT
    new_day: Indicates whether the current time signals a "First day",
             "New day", or False (for neither). This is important because we
             create the personas' long term planning on the new day. 
  """
  # We start by creating the wake up hour for the persona. 
  wake_up_hour = generate_wake_up_hour(persona)

  # When it is a new day, we start by creating the daily_req of the persona.
  # Note that the daily_req is a list of strings that describe the persona's
  # day in broad strokes.
  if new_day == "First day": 
    # Bootstrapping the daily plan for the start of then generation:
    # if this is the start of generation (so there is no previous day's 
    # daily requirement, or if we are on a new day, we want to create a new
    # set of daily requirements.
    persona.scratch.daily_req = generate_first_daily_plan(persona, 
                                                          wake_up_hour)
  elif new_day == "New day":
    revise_identity(persona)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - TODO
    # We need to create a new daily_req here...
    persona.scratch.daily_req = persona.scratch.daily_req

  # Based on the daily_req, we create an hourly schedule for the persona, 
  # which is a list of todo items with a time duration (in minutes) that 
  # add up to 24 hours.
  persona.scratch.f_daily_schedule = generate_hourly_schedule(persona, 
                                                              wake_up_hour)
  persona.scratch.f_daily_schedule_hourly_org = (persona.scratch
                                                   .f_daily_schedule[:])


  # Added March 4 -- adding plan to the memory.
  thought = f"This is {persona.scratch.name}'s plan for {persona.scratch.curr_time.strftime('%A %B %d')}:"
  for i in persona.scratch.daily_req: 
    thought += f" {i},"
  thought = thought[:-1] + "."
  created = persona.scratch.curr_time
  expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
  s, p, o = (persona.scratch.name, "plan", persona.scratch.curr_time.strftime('%A %B %d'))
  keywords = set(["plan"])
  thought_poignancy = 5
  thought_embedding_pair = (thought, get_embedding(thought))
  persona.a_mem.add_thought(created, expiration, s, p, o, 
                            thought, keywords, thought_poignancy, 
                            thought_embedding_pair, None)

  # print("Sleeping for 20 seconds...")
  # time.sleep(10)
  # print("Done sleeping!")



def _determine_action(persona, maze): 
  """
  Creates the next action sequence for the persona. 
  The main goal of this function is to run "add_new_action" on the persona's 
  scratch space, which sets up all the action related variables for the next 
  action. 
  As a part of this, the persona may need to decompose its hourly schedule as 
  needed.   
  INPUT
    persona: Current <Persona> instance whose action we are determining. 
    maze: Current <Maze> instance. 
  """
  def determine_decomp(act_desp, act_dura):
    """
    Given an action description and its duration, we determine whether we need
    to decompose it. If the action is about the agent sleeping, we generally
    do not want to decompose it, so that's what we catch here. 

    INPUT: 
      act_desp: the description of the action (e.g., "sleeping")
      act_dura: the duration of the action in minutes. 
    OUTPUT: 
      a boolean. True if we need to decompose, False otherwise. 
    """

    keywords = ["sleep", "sleeping", "aslepp"]
    act_desc2 = act_desp.strip() + "."

    if "sleep" not in act_desp and "bed" not in act_desp and "waking up." not in act_desc2: 
      return True
    elif "sleeping" in act_desp or "asleep" in act_desp or "in bed" in act_desp or "waking up." in act_desc2:
      return False
    elif "sleep" in act_desp or "bed" in act_desp: 
      if act_dura > 60: 
        return False
    return True

  # The goal of this function is to get us the action associated with 
  # <curr_index>. As a part of this, we may need to decompose some large 
  # chunk actions. 
  # Importantly, we try to decompose at least two hours worth of schedule at
  # any given point. 
  curr_index = persona.scratch.get_f_daily_schedule_index()
  curr_index_60 = persona.scratch.get_f_daily_schedule_index(advance=60)

  # * Decompose * 
  # During the first hour of the day, we need to decompose two hours 
  # sequence. We do that here. 
  if curr_index == 0:
    # This portion is invoked if it is the first hour of the day. 
    act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index]
    act_sector = generate_action_sector(act_desp, persona, maze)
    decomp_log =  None
 
    if act_dura >= 60: 
      # We decompose if the next action is longer than an hour, and fits the
      # criteria described in determine_decomp.
      if determine_decomp(act_desp, act_dura): 
        decomp_log = (generate_task_decomp(persona, act_desp, act_dura, match_index = curr_index))
        persona.scratch.f_daily_schedule[curr_index:curr_index+1] = decomp_log
        
        ### it makes more sense to log actiona and decomp here:
          ### 1) Save decomp in variable before assigning it to persona.scratch.f_daily_schedule X
          ### 2) classify action + description as hate/agression, and whether it was an interaction or not
            ### 2.1) Try to find a way to determine if it intergroup or within group ==> check some actions manually
            ### 2.2) Potentially, find good way of combining action and decomp based on what was created / if statements
          ### 3) Log action + classification X
            ### 3.1) Either filter for location or add location to log (preferred) X
            ### 3.2) If logging multiple actions, adjust time!
          ### 4) Make sure there is no double logging
          ### 5) Need to handle sleeping/waiting? probably filter those actions out! (in R code) X

      # might need to refine (yes/no/unknown)
      # potentially filter trivial actions like sleeping
      act_hate = reflect_action(persona, "", act_desp, decomp_log) ## adjust decomp to include both action and decomp
      group_status = reflect_groups(persona, act_desp, decomp_log)
      if decomp_log:
        try:
          full_action_str = f"{act_desp}:\n"
          for action, dur in decomp_log:
              full_action_str += f"- {action}\n"
        except:
          full_action_str = act_desp
      else:
        full_action_str = act_desp

      if group_status["interaction"]:
        if group_status["intergroup"]==True:
          if persona.scratch.group_identity == "Group A":
            target_group = "Group B"
          else:
            target_group = "Group A"
        elif group_status["intergroup"]==False:
          target_group = persona.scratch.group_identity
        else:
          target_group = "unknown"
      else:
        target_group = None

      logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "Action", "Decision": full_action_str, "Init": persona, "Target": None, 
                      "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": persona.scratch.group_condition, 
                      "Init Group": persona.scratch.group_identity, "Target Group": target_group, "convo_hate": "", "interview_hate": "", "act_hate": act_hate, "act_location": act_sector, }
      TRACKER.log_decision(logging_info, persona.scratch.sim_nr)

      robustness_case = "action" # add different cases to check: Convo, hiring, acting
      robustness_context = []
      robustness_items = run_gpt_robustness_check(persona, robustness_context, robustness_case) ## add code to construct context, run robustness checks and return

      # then log the outcome
      logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "robustness_action", "Decision": robustness_items, "Init": persona, "Target": None, 
                    "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": persona.scratch.group_condition, 
                    "Init Group": persona.scratch.group_identity, "Target Group": target_group, "convo_hate": "", "interview_hate": ""}
      TRACKER.log_decision(logging_info, persona.scratch.sim_nr)


    if curr_index_60 < len(persona.scratch.f_daily_schedule) and curr_index_60!=curr_index:
      act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index_60]
      act_sector = generate_action_sector(act_desp, persona, maze)
      if act_dura >= 60: 
        if determine_decomp(act_desp, act_dura): 
          decomp_log = (generate_task_decomp(persona, act_desp, act_dura, match_index=curr_index_60))
          persona.scratch.f_daily_schedule[curr_index_60:curr_index_60+1] = decomp_log

          act_hate = reflect_action(persona, "", act_desp, decomp_log) ## adjust decomp to include both action and decomp
          group_status = reflect_groups(persona, act_desp, decomp_log)
          if decomp_log:
            try:
              full_action_str = f"{act_desp}:\n"
              for action, dur in decomp_log:
                  full_action_str += f"- {action}\n"
            except:
              full_action_str = act_desp
          else:
            full_action_str = act_desp

          if group_status["interaction"]:
            if group_status["intergroup"]==True:
              if persona.scratch.group_identity == "Group A":
                target_group = "Group B"
              else:
                target_group = "Group A"
            elif group_status["intergroup"]==False:
              target_group = persona.scratch.group_identity
            else:
              target_group = "unknown"
          else:
            target_group = None
          logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "Action", "Decision": full_action_str, "Init": persona, "Target":  None, 
                          "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": persona.scratch.group_condition, 
                          "Init Group": persona.scratch.group_identity, "Target Group": target_group, "convo_hate": "", "interview_hate": "", "act_hate": act_hate, "act_location": act_sector, }
          TRACKER.log_decision(logging_info, persona.scratch.sim_nr)

          robustness_case = "action" # add different cases to check: Convo, hiring, acting
          robustness_context = []
          robustness_items = run_gpt_robustness_check(persona, robustness_context, robustness_case) ## add code to construct context, run robustness checks and return

          # then log the outcome
          logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "robustness_action", "Decision": robustness_items, "Init": persona, "Target": None, 
                        "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": persona.scratch.group_condition, 
                        "Init Group": persona.scratch.group_identity, "Target Group": target_group, "convo_hate": "", "interview_hate": ""}
          TRACKER.log_decision(logging_info, persona.scratch.sim_nr)
          
    if curr_index_60 + 1 < len(persona.scratch.f_daily_schedule):
      act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index_60+1]
      act_sector = generate_action_sector(act_desp, persona, maze)
      if act_dura >= 60: 
        if determine_decomp(act_desp, act_dura): 
          decomp_log = (generate_task_decomp(persona, act_desp, act_dura, match_index=curr_index_60+1))
          persona.scratch.f_daily_schedule[curr_index_60+1:curr_index_60+2] = decomp_log
          act_hate = reflect_action(persona, "", act_desp, decomp_log) ## adjust decomp to include both action and decomp
          group_status = reflect_groups(persona, act_desp, decomp_log)
          if decomp_log:
            try:
              full_action_str = f"{act_desp}:\n"
              for action, dur in decomp_log:
                  full_action_str += f"- {action}\n"
            except:
              full_action_str = act_desp
          else:
            full_action_str = act_desp
          if group_status["interaction"]:
            if group_status["intergroup"]==True:
              if persona.scratch.group_identity == "Group A":
                target_group = "Group B"
              else:
                target_group = "Group A"
            elif group_status["intergroup"]==False:
              target_group = persona.scratch.group_identity
            else:
              target_group = "unknown"
          else:
            target_group = None
          logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "Action", "Decision": full_action_str, "Init": persona, "Target":  None, 
                          "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": persona.scratch.group_condition, 
                          "Init Group": persona.scratch.group_identity, "Target Group": target_group, "convo_hate": "", "interview_hate": "", "act_hate": act_hate, "act_location": act_sector, }
          TRACKER.log_decision(logging_info, persona.scratch.sim_nr)

          robustness_case = "action" # add different cases to check: Convo, hiring, acting
          robustness_context = []
          robustness_items = run_gpt_robustness_check(persona, robustness_context, robustness_case) ## add code to construct context, run robustness checks and return

          # then log the outcome
          logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "robustness_action", "Decision": robustness_items, "Init": persona, "Target": None, 
                        "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": persona.scratch.group_condition, 
                        "Init Group": persona.scratch.group_identity, "Target Group": target_group, "convo_hate": "", "interview_hate": ""}
          TRACKER.log_decision(logging_info, persona.scratch.sim_nr)
  else:
    act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index]
    act_sector = generate_action_sector(act_desp, persona, maze)
    decomp_log =  None
    if act_dura >= 60: 
      # We decompose if the next action is longer than an hour, and fits the
      # criteria described in determine_decomp.
      if determine_decomp(act_desp, act_dura): 
        decomp_log = (generate_task_decomp(persona, act_desp, act_dura, match_index = curr_index))
        persona.scratch.f_daily_schedule[curr_index:curr_index+1] = decomp_log

      act_hate = reflect_action(persona, "", act_desp, decomp_log) ## adjust decomp to include both action and decomp
      group_status = reflect_groups(persona, act_desp, decomp_log)
      if decomp_log:
        try:
          full_action_str = f"{act_desp}:\n"
          for action, dur in decomp_log:
              full_action_str += f"- {action}\n"
        except:
          full_action_str = act_desp
      else:
        full_action_str = act_desp
        
      if group_status["interaction"]:
        if group_status["intergroup"]==True:
          if persona.scratch.group_identity == "Group A":
            target_group = "Group B"
          else:
            target_group = "Group A"
        elif group_status["intergroup"]==False:
          target_group = persona.scratch.group_identity
        else:
          target_group = "unknown"
      else:
        target_group = None
      logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "Action", "Decision": full_action_str, "Init": persona, "Target":  None, 
                      "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": persona.scratch.group_condition, 
                      "Init Group": persona.scratch.group_identity, "Target Group": target_group, "convo_hate": "", "interview_hate": "", "act_hate": act_hate, "act_location": act_sector, }
      TRACKER.log_decision(logging_info, persona.scratch.sim_nr)

      robustness_case = "action" # add different cases to check: Convo, hiring, acting
      robustness_context = []
      robustness_items = run_gpt_robustness_check(persona, robustness_context, robustness_case) ## add code to construct context, run robustness checks and return

      # then log the outcome
      logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "robustness_action", "Decision": robustness_items, "Init": persona, "Target": None, 
                    "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": persona.scratch.group_condition, 
                    "Init Group": persona.scratch.group_identity, "Target Group": target_group, "convo_hate": "", "interview_hate": ""}
      TRACKER.log_decision(logging_info, persona.scratch.sim_nr)

    if curr_index_60 < len(persona.scratch.f_daily_schedule) and curr_index_60!=curr_index:
      # If it is not the first hour of the day, this is always invoked (it is
      # also invoked during the first hour of the day -- to double up so we can
      # decompose two hours in one go). Of course, we need to have something to
      # decompose as well, so we check for that too. 
      if persona.scratch.curr_time.hour < 23:
        # And we don't want to decompose after 11 pm. 
        act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index_60]
        act_sector = generate_action_sector(act_desp, persona, maze)
        if act_dura >= 60: 
          if determine_decomp(act_desp, act_dura): 
            decomp_log = (generate_task_decomp(persona, act_desp, act_dura, match_index=curr_index_60))
            persona.scratch.f_daily_schedule[curr_index_60:curr_index_60+1] = decomp_log
            act_hate = reflect_action(persona, "", act_desp, decomp_log) ## adjust decomp to include both action and decomp
            group_status = reflect_groups(persona, act_desp, decomp_log)
            if decomp_log:
              try:
                full_action_str = f"{act_desp}:\n"
                for action, dur in decomp_log:
                    full_action_str += f"- {action}\n"
              except:
                full_action_str = act_desp
            else:
              full_action_str = act_desp

            if group_status["interaction"]:
              if group_status["intergroup"]==True:
                if persona.scratch.group_identity == "Group A":
                  target_group = "Group B"
                else:
                  target_group = "Group A"
              elif group_status["intergroup"]==False:
                target_group = persona.scratch.group_identity
              else:
                target_group = "unknown"
            else:
              target_group = None
            logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "Action", "Decision": full_action_str, "Init": persona, "Target":  None, 
                            "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": persona.scratch.group_condition, 
                            "Init Group": persona.scratch.group_identity, "Target Group": target_group, "convo_hate": "", "interview_hate": "", "act_hate": act_hate, "act_location": act_sector, }
            TRACKER.log_decision(logging_info, persona.scratch.sim_nr)

            robustness_case = "action" # add different cases to check: Convo, hiring, acting
            robustness_context = []
            robustness_items = run_gpt_robustness_check(persona, robustness_context, robustness_case) ## add code to construct context, run robustness checks and return

            # then log the outcome
            logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "robustness_action", "Decision": robustness_items, "Init": persona, "Target": None, 
                          "Sim_Time": persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": persona.scratch.group_condition, 
                          "Init Group": persona.scratch.group_identity, "Target Group": target_group, "convo_hate": "", "interview_hate": ""}
            TRACKER.log_decision(logging_info, persona.scratch.sim_nr)
  # * End of Decompose * 

  # Generate an <Action> instance from the action description and duration. By
  # this point, we assume that all the relevant actions are decomposed and 
  # ready in f_daily_schedule. 0
  print (curr_index)
  print (len(persona.scratch.f_daily_schedule))
  print (persona.scratch.name)
  print ("------")

  # 1440
  x_emergency = 0
  for i in persona.scratch.f_daily_schedule: 
    x_emergency += i[1]
  # print ("x_emergency", x_emergency)

  if 1440 - x_emergency > 0: 
    print ("x_emergency__AAA", x_emergency)
  persona.scratch.f_daily_schedule += [["sleeping", 1440 - x_emergency]]


  act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index] 
  print(act_desp)
  print("ACTION DEBUG END")


  # Finding the target location of the action and creating action-related
  # variables.
  act_world = maze.access_tile(persona.scratch.curr_tile)["world"]
  # act_sector = maze.access_tile(persona.scratch.curr_tile)["sector"]
  act_sector = generate_action_sector(act_desp, persona, maze)
  act_arena = generate_action_arena(act_desp, persona, maze, act_world, act_sector)
  act_address = f"{act_world}:{act_sector}:{act_arena}"
  act_game_object = generate_action_game_object(act_desp, act_address,
                                                persona, maze)
  new_address = f"{act_world}:{act_sector}:{act_arena}:{act_game_object}"
  act_pron = generate_action_pronunciatio(act_desp, persona)
  act_event = generate_action_event_triple(act_desp, persona)
  # Persona's actions also influence the object states. We set those up here. 
  act_obj_desp = generate_act_obj_desc(act_game_object, act_desp, persona)
  act_obj_pron = generate_action_pronunciatio(act_obj_desp, persona)
  act_obj_event = generate_act_obj_event_triple(act_game_object, 
                                                act_obj_desp, persona)
  
  # The first task (e.g., sleeping) should start at midnight!
  ## Therefore deduct passed time from act_dura if curr_index=0
  if curr_index==0:
    act_dura -= persona.scratch.curr_time.hour * 60

  # Adding the action to persona's queue. 
  persona.scratch.add_new_action(new_address, 
                                 int(act_dura), 
                                 act_desp, 
                                 act_pron, 
                                 act_event,
                                 None,
                                 None,
                                 None,
                                 None,
                                 act_obj_desp, 
                                 act_obj_pron, 
                                 act_obj_event)


def _choose_retrieved(persona, retrieved): 
  """
  Retrieved elements have multiple core "curr_events". We need to choose one
  event to which we are going to react to. We pick that event here. 
  INPUT
    persona: Current <Persona> instance whose action we are determining. 
    retrieved: A dictionary of <ConceptNode> that were retrieved from the 
               the persona's associative memory. This dictionary takes the
               following form: 
               dictionary[event.description] = 
                 {["curr_event"] = <ConceptNode>, 
                  ["events"] = [<ConceptNode>, ...], 
                  ["thoughts"] = [<ConceptNode>, ...] }
  """
  # Once we are done with the reflection, we might want to build a more  
  # complex structure here.
  
  # We do not want to take self events... for now 
  copy_retrieved = retrieved.copy()
  for event_desc, rel_ctx in copy_retrieved.items(): 
    curr_event = rel_ctx["curr_event"]
    if curr_event.subject == persona.name: 
      del retrieved[event_desc]

  # Always choose persona first.
  priority = []
  for event_desc, rel_ctx in retrieved.items(): 
    curr_event = rel_ctx["curr_event"]
    if (":" not in curr_event.subject 
        and curr_event.subject != persona.name): 
      priority += [rel_ctx]
  if priority: 
    return random.choice(priority)

  # Skip idle. 
  for event_desc, rel_ctx in retrieved.items(): 
    curr_event = rel_ctx["curr_event"]
    if "is idle" not in event_desc: 
      priority += [rel_ctx]
  if priority: 
    return random.choice(priority)
  return None


def _should_react(persona, retrieved, personas): 
  """
  Determines what form of reaction the persona should exihibit given the 
  retrieved values. 
  INPUT
    persona: Current <Persona> instance whose action we are determining. 
    retrieved: A dictionary of <ConceptNode> that were retrieved from the 
               the persona's associative memory. This dictionary takes the
               following form: 
               dictionary[event.description] = 
                 {["curr_event"] = <ConceptNode>, 
                  ["events"] = [<ConceptNode>, ...], 
                  ["thoughts"] = [<ConceptNode>, ...] }
    personas: A dictionary that contains all persona names as keys, and the 
              <Persona> instance as values. 
  """
  def lets_talk(init_persona, target_persona, retrieved):
    if (not target_persona.scratch.act_address 
        or not target_persona.scratch.act_description
        or not init_persona.scratch.act_address
        or not init_persona.scratch.act_description): 
      return False

    if ("sleeping" in target_persona.scratch.act_description 
        or "sleeping" in init_persona.scratch.act_description): 
      return False

    if init_persona.scratch.curr_time.hour == 23: 
      return False

    if "<waiting>" in target_persona.scratch.act_address: 
      return False

    if (target_persona.scratch.chatting_with 
      or init_persona.scratch.chatting_with): 
      return False

    if (target_persona.name in init_persona.scratch.chatting_with_buffer): 
      if init_persona.scratch.chatting_with_buffer[target_persona.name] > 0: 
        return False

    if generate_decide_to_talk(init_persona, target_persona, retrieved): 

      return True

    return False

  def lets_react(init_persona, target_persona, retrieved): 
    if (not target_persona.scratch.act_address 
        or not target_persona.scratch.act_description
        or not init_persona.scratch.act_address
        or not init_persona.scratch.act_description): 
      return False

    if ("sleeping" in target_persona.scratch.act_description 
        or "sleeping" in init_persona.scratch.act_description): 
      return False

    # return False
    if init_persona.scratch.curr_time.hour == 23: 
      return False

    if "waiting" in target_persona.scratch.act_description: 
      return False
    if init_persona.scratch.planned_path == []:
      return False

    if (init_persona.scratch.act_address 
        != target_persona.scratch.act_address): 
      return False

    ## or log avoid/approach here:
    react_mode = generate_decide_to_react(init_persona, 
                                          target_persona, retrieved)

    if react_mode == "1": 
      wait_until = ((target_persona.scratch.act_start_time 
        + datetime.timedelta(minutes=target_persona.scratch.act_duration - 1))
        .strftime("%B %d, %Y, %H:%M:%S"))
      return f"wait: {wait_until}"
    elif react_mode == "2":
      return False
      return "do other things"
    else:
      return False #"keep" 

  # If the persona is chatting right now, default to no reaction 
  if persona.scratch.chatting_with: 
    return False
  if "<waiting>" in persona.scratch.act_address: 
    return False

  # Recall that retrieved takes the following form: 
  # dictionary {["curr_event"] = <ConceptNode>, 
  #             ["events"] = [<ConceptNode>, ...], 
  #             ["thoughts"] = [<ConceptNode>, ...]}
  curr_event = retrieved["curr_event"]

  ############# If explicit interaction structure, add interactions here! #################
    ## When determining talking/waiting, also ask about wanting to interact
    ## Create an interaction function
      ## Given information used to decide tallking/waiting 
        # -> decide whether agent wants to do something
        # -> specify action
        # I.e., does the agent want to do something to or with the other person
    ## if talking, do interaction after talking
      ### I.e, have 3 modes: Talking, waiting, interacting (if talking, ask about interacting afterwards/user chat to decide?)
      ### Potentially branch out: If talk -> after talk ask for interaction, if not talk -> ask for interaction, if wait -> ask for interaction (or just wait)


  if ":" not in curr_event.subject: 
    # this is a persona event. 
    if lets_talk(persona, personas[curr_event.subject], retrieved):
      return f"chat with {curr_event.subject}" 
    react_mode = lets_react(persona, personas[curr_event.subject], 
                            retrieved)
    return react_mode
  return False


def _create_react(persona, inserted_act, inserted_act_dur,
                  act_address, act_event, chatting_with, chat, chatting_with_buffer,
                  chatting_end_time, 
                  act_pronunciatio, act_obj_description, act_obj_pronunciatio, 
                  act_obj_event, act_start_time=None): 
  p = persona 

  min_sum = 0
  for i in range (p.scratch.get_f_daily_schedule_hourly_org_index()): 
    min_sum += p.scratch.f_daily_schedule_hourly_org[i][1]
  start_hour = int (min_sum/60)

  if (p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1] >= 120):
    end_hour = start_hour + p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1]/60

  elif (p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1] + 
      p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()+1][1]): 
    end_hour = start_hour + ((p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1] + 
              p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()+1][1])/60)

  else: 
    end_hour = start_hour + 2
  end_hour = int(end_hour)

  dur_sum = 0
  count = 0 
  start_index = None
  end_index = None
  for act, dur in p.scratch.f_daily_schedule: 
    if dur_sum >= start_hour * 60 and start_index == None:
      start_index = count
    if dur_sum >= end_hour * 60 and end_index == None: 
      end_index = count
    dur_sum += dur
    count += 1

  ret = generate_new_decomp_schedule(p, inserted_act, inserted_act_dur, 
                                       start_hour, end_hour)
  p.scratch.f_daily_schedule[start_index:end_index] = ret
  p.scratch.add_new_action(act_address,
                           inserted_act_dur,
                           inserted_act,
                           act_pronunciatio,
                           act_event,
                           chatting_with,
                           chat,
                           chatting_with_buffer,
                           chatting_end_time,
                           act_obj_description,
                           act_obj_pronunciatio,
                           act_obj_event,
                           act_start_time)


def _chat_react(maze, persona, focused_event, reaction_mode, personas):
  # There are two personas -- the persona who is initiating the conversation
  # and the persona who is the target. We get the persona instances here. 
  init_persona = persona
  target_persona = personas[reaction_mode[9:].strip()]
  curr_personas = [init_persona, target_persona]

  # Actually creating the conversation here. 
  convo, duration_min = generate_convo(maze, init_persona, target_persona)
  convo_summary = generate_convo_summary(init_persona, convo)
  inserted_act = convo_summary
  inserted_act_dur = duration_min

  act_start_time = target_persona.scratch.act_start_time


  convo_transcript = ""
  for row in convo: 
    convo_transcript += f'{row[0]}: "{row[1]}"\n'

  curr_time = target_persona.scratch.curr_time
  if curr_time.second != 0: 
    temp_curr_time = curr_time + datetime.timedelta(seconds=60 - curr_time.second)
    chatting_end_time = temp_curr_time + datetime.timedelta(minutes=inserted_act_dur)
  else: 
    chatting_end_time = curr_time + datetime.timedelta(minutes=inserted_act_dur)

  for role, p in [("init", init_persona), ("target", target_persona)]: 
    if role == "init": 
      act_address = f"<persona> {target_persona.name}"
      act_event = (p.name, "chat with", target_persona.name)
      chatting_with = target_persona.name
      chatting_with_buffer = {}
      chatting_with_buffer[target_persona.name] = 800
    elif role == "target": 
      act_address = f"<persona> {init_persona.name}"
      act_event = (p.name, "chat with", init_persona.name)
      chatting_with = init_persona.name
      chatting_with_buffer = {}
      chatting_with_buffer[init_persona.name] = 800

    act_pronunciatio = "💬" 
    act_obj_description = None
    act_obj_pronunciatio = None
    act_obj_event = (None, None, None)

    _create_react(p, inserted_act, inserted_act_dur,
      act_address, act_event, chatting_with, convo, chatting_with_buffer, chatting_end_time,
      act_pronunciatio, act_obj_description, act_obj_pronunciatio, 
      act_obj_event, act_start_time)


def _wait_react(persona, reaction_mode): 
  p = persona

  inserted_act = f'waiting to start {p.scratch.act_description.split("(")[-1][:-1]}'
  end_time = datetime.datetime.strptime(reaction_mode[6:].strip(), "%B %d, %Y, %H:%M:%S")
  inserted_act_dur = (end_time.minute + end_time.hour * 60) - (p.scratch.curr_time.minute + p.scratch.curr_time.hour * 60) + 1

  act_address = f"<waiting> {p.scratch.curr_tile[0]} {p.scratch.curr_tile[1]}"
  act_event = (p.name, "waiting to start", p.scratch.act_description.split("(")[-1][:-1])
  chatting_with = None
  chat = None
  chatting_with_buffer = None
  chatting_end_time = None

  act_pronunciatio = "⌛" 
  act_obj_description = None
  act_obj_pronunciatio = None
  act_obj_event = (None, None, None)

  _create_react(p, inserted_act, inserted_act_dur,
    act_address, act_event, chatting_with, chat, chatting_with_buffer, chatting_end_time,
    act_pronunciatio, act_obj_description, act_obj_pronunciatio, act_obj_event)


def plan(persona, maze, personas, new_day, retrieved): 
  """
  Main cognitive function of the chain. It takes the retrieved memory and 
  perception, as well as the maze and the first day state to conduct both 
  the long term and short term planning for the persona. 

  INPUT: 
    maze: Current <Maze> instance of the world. 
    personas: A dictionary that contains all persona names as keys, and the 
              Persona instance as values. 
    new_day: This can take one of the three values. 
      1) <Boolean> False -- It is not a "new day" cycle (if it is, we would
         need to call the long term planning sequence for the persona). 
      2) <String> "First day" -- It is literally the start of a simulation,
         so not only is it a new day, but also it is the first day. 
      2) <String> "New day" -- It is a new day. 
    retrieved: dictionary of dictionary. The first layer specifies an event,
               while the latter layer specifies the "curr_event", "events", 
               and "thoughts" that are relevant.
  OUTPUT 
    The target action address of the persona (persona.scratch.act_address).
  """ 
  # PART 1: Generate the hourly schedule. 
  if new_day: 
    _long_term_planning(persona, new_day) ## potentially add group identity stuff here (threats)

  # PART 2: If the current action has expired, we want to create a new plan.
  if persona.scratch.act_check_finished(): 
    _determine_action(persona, maze)

  # PART 3: If you perceived an event that needs to be responded to (saw 
  # another persona), and retrieved relevant information. 
  # Step 1: Retrieved may have multiple events represented in it. The first 
  #         job here is to determine which of the events we want to focus 
  #         on for the persona. 
  #         <focused_event> takes the form of a dictionary like this: 
  #         dictionary {["curr_event"] = <ConceptNode>, 
  #                     ["events"] = [<ConceptNode>, ...], 
  #                     ["thoughts"] = [<ConceptNode>, ...]}
  focused_event = False
  if retrieved.keys(): 
    focused_event = _choose_retrieved(persona, retrieved)
  
  # Step 2: Once we choose an event, we need to determine whether the
  #         persona will take any actions for the perceived event. There are
  #         three possible modes of reaction returned by _should_react. 
  #         a) "chat with {target_persona.name}"
  #         b) "react"
  #         c) False
  if focused_event: 
    reaction_mode = _should_react(persona, focused_event, personas)
    curr_event = focused_event["curr_event"]
    if reaction_mode: 
      # If we do want to chat, then we generate conversation 
      if reaction_mode[:9] == "chat with":
        reaction_type = "chat"
        _chat_react(maze, persona, focused_event, reaction_mode, personas)
      elif reaction_mode[:4] == "wait": 
        _wait_react(persona, reaction_mode)
        reaction_type = "wait"

      ### Add logging here for approach/avoid
      
      ############### If this does not work #########################
        ### add explicit action prompt when meeting other agents/react (add option interaction)
        ### I.e., explicit structure: 
            # Do you want to interact with this person? -> what do you want to do? -> classify -> add to schedule/decomp (create_react)
            # Based on reaction mode? If talk, add interaction after talk with content generated after the convo (in conversation.py?)
            # If not talk, ask for interaction here and add it with create_react?
      
      ### log approaching
      init_persona = persona
      target_persona = personas[curr_event.subject]

      ## Update to something like approach of "interacted"
        ## Potentially split approach to talk/wait?
      logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "Approach", "Decision": reaction_type, "Init": init_persona, 
                      "Target": target_persona, "Sim_Time": init_persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", 
                      "target_rating": "", "target_rating_soc": "", "Group Condition": init_persona.scratch.group_condition, 
                      "Init Group": init_persona.scratch.group_identity, "Target Group": target_persona.scratch.group_identity, 
                "convo_hate": "", "interview_hate": ""}
      TRACKER.log_decision(logging_info, init_persona.scratch.sim_nr)

      # elif reaction_mode == "do other things": 
      #   _chat_react(persona, focused_event, reaction_mode, personas)
    else:
      if curr_event.subject in personas: # if focused object is another person
        init_persona = persona
        target_persona = personas[curr_event.subject]
        ## Update to something like avoid of "interacted"
          ## Potentially split avoid to talk/wait?
        logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "Approach", "Decision": "avoid", "Init": init_persona, 
                        "Target": target_persona, "Sim_Time": init_persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", 
                        "target_rating": "", "target_rating_soc": "", "Group Condition": init_persona.scratch.group_condition,
                        "Init Group": init_persona.scratch.group_identity, "Target Group": target_persona.scratch.group_identity, 
                  "convo_hate": "", "interview_hate": ""}
        TRACKER.log_decision(logging_info, init_persona.scratch.sim_nr)
        ############ End log ############

  # Step 3: Chat-related state clean up. 
  # If the persona is not chatting with anyone, we clean up any of the 
  # chat-related states here. 
  if persona.scratch.act_event[1] != "chat with":
    persona.scratch.chatting_with = None
    persona.scratch.chat = None
    persona.scratch.chatting_end_time = None
  # We want to make sure that the persona does not keep conversing with each
  # other in an infinite loop. So, chatting_with_buffer maintains a form of 
  # buffer that makes the persona wait from talking to the same target 
  # immediately after chatting once. We keep track of the buffer value here. 
  curr_persona_chat_buffer = persona.scratch.chatting_with_buffer
  for persona_name, buffer_count in curr_persona_chat_buffer.items():
    if persona_name != persona.scratch.chatting_with: 
      persona.scratch.chatting_with_buffer[persona_name] -= 1

  return persona.scratch.act_address






































 
