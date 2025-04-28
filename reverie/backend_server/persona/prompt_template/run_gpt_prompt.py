"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: run_gpt_prompt.py
Description: Defines all run gpt prompt functions. These functions directly
interface with the safe_generate_response function.
"""
import re
import datetime
import sys
import ast
import random
random.seed(0)

sys.path.append('../../')

from global_methods import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.print_prompt import *

def get_random_alphanumeric(i=6, j=6): 
  """
  Returns a random alpha numeric strength that has the length of somewhere
  between i and j. 

  INPUT: 
    i: min_range for the length
    j: max_range for the length
  OUTPUT: 
    an alpha numeric str with the length of somewhere between i and j.
  """
  k = random.randint(i, j)
  x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
  return x


##############################################################################
# CHAPTER 1: Run GPT Prompt
##############################################################################

def run_gpt_prompt_wake_up_hour(persona, test_input=None, verbose=False): 
  """
  Given the persona, returns an integer that indicates the hour when the 
  persona wakes up.  

  INPUT: 
    persona: The Persona class instance 
  OUTPUT: 
    integer for the wake up hour.
  """
  def create_prompt_input(persona, test_input=None): 
    if test_input: return test_input
    prompt_input = [persona.scratch.get_str_iss(),
                    persona.scratch.get_str_lifestyle(),
                    persona.scratch.get_str_firstname()]
    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    cr = int(gpt_response.strip().lower().split("am")[0])
    return cr
  
  def __func_validate(gpt_response, prompt=""): 
    try: __func_clean_up(gpt_response, prompt="")
    except: return False
    return True

  def get_fail_safe(): 
    fs = 8
    return fs

  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 5, 
  #            "temperature": 0.8, "top_p": 1, "stream": False,
  #            "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": ["\n"]}

  llm_param = {"max_new_tokens": 5, "temperature": 0.8, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": ["\n"],
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
    
  prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
  prompt_input = create_prompt_input(persona, test_input)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe()

  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
  
  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
    
  return output, [output, prompt, llm_param, prompt_input, fail_safe]


def run_gpt_prompt_daily_plan(persona, 
                              wake_up_hour, 
                              test_input=None, 
                              verbose=False):
  """
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form: 
  'wake up and complete morning routine at 6:00 am', 
  'eat breakfast at 7:00 am',.. 
  Note that the actions come without a period. 

  INPUT: 
    persona: The Persona class instance 
  OUTPUT: 
    a list of daily actions in broad strokes.
  """
  def create_prompt_input(persona, wake_up_hour, test_input=None):
    if test_input: return test_input
    prompt_input = []
    prompt_input += [persona.scratch.get_str_iss()]
    prompt_input += [persona.scratch.get_str_lifestyle()]
    prompt_input += [persona.scratch.get_str_curr_date_str()]
    prompt_input += [persona.scratch.get_str_firstname()]
    prompt_input += [f"{str(wake_up_hour)}:00 am"]
    return prompt_input

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

  llm_param = {"max_new_tokens": 500, "temperature": 0.75, "top_p": 1, "min_p": 0.1, "top_k": 35, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1.25, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
    
  prompt_template = "persona/prompt_template/v2/daily_planning_v6.txt"
  prompt_input = create_prompt_input(persona, wake_up_hour, test_input)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe()

  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
  output_filtered = output[1:] if (f"{wake_up_hour}:00" in output[0].lower() or f"{wake_up_hour}am" in output[0].lower()) else output

  # print("DEBUG BROAD SCHEDULE")
  # print("output: ", output)

  output = ([f"wake up and complete morning routine at {wake_up_hour}:00 am"]
              + output_filtered)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
    
  return output, [output, prompt, llm_param, prompt_input, fail_safe]

### There is an issue here about different sleep times. 
### If sleeping after midnight, the prior activity list should start after that time 
### (otherwise the time between midnight and sleep time will be erroneously "sleeping")

## Also adjust template so that the same action isnt repeated for 5+ hours (e.g., getting ready for hours, exact same general action during work, etc)
def run_gpt_prompt_generate_hourly_schedule(persona, 
                                            curr_hour_str,
                                            p_f_ds_hourly_org, 
                                            hour_str,
                                            intermission2=None,
                                            test_input=None, 
                                            verbose=False): 
  def create_prompt_input(persona, 
                          curr_hour_str, 
                          p_f_ds_hourly_org,
                          hour_str,
                          intermission2=None,
                          test_input=None): 
    if test_input: return test_input
    schedule_format = ""
    for i in hour_str: 
      schedule_format += f"[{persona.scratch.get_str_curr_date_str()} -- {i}]"
      schedule_format += f" Activity: {persona.scratch.first_name} is <DOING ACTIVITY>\n"
    schedule_format = schedule_format[:-1]

    intermission_str = f""
    for count, i in enumerate(persona.scratch.daily_req): 
      intermission_str += f"{str(count+1)}) {i}, "
    intermission_str = intermission_str[:-2]

    prior_schedule = ""
    if p_f_ds_hourly_org: 
      prior_schedule = "\n"
      for count, i in enumerate(p_f_ds_hourly_org): 
        prior_schedule += f"[(ID:{get_random_alphanumeric()})" 
        prior_schedule += f" {persona.scratch.get_str_curr_date_str()} --"
        prior_schedule += f" {hour_str[count]}] Activity:"
        prior_schedule += f" {persona.scratch.get_str_firstname()}"
        prior_schedule += f" is {i}\n"
    print("prior_schedule", prior_schedule)
    # prompt_ending = f" What activity is {persona.scratch.get_str_firstname()} doing at {persona.scratch.get_str_curr_date_str()} at {curr_hour_str}? {persona.scratch.get_str_firstname()} is [complete the activity]" 
    
    prompt_ending = f"[(ID:{get_random_alphanumeric()})"
    prompt_ending += f" {persona.scratch.get_str_curr_date_str()}"
    prompt_ending += f" -- {curr_hour_str}] Activity:"
    prompt_ending += f" {persona.scratch.get_str_firstname()} is <DOING ACTIVITY>"

    if intermission2: 
      intermission2 = f"\n{intermission2}"

    prompt_input = []
    prompt_input += [schedule_format]
    prompt_input += [persona.scratch.get_str_iss()]

    prompt_input += [prior_schedule + "\n"]
    prompt_input += [intermission_str]
    if intermission2: 
      prompt_input += [intermission2]
    else: 
      prompt_input += [""]
    prompt_input += [prompt_ending]
    prompt_input += [persona.scratch.get_str_firstname()]

    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    cr = gpt_response.strip()
    if cr[-1] == ".":
      cr = cr[:-1]
    return cr

  def __func_validate(gpt_response, prompt=""): 
    try: __func_clean_up(gpt_response, prompt="")
    except: 
        print("Failed LLM response!!")
        print(gpt_response)
        return False
    return True

  def get_fail_safe(): 
    fs = "asleep"
    return fs

  print("INTERMISSION: ", intermission2)

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 50, 
  #              "temperature": 0.5, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": ["\n"]}

  llm_param = {"max_new_tokens": 50, "temperature": 0.75, "top_p": 1, "min_p": 0.1, "top_k": 35, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1.25, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": ["\n"],
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }

  prompt_template = "persona/prompt_template/v2/generate_hourly_schedule_v2.txt"
  prompt_input = create_prompt_input(persona, 
                                     curr_hour_str, 
                                     p_f_ds_hourly_org,
                                     hour_str, 
                                     intermission2,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe()
  
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
  
  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
    
  return output, [output, prompt, llm_param, prompt_input, fail_safe]




def run_gpt_prompt_task_decomp(persona, 
                               task, 
                               duration,
                               match_index, 
                               test_input=None, 
                               verbose=False): 
    
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
    
  def fix_broken_lines(text):
    # Replace newline characters that are NOT followed by a number and a ')'
    fixed_text = re.sub(r'\n(?!\s*\d+\))', ' ', text)
    # Optionally, collapse multiple spaces into one
    fixed_text = re.sub(r' {2,}', ' ', fixed_text)
    return fixed_text
        
  def create_prompt_input(persona, task, duration, match_index, test_input=None):

    """
    Today is Saturday June 25. From 00:00 ~ 06:00am, Maeve is 
    planning on sleeping, 06:00 ~ 07:00am, Maeve is 
    planning on waking up and doing her morning routine, 
    and from 07:00am ~08:00am, Maeve is planning on having breakfast.  
    """

    # print("############# DURATION S ################")
    # print(duration)
    # print("############# DURATION E ################") 
      
    curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
    all_indices = []
    # if curr_f_org_index > 0: 
    #   all_indices += [curr_f_org_index-1]
    all_indices += [curr_f_org_index]
    if curr_f_org_index+1 <= len(persona.scratch.f_daily_schedule_hourly_org): 
      all_indices += [curr_f_org_index+1]
    if curr_f_org_index+2 <= len(persona.scratch.f_daily_schedule_hourly_org): 
      all_indices += [curr_f_org_index+2]

    curr_time_range = ""

    print ("DEBUG")
    print (persona.scratch.f_daily_schedule_hourly_org)

    summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. '
    summ_str += f'From '
    for index in all_indices: 
      print ("index", index)
      if index < len(persona.scratch.f_daily_schedule_hourly_org): 
        start_min = 0
        for i in range(index): 
          start_min += persona.scratch.f_daily_schedule_hourly_org[i][1]
        end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
        start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") 
                      + datetime.timedelta(minutes=start_min)) 
        end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") 
                      + datetime.timedelta(minutes=end_min)) 
        start_time_str = start_time.strftime("%H:%M%p")
        end_time_str = end_time.strftime("%H:%M%p")
        summ_str += f"{start_time_str} ~ {end_time_str}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
        print("Index: ", index)
        print("Match Index: ", match_index)
        if index == match_index:
          curr_time_range = f'{start_time_str} ~ {end_time_str}'
    summ_str = summ_str[:-2] + "."

    prompt_input = []
    prompt_input += [persona.scratch.get_str_iss()]
    prompt_input += [summ_str]
    # prompt_input += [persona.scratch.get_str_curr_date_str()]
    prompt_input += [persona.scratch.get_str_firstname()]
    prompt_input += [persona.scratch.get_str_firstname()]
    prompt_input += [task]
    prompt_input += [curr_time_range]
    prompt_input += [duration]
    prompt_input += [persona.scratch.get_str_firstname()]
    if match_index !=all_indices[0]:
       prompt_input += [f"Important: The subtasks must not have anything in common with \"{persona.scratch.f_daily_schedule_hourly_org[all_indices[0]][0]}\", as {persona.scratch.get_str_firstname()} has already done that."]
    else:
       prompt_input += [""]
       
    # if "Klaus" in persona.name:
    #   sys.exit()
    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    print ("TOODOOOOOO")
    gpt_response = remove_notes_from_plan(gpt_response)
    gpt_response = fix_broken_lines(gpt_response)
    gpt_response = gpt_response.replace('\n(', '(') #remove weird newlines within the same task
    print (gpt_response)
    print ("-==- -==- -==- ")

    # TODO SOMETHING HERE sometimes fails... See screenshot
    temp = [i.strip() for i in gpt_response.split("\n")]
    _cr = []
    cr = []
    for count, i in enumerate(temp): 
      if count >= 0: 
        if "(duration in minutes:" not in i or not re.search(r"\(duration in minutes: \d+", i):
          print(f"Skipping incomplete line: {i}")
          continue  # Skip incomplete lines
        else:
          # Process the line if it is complete
          _cr.append(" ".join([j.strip() for j in i.split(" ")][3:]))
      else: 
        _cr += [i]
    for count, i in enumerate(_cr): 
      k = [j.strip() for j in i.split("(duration in minutes:")]
      task = k[0]
      if task[-1] == ".": 
        task = task[:-1]
      cleaned_string = re.sub("[^0-9]", "", k[1].split(",")[0].strip()) #remove non numeric chars that might be generated: e.g., "duration in minutes: 5)" instead of "duration in minutes: 5, 15 left)"
      duration = int(cleaned_string)  
      # duration = int(k[1].split(",")[0].strip())
      cr += [[task, duration]]

    # print("################ Prompt START ################")
    # print(prompt)
    # print("################ Prompt END ################")
    total_expected_min = int(prompt.split("(total duration in minutes")[-1]
                                   .split(")")[0].strip())
    
    # TODO -- now, you need to make sure that this is the same as the sum of 
    #         the current action sequence. 
    curr_min_slot = [["dummy", -1],] # (task_name, task_index)
    for count, i in enumerate(cr): 
      i_task = i[0] 
      i_duration = i[1]

      i_duration -= (i_duration % 5)
      if i_duration > 0: 
        for j in range(i_duration): 
          curr_min_slot += [(i_task, count)]       
    curr_min_slot = curr_min_slot[1:]   

    if len(curr_min_slot) > total_expected_min: 
      last_task = curr_min_slot[60]
      for i in range(1, 6): 
        curr_min_slot[-1 * i] = last_task
    elif len(curr_min_slot) < total_expected_min: 
      last_task = curr_min_slot[-1]
      for i in range(total_expected_min - len(curr_min_slot)):
        curr_min_slot += [last_task]

    cr_ret = [["dummy", -1],]
    for task, task_index in curr_min_slot: 
      if task != cr_ret[-1][0]: 
        cr_ret += [[task, 1]]
      else: 
        cr_ret[-1][1] += 1
    cr = cr_ret[1:]

    return cr

  def __func_validate(gpt_response, prompt=""): 
    # TODO -- this sometimes generates error 
    try: 
      __func_clean_up(gpt_response, prompt)
    except: 
      print("######### DECOMP ERROR #########")
      print(gpt_response)
      pass
      # return False
    return gpt_response

  def get_fail_safe(): 
    fs = ["asleep"]
    return fs

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 1000, 
  #            "temperature": 0.01, "top_p": 1, "stream": False,
  #            "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 1250, "temperature": 0.50, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1.25, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
    
  prompt_template = "persona/prompt_template/v2/task_decomp_v3.txt"
  prompt_input = create_prompt_input(persona, task, duration, match_index)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe()

  print ("?????")
  print (prompt)
  print ("?????")
  output = safe_generate_response(prompt, llm_param, 5, get_fail_safe(),
                                   __func_validate, __func_clean_up)


  # TODO THERE WAS A BUG HERE... 
  # This is for preventing overflows...
  """
  File "/Users/joonsungpark/Desktop/Stanford/Projects/
  generative-personas/src_exploration/reverie_simulation/
  brain/get_next_action_v3.py", line 364, in run_gpt_prompt_task_decomp
  fin_output[-1][1] += (duration - ftime_sum)
  IndexError: list index out of range
  """

  print ("IMPORTANT VVV DEBUG")

  # print (prompt_input)
  # print (prompt)
  # print (output)

  fin_output = []
  time_sum = 0
  for i_task, i_duration in output: 
    time_sum += i_duration
    # HM?????????
    # if time_sum < duration: 
    if time_sum <= duration: 
      fin_output += [[i_task, i_duration]]
    else: 
      break
  ftime_sum = 0
  for fi_task, fi_duration in fin_output: 
    ftime_sum += fi_duration
  
  # print ("for debugging... line 365", fin_output)
  fin_output[-1][1] += (duration - ftime_sum)
  output = fin_output 

  task_decomp = output
  ret = []
  for decomp_task, duration in task_decomp: 
    ret += [[f"{task} ({decomp_task})", duration]]
  output = ret


  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
    
  return output, [output, prompt, llm_param, prompt_input, fail_safe]


def run_gpt_prompt_action_sector(action_description, 
                                persona, 
                                maze, 
                                test_input=None, 
                                verbose=False):
  def create_prompt_input(action_description, persona, maze, test_input=None): 
    act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
    
    prompt_input = []
    
    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [persona.scratch.living_area.split(":")[1]]
    x = f"{act_world}:{persona.scratch.living_area.split(':')[1]}"
    prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]


    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"]
    x = f"{act_world}:{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]

    if persona.scratch.get_str_daily_plan_req() != "": 
      prompt_input += [f"\n{persona.scratch.get_str_daily_plan_req()}"]
    else: 
      prompt_input += [""]


    # MAR 11 TEMP
    accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
    curr = accessible_sector_str.split(", ")
    fin_accessible_sectors = []

    ### filter out artists coliving space if persona does not have access to any arena inside
    for i in curr: 
      special_sector = "artist's co-living space"
      if special_sector in i:
        x = f"{act_world}:{special_sector}"
        special_arenas = persona.s_mem.get_str_accessible_sector_arenas(x)
        if not special_arenas:
          continue
        else:
          special_curr = special_arenas.split(", ")
          for ii in special_curr: 
            if any(substring in ii for substring in ("'s room", "'s bathroom", "'s bedroom")): 
              if persona.scratch.last_name in ii: 
                fin_accessible_sectors += [special_sector]
                break
              else:
                pass
            else: 
              fin_accessible_sectors += [special_sector]
              break
      elif ("'s house" in i) or ('s apartment' in i): 
        if persona.scratch.last_name in i: 
          fin_accessible_sectors += [i]
      else: 
        fin_accessible_sectors += [i]
    accessible_sector_str = ", ".join(fin_accessible_sectors)
    # END MAR 11 TEMP

    prompt_input += [accessible_sector_str]



    action_description_1 = action_description
    action_description_2 = action_description
    if "(" in action_description: 
      action_description_1 = action_description.split("(")[0].strip()
      action_description_2 = action_description.split("(")[-1][:-1]
    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [action_description]
    # prompt_input += [action_description_1]
    # prompt_input += [action_description_2]
    return prompt_input




  def __func_clean_up(gpt_response, prompt=""):
    cleaned_response = gpt_response.split("{")[-1]
    cleaned_response = cleaned_response.split("}")[0]
    return cleaned_response

  def __func_validate(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 1: 
      return False
    # if "}" not in gpt_response:
    #   return False
    if "," in gpt_response: 
      return False
    return True
  
  def get_fail_safe(): 
    fs = ("kitchen")
    return fs


  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 15, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
    
  prompt_template = "persona/prompt_template/v1/action_location_sector_v1.txt"
  prompt_input = create_prompt_input(action_description, persona, maze)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
  y = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
  x = [i.strip() for i in persona.s_mem.get_str_accessible_sectors(y).split(",")]
  if output not in x: 
    # output = random.choice(x)
    output = persona.scratch.living_area.split(":")[1]

  print ("DEBUG", random.choice(x), "------", output)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)

  return output, [output, prompt, llm_param, prompt_input, fail_safe]



def run_gpt_prompt_action_arena(action_description, 
                                persona, 
                                maze, act_world, act_sector,
                                test_input=None, 
                                verbose=False):
  def create_prompt_input(action_description, persona, maze, act_world, act_sector, test_input=None): 
    prompt_input = []
    # prompt_input += [persona.scratch.get_str_name()]
    # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["arena"]]
    # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["sector"]]
    prompt_input += [persona.scratch.get_str_name()]
    curr_sector = maze.access_tile(persona.scratch.curr_tile)["sector"]
    curr_arena = maze.access_tile(persona.scratch.curr_tile)["arena"]
    prompt_input += [curr_sector]
    prompt_input += [curr_arena]
    
    x = f"{act_world}:{act_sector}"
    prompt_input += [act_sector]
    
    # MAR 11 TEMP
    accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(x)
    curr = accessible_arena_str.split(", ")
    fin_accessible_arenas = []
    for i in curr: 
      if any(substring in i for substring in ("'s room", "'s bathroom", "'s bedroom")): 
        if persona.scratch.last_name in i: 
          fin_accessible_arenas += [i]
      else: 
        fin_accessible_arenas += [i]
    accessible_arena_str = ", ".join(fin_accessible_arenas)
    # END MAR 11 TEMP

    prompt_input += [accessible_arena_str]

    action_description_1 = action_description
    action_description_2 = action_description
    # if "(" in action_description: 
    #   action_description_1 = action_description.split("(")[0].strip()
    #   action_description_2 = action_description.split("(")[-1][:-1]
    prompt_input += [action_description_1]
    prompt_input += [action_description_2]

    prompt_input += [act_sector]

    prompt_input += [accessible_arena_str]
    # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["arena"]]
    # x = f"{maze.access_tile(persona.scratch.curr_tile)['world']}:{maze.access_tile(persona.scratch.curr_tile)['sector']}:{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    # prompt_input += [persona.s_mem.get_str_accessible_arena_game_objects(x)]
    
    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    cleaned_response = gpt_response.split("{")[-1]
    cleaned_response = cleaned_response.split("}")[0]
    return cleaned_response

  def __func_validate(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 1: 
      return False
    if "," in gpt_response: 
      return False
    if gpt_response == "{}" or gpt_response == "{" or gpt_response == "}": # fail if empty / no arena provided
      return False
    if any(word in gpt_response for word in ["afraid", "can't", "cannot", "assist"]): # capture refusal
      print("REFUSAL: area selection")
      return False
    return True
  
  def get_fail_safe(): 
    fs = ("kitchen")
    return fs

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
    
  llm_param = {"max_new_tokens": 15, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
  
  prompt_template = "persona/prompt_template/v1/action_location_object_vMar11.txt"
  prompt_input = create_prompt_input(action_description, persona, maze, act_world, act_sector)
  prompt = generate_prompt(prompt_input, prompt_template)

  ### in case non-accessible location is chosen:
  x = f"{act_world}:{act_sector}"
  accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(x)
  curr = accessible_arena_str.split(", ")
  fin_accessible_arenas = []
  for i in curr: 
    if any(substring in i for substring in ("'s room", "'s bathroom", "'s bedroom")): 
      if persona.scratch.last_name in i: 
        fin_accessible_arenas += [i]
    else: 
      fin_accessible_arenas += [i] # list of accessible areas

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if output == "":
    output = fail_safe

  print (output)
    # check if output matches partially any location (e.g., "bedroom" instead of "Eddy's bedroom")
      ## find all locations that match
      ## check if matching locations contain name
      ## choose randomly among candidates
  partial_matching = [s for s in fin_accessible_arenas if output in s]
  partial_matching_w_name = [s for s in partial_matching if persona.scratch.get_str_name() in s]

  if output not in fin_accessible_arenas:
    if partial_matching_w_name:
        output = random.choice(partial_matching_w_name)
    else:
        if partial_matching:
            output = random.choice(partial_matching)
        else:
            output = random.choice(fin_accessible_arenas)
    print ("updated output: ", output)
  else:
    pass    

  # y = f"{act_world}:{act_sector}"
  # x = [i.strip() for i in persona.s_mem.get_str_accessible_sector_arenas(y).split(",")]
  # if output not in x: 
  #   output = random.choice(x)

  ### You could add that all punctuation is stripped from output string!

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)

  return output, [output, prompt, llm_param, prompt_input, fail_safe]



def run_gpt_prompt_action_game_object(action_description, 
                                      persona, 
                                      maze,
                                      temp_address,
                                      test_input=None, 
                                      verbose=False): 
  def create_prompt_input(action_description, 
                          persona, 
                          temp_address, 
                          test_input=None): 
    prompt_input = []
    if "(" in action_description: 
      action_description = action_description.split("(")[-1][:-1]
      
    prompt_input += [action_description]
    prompt_input += [persona
                     .s_mem.get_str_accessible_arena_game_objects(temp_address)]
    return prompt_input
  
  def __func_validate(gpt_response, prompt=""): 
    if len(gpt_response.strip()) < 1: 
      return False
    return True

  def __func_clean_up(gpt_response, prompt=""):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  def get_fail_safe(): 
    fs = ("bed")
    return fs

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 15, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
    
  prompt_template = "persona/prompt_template/v1/action_object_v2.txt"
  prompt_input = create_prompt_input(action_description, 
                                     persona, 
                                     temp_address, 
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  x = [i.strip() for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",")]
  if output not in x: 
    output = random.choice(x)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]




def run_gpt_prompt_pronunciatio(action_description, persona, verbose=False): 
  def create_prompt_input(action_description): 
    if "(" in action_description: 
      action_description = action_description.split("(")[-1].split(")")[0]
    prompt_input = [action_description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    cr = gpt_response.strip()
    if len(cr) > 3:
      cr = cr[:3]
    return cr

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt="")
      if len(gpt_response) == 0: 
        return False
    except: return False
    return True 

  def get_fail_safe(): 
    fs = "🤷‍♂️" # replace empty icons with icon symbolizing nothing/unaffected/dont know
    return fs


  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    cr = gpt_response.strip()
    if len(cr) > 3:
      cr = cr[:3]
    if cr == "":
      cr =  "🤷‍♂️"
    return cr

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt="")
      if gpt_response == "":
        gpt_response =  "🤷‍♂️"
      if len(gpt_response) == 0: 
        return False
    except: return False
    return True 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 4") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 15, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
    
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt" ########
  prompt_input = create_prompt_input(action_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = '🛁🧖' ########  
  special_instruction = "The value for the output must ONLY contain the emojis." ########
  fail_safe = get_fail_safe()
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]






def run_gpt_prompt_event_triple(action_description, persona, verbose=False): 
  def create_prompt_input(action_description, persona): 
    if "(" in action_description: 
      action_description = action_description.split("(")[-1].split(")")[0]
    prompt_input = [persona.name, 
                    action_description,
                    persona.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    print("#############")
    print(gpt_response)
    print("#############")
    cr = gpt_response.strip()
    cr = [i.strip() for i in cr.split(")")[0].split(",")]
    return cr

  def __func_validate(gpt_response, prompt=""): 
    try: 
      gpt_response = __func_clean_up(gpt_response, prompt="")
      if not (len(gpt_response) == 2 or len(gpt_response) ==3): # models can return either (Subject, predicate, object) or without subject
        return False
    except: return False
    return True 

  def get_fail_safe(persona): 
    fs = (persona.name, "is", "idle")
    return fs

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 30, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": ["\n"]}

  llm_param = {"max_new_tokens": 30, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": ["\n"]
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
    
  prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
  prompt_input = create_prompt_input(action_description, persona)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe(persona) ########
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
  if len(output) == 3:
    output = (persona.name, output[1], output[2])
  elif len(output) == 2:
    output = (persona.name, output[0], output[1])
  else:
    output = fail_safe

  print("output", output)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]


############ Check why this output does not work!
def run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona, verbose=False): 
  def create_prompt_input(act_game_object, act_desp, persona): 
    prompt_input = [act_game_object, 
                    persona.name,
                    act_desp,
                    act_game_object,
                    act_game_object]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    cr = gpt_response.strip()
    if cr[-1] == ".": cr = cr[:-1]
    return cr

  def __func_validate(gpt_response, prompt=""): 
    try: 
      gpt_response = __func_clean_up(gpt_response, prompt="")
    except: 
      return False
    return True 

  def get_fail_safe(act_game_object): 
    fs = f"{act_game_object} is idle"
    return fs

  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    cr = gpt_response.strip()
    if cr[-1] == ".": cr = cr[:-1]
    return cr

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      gpt_response = __func_clean_up(gpt_response, prompt="")
    except: 
      return False
    return True 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 6") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 15, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
    
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt" ########
  prompt_input = create_prompt_input(act_game_object, act_desp, persona)  ########
  prompt = generate_prompt(prompt_input, prompt_template)

  example_output = "being fixed" ########
  special_instruction = "The output should ONLY contain the phrase that should go in <fill in>." ########
  fail_safe = get_fail_safe(act_game_object) ########
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  print("output", output)
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]


def run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona, verbose=False): 
  def create_prompt_input(act_game_object, act_obj_desc): 
    prompt_input = [act_game_object, 
                    act_obj_desc,
                    act_game_object]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    # print("#############")
    # print(gpt_response)
    # print("#############")
    cr = gpt_response.strip()
    cr = [i.strip() for i in cr.split(")")[0].split(",")]
    return cr

  def __func_validate(gpt_response, prompt=""): 
    try: 
      gpt_response = __func_clean_up(gpt_response, prompt="")
      if not (len(gpt_response) == 2 or len(gpt_response) ==3): 
        return False
    except: return False
    return True 

  def get_fail_safe(act_game_object): 
    fs = (act_game_object, "is", "idle")
    return fs

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 30, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": ["\n"]}

  llm_param = {"max_new_tokens": 30, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": ["\n"]
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
    
  prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
  prompt_input = create_prompt_input(act_game_object, act_obj_desc)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe(act_game_object)
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
  if len(output) == 3:
    output = (act_game_object, output[1], output[2])
  elif len(output) == 2:
    output = (act_game_object, output[0], output[1])
  else:
    output = fail_safe

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]





def run_gpt_prompt_new_decomp_schedule(persona, 
                                       main_act_dur, 
                                       truncated_act_dur, 
                                       start_time_hour,
                                       end_time_hour, 
                                       inserted_act,
                                       inserted_act_dur,
                                       test_input=None, 
                                       verbose=False): 
  def create_prompt_input(persona, 
                           main_act_dur, 
                           truncated_act_dur, 
                           start_time_hour,
                           end_time_hour, 
                           inserted_act,
                           inserted_act_dur,
                           test_input=None): 
    persona_name = persona.name
    start_hour_str = start_time_hour.strftime("%H:%M %p")
    end_hour_str = end_time_hour.strftime("%H:%M %p")

    original_plan = ""
    for_time = start_time_hour
    for i in main_act_dur: 
      original_plan += f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + i[0]
      original_plan += "\n"
      for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init = ""
    for_time = start_time_hour
    for count, i in enumerate(truncated_act_dur): 
      new_plan_init += f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + i[0]
      new_plan_init += "\n"
      if count < len(truncated_act_dur) - 1: 
        for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init += (for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M") + " ~"

    prompt_input = [persona_name, 
                    start_hour_str,
                    end_hour_str,
                    original_plan,
                    persona_name,
                    inserted_act,
                    inserted_act_dur,
                    persona_name,
                    start_hour_str,
                    end_hour_str,
                    end_hour_str,
                    new_plan_init]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    new_schedule = prompt + " " + gpt_response.strip()
    new_schedule = new_schedule.split("The revised schedule:")[-1].strip()
    new_schedule = new_schedule.split("\n")

    ret_temp = []
    for i in new_schedule: 
      ret_temp += [i.split(" -- ")]

    ret = []
    for time_str, action in ret_temp:
      start_time = time_str.split(" ~ ")[0].strip()
      end_time = time_str.split(" ~ ")[1].strip()
      delta = datetime.datetime.strptime(end_time, "%H:%M") - datetime.datetime.strptime(start_time, "%H:%M")
      delta_min = int(delta.total_seconds()/60)
      if delta_min < 0: delta_min = 0
      ret += [[action, delta_min]]

    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      gpt_response = __func_clean_up(gpt_response, prompt)
      dur_sum = 0
      for act, dur in gpt_response: 
        dur_sum += dur
        if str(type(act)) != "<class 'str'>":
          return False 
        if str(type(dur)) != "<class 'int'>":
          return False
      x = prompt.split("\n")[0].split("originally planned schedule from")[-1].strip()[:-1]
      x = [datetime.datetime.strptime(i.strip(), "%H:%M %p") for i in x.split(" to ")]
      delta_min = int((x[1] - x[0]).total_seconds()/60)

      if int(dur_sum) != int(delta_min): 
        return False

    except: 
      return False
    return True 

  def get_fail_safe(main_act_dur, truncated_act_dur): 
    dur_sum = 0
    for act, dur in main_act_dur: dur_sum += dur

    ret = truncated_act_dur[:]
    ret += main_act_dur[len(ret)-1:]

    # If there are access, we need to trim... 
    ret_dur_sum = 0
    count = 0
    over = None
    for act, dur in ret: 
      ret_dur_sum += dur
      if ret_dur_sum == dur_sum: 
        break
      if ret_dur_sum > dur_sum: 
        over = ret_dur_sum - dur_sum
        break
      count += 1 

    if over: 
      ret = ret[:count+1]
      ret[-1][1] -= over

    return ret

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 1000, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 1000, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
    
  prompt_template = "persona/prompt_template/v2/new_decomp_schedule_v1.txt"
  prompt_input = create_prompt_input(persona, 
                                     main_act_dur, 
                                     truncated_act_dur, 
                                     start_time_hour,
                                     end_time_hour, 
                                     inserted_act,
                                     inserted_act_dur,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe(main_act_dur, truncated_act_dur)
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
  
  # print ("* * * * output")
  # print (output)
  # print ('* * * * fail_safe')
  # print (fail_safe)



  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]

### add group identity and threat stimuli to context/persona descriptions
def run_gpt_prompt_decide_to_talk(persona, target_persona, retrieved,test_input=None, 
                                       verbose=False): 
  def create_prompt_input(init_persona, target_persona, retrieved, 
                          test_input=None): 
    
    def truncate_context(context, word_limit=5000):
      words = context.split()  # Split context into words
      truncated_words = words[:word_limit]  # Take only the first 500 words
      truncated_context = " ".join(truncated_words)  # Recombine into a string
      return truncated_context
    
    last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
    last_chatted_time = ""
    last_chat_about = ""
    if last_chat: 
      last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
      last_chat_about = last_chat.description

    context = ""
    for c_node in retrieved["events"]: 
      curr_desc = c_node.description.split(" ")
      curr_desc[2:3] = ["was"]
      curr_desc = " ".join(curr_desc)
      context +=  f"{curr_desc}. "
    context += "\n"
    for c_node in retrieved["thoughts"]: 
      context +=  f"{c_node.description}. "
    context = truncate_context(context) # max word limit for context

    curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_act_desc = init_persona.scratch.act_description
    if "(" in init_act_desc: 
      init_act_desc = init_act_desc.split("(")[-1][:-1]
    
    if len(init_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc: 
      init_p_desc = f"{init_persona.name} is already {init_act_desc}"
    elif "waiting" in init_act_desc:
      init_p_desc = f"{init_persona.name} is {init_act_desc}"
    else: 
      init_p_desc = f"{init_persona.name} is on the way to {init_act_desc}"

    target_act_desc = target_persona.scratch.act_description
    if "(" in target_act_desc: 
      target_act_desc = target_act_desc.split("(")[-1][:-1]
    
    if len(target_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc: 
      target_p_desc = f"{target_persona.name} is already {target_act_desc}"
    elif "waiting" in init_act_desc:
      target_p_desc = f"{init_persona.name} is {init_act_desc}"
    else: 
      target_p_desc = f"{target_persona.name} is on the way to {target_act_desc}"

    ## added group identity
      ## add threat stimuli? Added after group identity conditionally? 
    perceived_features = f"Here are some things that {init_persona.name} perceives of {target_persona.name}:\n"
    for feature in target_persona.scratch.features:
      if not feature[0] == "Group Identity":
        if isinstance(feature[1][0], (int, float)):
            perceived_features += f"{feature[0]}: {feature[1][0]} on a scale from {feature[1][1][0]} to {feature[1][1][1]}\n"
        else:
            perceived_features += f"{feature[0]}: {feature[1][0]}\n"
      else:
        if target_persona.scratch.group_condition in [1,2,4,6]:
          perceived_features += f"Group Identity: {feature[1][0]}\n"

    prompt_input = []
    prompt_input += [context]

    prompt_input += [curr_time]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    prompt_input += [last_chatted_time]
    prompt_input += [last_chat_about]

    prompt_input += [init_p_desc]
    prompt_input += [target_p_desc]
    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    prompt_input += [perceived_features]
    group_information = f"""Some additional information about {init_persona.scratch.name}:\n"""
    if init_persona.scratch.group_condition in [1,2,4,6]: #add info about threat in respective conditions
      group_information += f"{init_persona.scratch.group_identity_text}\n"
    if init_persona.scratch.group_condition in [1,2,3,4,6]:
      group_information += f"{init_persona.scratch.threat_text}\n"
    if init_persona.scratch.group_condition not in [1,2,3,4,6]:
      group_information = ""

    prompt_input += [group_information]
    return prompt_input
  
  def __func_validate(gpt_response, prompt=""): 
    try: 
      if gpt_response.split("Answer in yes or no:")[-1].strip().lower() in ["yes", "no"]: 
        return True
      return False     
    except:
      return False 

  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split("Answer in yes or no:")[-1].strip().lower()

  def get_fail_safe(): 
    fs = "yes"
    return fs



  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 20, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 20, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
    
  prompt_template = "persona/prompt_template/v2/decide_to_talk_v2.txt"
  prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]

### add group identity and threat stimuli to context/persona descriptions
def run_gpt_prompt_decide_to_react(persona, target_persona, retrieved,test_input=None, 
                                       verbose=False): 
  def create_prompt_input(init_persona, target_persona, retrieved, 
                          test_input=None): 
    
    def truncate_context(context, word_limit=5000):
      words = context.split()  # Split context into words
      truncated_words = words[:word_limit]  # Take only the first 500 words
      truncated_context = " ".join(truncated_words)  # Recombine into a string
      return truncated_context

    context = ""
    for c_node in retrieved["events"]: 
      curr_desc = c_node.description.split(" ")
      curr_desc[2:3] = ["was"]
      curr_desc = " ".join(curr_desc)
      context +=  f"{curr_desc}. "
    context += "\n"
    for c_node in retrieved["thoughts"]: 
      context +=  f"{c_node.description}. "
    context = truncate_context(context) # max word limit for context

    curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_act_desc = init_persona.scratch.act_description
    if "(" in init_act_desc: 
      init_act_desc = init_act_desc.split("(")[-1][:-1]
    if len(init_persona.scratch.planned_path) == 0: 
      loc = ""
      if ":" in init_persona.scratch.act_address:
        loc = init_persona.scratch.act_address.split(":")[-1] + " in " + init_persona.scratch.act_address.split(":")[-2]
      init_p_desc = f"{init_persona.name} is already {init_act_desc} at {loc}"
    else: 
      loc = ""
      if ":" in init_persona.scratch.act_address:
        loc = init_persona.scratch.act_address.split(":")[-1] + " in " + init_persona.scratch.act_address.split(":")[-2]
      init_p_desc = f"{init_persona.name} is on the way to {init_act_desc} at {loc}"

    target_act_desc = target_persona.scratch.act_description
    if "(" in target_act_desc: 
      target_act_desc = target_act_desc.split("(")[-1][:-1]
    if len(target_persona.scratch.planned_path) == 0: 
      loc = ""
      if ":" in target_persona.scratch.act_address:
        loc = target_persona.scratch.act_address.split(":")[-1] + " in " + target_persona.scratch.act_address.split(":")[-2]
      target_p_desc = f"{target_persona.name} is already {target_act_desc} at {loc}"
    else: 
      loc = ""
      if ":" in target_persona.scratch.act_address:
        loc = target_persona.scratch.act_address.split(":")[-1] + " in " + target_persona.scratch.act_address.split(":")[-2]
      target_p_desc = f"{target_persona.name} is on the way to {target_act_desc} at {loc}"

    ## added group identity
      ## add threat stimuli? Added after group identity conditionally? 
    perceived_features = f"Here are some things that {init_persona.scratch.name} perceives of {target_persona.scratch.name}:\n"
    for feature in target_persona.scratch.features:
      if not feature[0] == "Group Identity":
        if isinstance(feature[1][0], (int, float)):
            perceived_features += f"{feature[0]}: {feature[1][0]} on a scale from {feature[1][1][0]} to {feature[1][1][1]}\n"
        else:
            perceived_features += f"{feature[0]}: {feature[1][0]}\n"
      else:
        if target_persona.scratch.group_condition in [1,2,4,6]:
          perceived_features += f"Group Identity: {feature[1][0]}\n"

    # add perceived features here and to the prompt template
    prompt_input = []
    prompt_input += [context]
    prompt_input += [curr_time]
    prompt_input += [init_p_desc]
    prompt_input += [target_p_desc]

    prompt_input += [init_persona.name]
    prompt_input += [init_act_desc]
    prompt_input += [target_persona.name]
    prompt_input += [target_act_desc]
    prompt_input += [init_act_desc]

    prompt_input += [perceived_features]

    group_information = f"""Some additional information about {init_persona.scratch.name}:\n"""
    if init_persona.scratch.group_condition in [1,2,4,6]: #add info about threat in respective conditions
      group_information += f"{init_persona.scratch.group_identity_text}\n"
    if init_persona.scratch.group_condition in [1,2,3,4,6]:
      group_information += f"{init_persona.scratch.threat_text}\n"
    if init_persona.scratch.group_condition not in [1,2,3,4,6]:
      group_information = ""

    prompt_input += [group_information]
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
    fs = "3"
    return fs


  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 20, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

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
  prompt_template = "persona/prompt_template/v2/decide_to_react_v1.txt"
  prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]


def run_gpt_prompt_create_conversation(persona, target_persona, curr_loc,
                                       test_input=None, verbose=False): 
  def create_prompt_input(init_persona, target_persona, curr_loc, 
                          test_input=None): 

    prev_convo_insert = "\n"
    if init_persona.a_mem.seq_chat: 
      for i in init_persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((init_persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, they had the following conversation.\n'
          for row in i.filling: 
            prev_convo_insert += f'{row[0]}: "{row[1]}"\n'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if init_persona.a_mem.seq_chat: 
      if int((init_persona.scratch.curr_time - init_persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""


    init_persona_thought_nodes = init_persona.a_mem.retrieve_relevant_thoughts(target_persona.scratch.act_event[0],
                                target_persona.scratch.act_event[1],
                                target_persona.scratch.act_event[2])
    init_persona_thought = ""
    for i in init_persona_thought_nodes: 
      init_persona_thought += f"-- {i.description}\n"

    target_persona_thought_nodes = target_persona.a_mem.retrieve_relevant_thoughts(init_persona.scratch.act_event[0],
                                init_persona.scratch.act_event[1],
                                init_persona.scratch.act_event[2])
    target_persona_thought = ""
    for i in target_persona_thought_nodes: 
      target_persona_thought += f"-- {i.description}\n"

    init_persona_curr_desc = ""
    if init_persona.scratch.planned_path: 
      init_persona_curr_desc = f"{init_persona.name} is on the way to {init_persona.scratch.act_description}"
    else: 
      init_persona_curr_desc = f"{init_persona.name} is {init_persona.scratch.act_description}"

    target_persona_curr_desc = ""
    if target_persona.scratch.planned_path: 
      target_persona_curr_desc = f"{target_persona.name} is on the way to {target_persona.scratch.act_description}"
    else: 
      target_persona_curr_desc = f"{target_persona.name} is {target_persona.scratch.act_description}"
 

    curr_loc = curr_loc["arena"]

    prompt_input = []
    prompt_input += [init_persona.scratch.get_str_iss()]
    prompt_input += [target_persona.scratch.get_str_iss()]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    prompt_input += [init_persona_thought]

    prompt_input += [target_persona.name]
    prompt_input += [init_persona.name]
    prompt_input += [target_persona_thought]

    prompt_input += [init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S")]

    prompt_input += [init_persona_curr_desc]
    prompt_input += [target_persona_curr_desc]

    prompt_input += [prev_convo_insert]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]

    prompt_input += [curr_loc]
    prompt_input += [init_persona.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    # print ("???")
    # print (gpt_response)


    gpt_response = (prompt + gpt_response).split("What would they talk about now?")[-1].strip()
    content = re.findall('"([^"]*)"', gpt_response)

    speaker_order = []
    for i in gpt_response.split("\n"): 
      name = i.split(":")[0].strip() 
      if name: 
        speaker_order += [name]

    ret = []
    for count, speaker in enumerate(speaker_order): 
      ret += [[speaker, content[count]]]

    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(init_persona, target_persona): 
    convo = [[init_persona.name, "Hi!"], 
             [target_persona.name, "Hi!"]]
    return convo


  gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 1000, 
               "temperature": 0.7, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 1000, "temperature": 0.7, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  
  prompt_template = "persona/prompt_template/v2/create_conversation_v2.txt"
  prompt_input = create_prompt_input(persona, target_persona, curr_loc, 
                                     test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe(persona, target_persona)
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]









# adjusted prompt template to fit in better with how LLM_safe_generate_response() and the instructions work!
  # potentially change which function is used to prompt (the json output seems unneccessary?) Could just extract a string.
  # but works so far!
def run_gpt_prompt_summarize_conversation(persona, conversation, test_input=None, verbose=False): 
  def create_prompt_input(conversation, test_input=None): 
    convo_str = ""
    for row in conversation: 
      convo_str += f'{row[0]}: "{row[1]}"\n'

    prompt_input = [convo_str]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = gpt_response.split("conversation about")[-1]

    if not gpt_response.strip().startswith("conversing about"):
      ret = "conversing about " + gpt_response.strip()
    else:
      ret = gpt_response.strip()
    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "conversing with a housemate about morning greetings"


  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    gpt_response = gpt_response.split("conversation about")[-1]
    if not gpt_response.strip().startswith("conversing about"):
      ret = "conversing about " + gpt_response.strip()
    else:
      ret = gpt_response.strip()
    return ret

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 


  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 11") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 75, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
  
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_conversation_v1.txt" ########
  prompt_input = create_prompt_input(conversation, test_input)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  # example_output = "what to eat for lunch" ########
  # special_instruction = "The output must continue the sentence above by filling in the <fill in> tag. Don't start with 'this is a conversation about...' Just finish the sentence but do not miss any important details (including who are chatting)." ########
  fail_safe = get_fail_safe() ########
  output = safe_generate_response(prompt, llm_param, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    output = output.rstrip(".!?")
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]
  # LLM Plugin ===========================================================


  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 50, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
  # prompt_template = "persona/prompt_template/v2/summarize_conversation_v1.txt"
  # prompt_input = create_prompt_input(conversation, test_input)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, llm_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose: 
  #   print_run_prompts(prompt_template, persona, llm_param, 
  #                     prompt_input, prompt, output)
  
  # return output, [output, prompt, llm_param, prompt_input, fail_safe]




def run_gpt_prompt_extract_keywords(persona, description, test_input=None, verbose=False): 
  def create_prompt_input(description, test_input=None): 
    if "\n" in description: 
      description = description.replace("\n", " <LINE_BREAK> ")
    prompt_input = [description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    print ("???")
    print (gpt_response)
    gpt_response = gpt_response.strip().split("Emotive keywords:")
    factual = [i.strip() for i in gpt_response[0].split(",")]
    emotive = [i.strip() for i in gpt_response[1].split(",")]
    all_keywords = factual + emotive
    ret = []
    for i in all_keywords: 
      if i: 
        i = i.lower()
        if i[-1] == ".": 
          i = i[:-1]
        ret += [i]
    print (ret)
    return set(ret)

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return []

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 50, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 50, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
    
  prompt_template = "persona/prompt_template/v2/get_keywords_v1.txt"
  prompt_input = create_prompt_input(description, test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)


  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]









def run_gpt_prompt_keyword_to_thoughts(persona, keyword, concept_summary, test_input=None, verbose=False): 
  def create_prompt_input(persona, keyword, concept_summary, test_input=None): 
    prompt_input = [keyword, concept_summary, persona.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = gpt_response.strip()
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return ""

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 40, 
  #              "temperature": 0.7, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 40, "temperature": 0.7, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  
      
  prompt_template = "persona/prompt_template/v2/keyword_to_thoughts_v1.txt"
  prompt_input = create_prompt_input(persona, keyword, concept_summary)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]









def run_gpt_prompt_convo_to_thoughts(persona, 
                                    init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target, test_input=None, verbose=False): 
  def create_prompt_input(init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target, test_input=None): 
    prompt_input = [init_persona_name,
                    target_persona_name,
                    convo_str,
                    init_persona_name,
                    fin_target]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = gpt_response.strip()
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return ""

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 40, 
  #              "temperature": 0.7, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
    
  llm_param = {"max_new_tokens": 40, "temperature": 0.7, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
  
  prompt_template = "persona/prompt_template/v2/convo_to_thoughts_v1.txt"
  prompt_input = create_prompt_input(init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]



def run_gpt_prompt_event_poignancy(persona, event_description, test_input=None, verbose=False): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    if isinstance(gpt_response, int):
      return gpt_response
    
    gpt_response = int(gpt_response.strip())
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return 4



  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    if isinstance(gpt_response, int):
      return gpt_response
    
    gpt_response = int(gpt_response.strip())
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 7") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
    
  llm_param = {"max_new_tokens": 15, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
  
    
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt" ########
  prompt_input = create_prompt_input(persona, event_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "5" ########
  special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10." ########
  fail_safe = get_fail_safe() ########
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  print("output", output)
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]
  # LLM Plugin ===========================================================




  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 3, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
  # prompt_template = "persona/prompt_template/v2/poignancy_event_v1.txt"
  # prompt_input = create_prompt_input(persona, event_description)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, llm_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose: 
  #   print_run_prompts(prompt_template, persona, llm_param, 
  #                     prompt_input, prompt, output)
  
  # return output, [output, prompt, llm_param, prompt_input, fail_safe]


def run_gpt_prompt_thought_poignancy(persona, event_description, test_input=None, verbose=False): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    if isinstance(gpt_response, int):
      return gpt_response
    
    gpt_response = int(gpt_response.strip())
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return 4

  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    if isinstance(gpt_response, int):
      return gpt_response
    
    gpt_response = int(gpt_response.strip())
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 8") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 15, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
  
    
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_thought_v1.txt" ########
  prompt_input = create_prompt_input(persona, event_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "5" ########
  special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10." ########
  fail_safe = get_fail_safe() ########
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]
  # LLM Plugin ===========================================================



  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 3, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
  # prompt_template = "persona/prompt_template/v2/poignancy_thought_v1.txt"
  # prompt_input = create_prompt_input(persona, event_description)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, llm_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose: 
  #   print_run_prompts(prompt_template, persona, llm_param, 
  #                     prompt_input, prompt, output)
  
  # return output, [output, prompt, llm_param, prompt_input, fail_safe]



def run_gpt_prompt_chat_poignancy(persona, event_description, test_input=None, verbose=False): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    
    gpt_response = int(gpt_response.strip())
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return 4


  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    if isinstance(gpt_response, int):
      return gpt_response
    
    gpt_response = int(gpt_response.strip())
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 9") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
    
  llm_param = {"max_new_tokens": 15, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
  
    
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_chat_v1.txt" ########
  prompt_input = create_prompt_input(persona, event_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "5" ########
  special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10." ########
  fail_safe = get_fail_safe() ########
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]
  # LLM Plugin ===========================================================




  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 3, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
  # prompt_template = "persona/prompt_template/v2/poignancy_chat_v1.txt"
  # prompt_input = create_prompt_input(persona, event_description)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, llm_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose: 
  #   print_run_prompts(prompt_template, persona, llm_param, 
  #                     prompt_input, prompt, output)
  
  # return output, [output, prompt, llm_param, prompt_input, fail_safe]





def run_gpt_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False): 
  def create_prompt_input(persona, statements, n, test_input=None): 
    prompt_input = [statements, str(n)]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    if not type(gpt_response)==list:
      if gpt_response[0]=="[" and gpt_response[-1]=="]":
        return ast.literal_eval(gpt_response)
      else:
        gpt_response = "1) " + gpt_response.strip()
        ret = []
        for i in gpt_response.split("\n"): 
          ret += [i.split(") ")[-1].rstrip(",")]
        return ret
    else:
      return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(n): 
    return ["Who am I"] * n


  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    if not type(gpt_response)==list:
      ret = ast.literal_eval(gpt_response)
    else:
      ret = gpt_response
    return ret

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 


  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 12") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 150, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
  
    
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt" ########
  prompt_input = create_prompt_input(persona, statements, n)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = '["What should Jane do for lunch", "Does Jane like strawberry", "Who is Jane"]' ########
  special_instruction = "Output must be a list of str." ########
  fail_safe = get_fail_safe(n) ########
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]
  # LLM Plugin ===========================================================






  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 150, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 150, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  

  prompt_template = "persona/prompt_template/v2/generate_focal_pt_v1.txt"
  prompt_input = create_prompt_input(persona, statements, n)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe(n)
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]




### check if this is generated correctly (it chooses fail safe)
    # likely needs to update safe_generate_response to correctly handle the output which likely deviates from list format
    # was likely because of too low token number allowed
    # might want to check how finicky the prompt template is!
def run_gpt_prompt_insight_and_guidance(persona, statements, n, test_input=None, verbose=False): 
  def create_prompt_input(persona, statements, n, test_input=None): 
    prompt_input = [statements, str(n)]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = gpt_response.strip()
    # If the response doesn't start with a number, try to add a default prefix.
    if not gpt_response[0].isdigit():
        gpt_response = "1. " + gpt_response

    ret = dict()
    pattern = r'^(\d+)\.\s+(.*?)(?:\.\s+)?\(because of\s+([0-9]+(?:,\s*[0-9]*)*)\)?\.?$'

    for line in gpt_response.split("\n"):
        line = line.strip()
        if not line:
            continue  # Skip empty lines
            
        match = re.match(pattern, line)
        if match:
          # Extract groups:
          # group 1: the line number (not used further)
          # group 2: the insight text
          # group 3: the comma-separated numbers
          insight_text = match.group(2).strip()
          numbers_str = match.group(3)
          try:
              numbers = [int(num.strip()) for num in numbers_str.split(",") if num.strip().isdigit()]
          except Exception:
              continue  # Skip this line if conversion fails
          ret[insight_text] = numbers
        else:
            # If the line doesn't match, skip it.
          continue  
    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      if __func_clean_up(gpt_response, prompt):
        return True
      else:
        return False
    except:
      print("FAILED INSIGHT GENERATION", gpt_response)
      return False 

  def get_fail_safe(n): 
    return ["I am hungry"] * n




  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 150, 
  #              "temperature": 0.5, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 350, "temperature": 0.5, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  

  prompt_template = "persona/prompt_template/v2/insight_and_evidence_v1.txt"
  prompt_input = create_prompt_input(persona, statements, n)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe(n)
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]








def run_gpt_prompt_agent_chat_summarize_ideas(persona, target_persona, statements, curr_context, test_input=None, verbose=False): 
  def create_prompt_input(persona, target_persona, statements, curr_context, test_input=None): 
    prompt_input = [persona.scratch.get_str_curr_date_str(), curr_context, persona.scratch.currently, 
                    statements, persona.scratch.name, target_persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.split('"')[0].strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 17") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 100, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  
    
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_ideas_v1.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe is working on a project' ########
  special_instruction = 'The output should be a string that responds to the question.' ########
  fail_safe = get_fail_safe() ########
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]
  # LLM Plugin ===========================================================



  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 150, 
  #              "temperature": 0.5, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
  # prompt_template = "persona/prompt_template/v2/summarize_chat_ideas_v1.txt"
  # prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, llm_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose: 
  #   print_run_prompts(prompt_template, persona, llm_param, 
  #                     prompt_input, prompt, output)
  
  # return output, [output, prompt, llm_param, prompt_input, fail_safe]




def run_gpt_prompt_agent_chat_summarize_relationship(persona, target_persona, statements, test_input=None, verbose=False): 
  def create_prompt_input(persona, target_persona, statements, test_input=None): 
    prompt_input = [statements, persona.scratch.name, target_persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.split('"')[0].strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 18") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 100, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
    
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, statements)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe is working on a project' ########
  special_instruction = 'The output should be a string that responds to the question.' ########
  fail_safe = get_fail_safe() ########
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]
  # LLM Plugin ===========================================================


  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 150, 
  #              "temperature": 0.5, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
  # prompt_template = "persona/prompt_template/v2/summarize_chat_relationship_v1.txt"
  # prompt_input = create_prompt_input(persona, target_persona, statements)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, llm_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose: 
  #   print_run_prompts(prompt_template, persona, llm_param, 
  #                     prompt_input, prompt, output)
  
  # return output, [output, prompt, llm_param, prompt_input, fail_safe]





def run_gpt_prompt_agent_chat(maze, persona, target_persona,
                               curr_context, 
                               init_summ_idea, 
                               target_summ_idea, test_input=None, verbose=False): 
  def create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea, test_input=None): 
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat: 
      for i in persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description}. This context takes place after that conversation.'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if persona.a_mem.seq_chat: 
      if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""
    print (prev_convo_insert)

    curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    curr_location = f"{curr_arena} in {curr_sector}"
    

    prompt_input = [persona.scratch.currently, 
                    target_persona.scratch.currently, 
                    prev_convo_insert,
                    curr_context, 
                    curr_location,

                    persona.scratch.name,
                    init_summ_idea, 
                    persona.scratch.name,
                    target_persona.scratch.name,

                    target_persona.scratch.name,
                    target_summ_idea, 
                    target_persona.scratch.name,
                    persona.scratch.name,

                    persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    print (gpt_response)

    gpt_response = (prompt + gpt_response).split("Here is their conversation.")[-1].strip()
    content = re.findall('"([^"]*)"', gpt_response)

    speaker_order = []
    for i in gpt_response.split("\n"): 
      name = i.split(":")[0].strip() 
      if name: 
        speaker_order += [name]

    ret = []
    for count, speaker in enumerate(speaker_order): 
      ret += [[speaker, content[count]]]

    return ret



  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."




  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    # ret = ast.literal_eval(gpt_response)

    print ("a;dnfdap98fh4p9enf HEREE!!!")
    for row in gpt_response: 
      print (row)

    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    return True


  # print ("HERE JULY 23 -- ----- ") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 15, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  
  prompt_template = "persona/prompt_template/v3_ChatGPT/agent_chat_v1.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = '[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]' ########
  special_instruction = 'The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"].' ########
  fail_safe = get_fail_safe() ########
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  # print ("HERE END JULY 23 -- ----- ") ########
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]
  # LLM Plugin ===========================================================






  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 2000, 
  #              "temperature": 0.7, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
  # prompt_template = "persona/prompt_template/v2/agent_chat_v1.txt"
  # prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, llm_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose: 
  #   print_run_prompts(prompt_template, persona, llm_param, 
  #                     prompt_input, prompt, output)
  
  # return output, [output, prompt, llm_param, prompt_input, fail_safe]


# =======================
# =======================
# =======================
# =======================







def run_gpt_prompt_summarize_ideas(persona, statements, question, test_input=None, verbose=False): 
  def create_prompt_input(persona, statements, question, test_input=None): 
    prompt_input = [statements, persona.scratch.name, question]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.split('"')[0].strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 16") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 100, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_ideas_v1.txt" ########
  prompt_input = create_prompt_input(persona, statements, question)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe is working on a project' ########
  special_instruction = 'The output should be a string that responds to the question.' ########
  fail_safe = get_fail_safe() ########
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]
  # LLM Plugin ===========================================================


  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 150, 
  #              "temperature": 0.5, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
  # prompt_template = "persona/prompt_template/v2/summarize_ideas_v1.txt"
  # prompt_input = create_prompt_input(persona, statements, question)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, llm_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose: 
  #   print_run_prompts(prompt_template, persona, llm_param, 
  #                     prompt_input, prompt, output)
  
  # return output, [output, prompt, llm_param, prompt_input, fail_safe]



def run_gpt_prompt_generate_next_convo_line(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None, verbose=False): 
  def create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None): 
    prompt_input = [persona.scratch.name, 
                    persona.scratch.get_str_iss(),
                    persona.scratch.name, 
                    interlocutor_desc, 
                    prev_convo, 
                    persona.scratch.name,
                    retrieved_summary, 
                    persona.scratch.name,]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."



  # # LLM Plugin ===========================================================
  # def __chat_func_clean_up(gpt_response, prompt=""): ############
  #   return gpt_response.split('"')[0].strip()

  # def __chat_func_validate(gpt_response, prompt=""): ############
  #   try: 
  #     __func_clean_up(gpt_response, prompt)
  #     return True
  #   except:
  #     return False 

  # print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 15") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
  # prompt_template = "persona/prompt_template/v3_ChatGPT/generate_next_convo_line_v1.txt" ########
  # prompt_input = create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary)  ########
  # prompt = generate_prompt(prompt_input, prompt_template)
  # example_output = 'Hello' ########
  # special_instruction = 'The output should be a string that responds to the question. Again, only use the context included in the "Note" to generate the response' ########
  # fail_safe = get_fail_safe() ########
  # output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
  #                                         __chat_func_validate, __chat_func_clean_up, True)
  # if output != False: 
  #   return output, [output, prompt, llm_param, prompt_input, fail_safe]
  # # LLM Plugin ===========================================================



  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 250, 
  #              "temperature": 1, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 250, "temperature": 1, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  
  prompt_template = "persona/prompt_template/v2/generate_next_convo_line_v1.txt"
  prompt_input = create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]






def run_gpt_prompt_generate_whisper_inner_thought(persona, whisper, test_input=None, verbose=False): 
  def create_prompt_input(persona, whisper, test_input=None): 
    prompt_input = [persona.scratch.name, whisper]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 50, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 75, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
    
  prompt_template = "persona/prompt_template/v2/whisper_inner_thought_v1.txt"
  prompt_input = create_prompt_input(persona, whisper)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]



def run_gpt_prompt_planning_thought_on_convo(persona, all_utt, test_input=None, verbose=False): 
  def create_prompt_input(persona, all_utt, test_input=None): 
    prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 50, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 50, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }

  prompt_template = "persona/prompt_template/v2/planning_thought_on_convo_v1.txt"
  prompt_input = create_prompt_input(persona, all_utt)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]



def run_gpt_prompt_memo_on_convo(persona, all_utt, test_input=None, verbose=False): 
  def create_prompt_input(persona, all_utt, test_input=None): 
    prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # LLM Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 


  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 15") ########
  # gpt_param = {"engine": "text-davinci-002", "max_new_tokens": 15, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 50, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  
  prompt_template = "persona/prompt_template/v3_ChatGPT/memo_on_convo_v1.txt" ########
  prompt_input = create_prompt_input(persona, all_utt)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe was interesting to talk to.' ########
  special_instruction = 'The output should ONLY contain a string that summarizes anything interesting that the agent may have noticed' ########
  fail_safe = get_fail_safe() ########
  output = LLM_safe_generate_response(prompt, llm_param, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, llm_param, prompt_input, fail_safe]
  else:
    return fail_safe, [output, prompt, llm_param, prompt_input, fail_safe]
  # LLM Plugin ===========================================================

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 50, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
    
  llm_param = {"max_new_tokens": 50, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }

  prompt_template = "persona/prompt_template/v2/memo_on_convo_v1.txt"
  prompt_input = create_prompt_input(persona, all_utt)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, llm_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, llm_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, llm_param, prompt_input, fail_safe]




def run_gpt_generate_safety_score(persona, comment, test_input=None, verbose=False): 
  def create_prompt_input(comment, test_input=None):
    prompt_input = [comment]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    # gpt_response = json.loads(gpt_response)
    return gpt_response["output"]

  def __chat_func_validate(gpt_response, prompt=""): 
    try: 
      fields = ["output"]
      response = json.loads(gpt_response)
      for field in fields: 
        if field not in response: 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  print ("11")
  prompt_template = "persona/prompt_template/safety/anthromorphosization_v1.txt" 
  prompt_input = create_prompt_input(comment) 
  print ("22")
  prompt = generate_prompt(prompt_input, prompt_template)
  print (prompt)
  fail_safe = get_fail_safe() 

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 50, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}
    
  llm_param = {"max_new_tokens": 50, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
     }
  
  output = LLM_safe_generate_response_OLD(prompt, 3, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, llm_param, verbose)
  print (output)

  return output, [output, prompt, llm_param, prompt_input, fail_safe]


def run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None, verbose=False): 
  def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None):
    persona = init_persona
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat: 
      for i in persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description}. This context takes place after that conversation.'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if persona.a_mem.seq_chat: 
      if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""

    curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    curr_location = f"{curr_arena} in {curr_sector}"

    retrieved_str = ""
    for key, vals in retrieved.items(): 
      for v in vals: 
        retrieved_str += f"- {v.description}\n"


    convo_str = ""
    for i in curr_chat:
      convo_str += ": ".join(i) + "\n"
    if convo_str == "": 
      convo_str = "\nThe conversation has not started yet -- start it!"

    ISS = ""
    ISS += f"-Age: {init_persona.scratch.age}\n"
    ISS += f"-Personality: {init_persona.scratch.innate}\n"
    ISS += f"-Short biography: {init_persona.scratch.learned}\n"
    ISS += f"-Living context: {init_persona.scratch.currently}\n" # summary about self
    if init_persona.scratch.group_condition in [1,2,4,6]:
      ISS += f"Group Identity: {init_persona.scratch.group_identity_text}\n"
    if init_persona.scratch.group_condition in [1,2,3,4,6]:
      ISS += f"Additional information: {init_persona.scratch.threat_text}\n"

    init_iss = f"You are {init_persona.scratch.name}. Here is some information about your personality, biography, and living context:\n{ISS}\n"
    # init_iss = f"Here is a brief description of you, {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"

    # Do I need to take short biography out? And personality? Is this too much information?
    init_iss_target  = f"You are talking to {target_persona.scratch.name}. Here is a brief description of them:\n"
    init_iss_target += f"-Age: {target_persona.scratch.age}\n"
    init_iss_target += f"-Personality: {target_persona.scratch.innate}\n"
    init_iss_target += f"-Short biography: {target_persona.scratch.learned}\n"
    
    perceived_features = f"Here are some things you perceive of {target_persona.name}:\n"
    for feature in target_persona.scratch.features:
      if not feature[0] == "Group Identity":
        if isinstance(feature[1][0], (int, float)):
            perceived_features += f"{feature[0]}: {feature[1][0]} on a scale from {feature[1][1][0]} to {feature[1][1][1]}\n"
        else:
            perceived_features += f"{feature[0]}: {feature[1][0]}\n"
      else:
        if target_persona.scratch.group_condition in [1,2,4,6]:
          perceived_features += f"Group Identity: {feature[1][0]}\n"      
        
    prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
      curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
      convo_str, init_persona.scratch.name, target_persona.scratch.name,
      init_persona.scratch.name, init_persona.scratch.name,
      init_persona.scratch.name, perceived_features, init_iss_target
      ]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 

    cleaned_dict = dict()
    cleaned = []
    for key, val in gpt_response.items(): 
      cleaned += [val]
    cleaned_dict["utterance"] = cleaned[0]
    cleaned_dict["end"] = True
    if "f" in str(cleaned[1]) or "F" in str(cleaned[1]): 
      cleaned_dict["end"] = False

    return cleaned_dict

  def __chat_func_validate(gpt_response, prompt=""): 
    print ("ugh...")
    try: 
      print(__chat_func_clean_up(gpt_response))
      return True
    except:
      return False 

  def get_fail_safe():
    cleaned_dict = dict()
    cleaned_dict["utterance"] = "..."
    cleaned_dict["end"] = False
    return cleaned_dict
  
  llm_param = {"max_new_tokens": 350, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
  }

  print ("11")
  prompt_template = "persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt" 
  prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat) 
  print ("22")
  prompt = generate_prompt(prompt_input, prompt_template) 
  print (prompt)
  fail_safe = get_fail_safe() 
  output = LLM_safe_generate_response_OLD(prompt, 3, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, llm_param, verbose)
  print (output)
  
  

  return output, [output, prompt, llm_param, prompt_input, fail_safe]


def run_gpt_interviewer(hiring_persona, employee_persona, retrieved, curr_chat, job_details_str, relationship, test_input=None, verbose=False): 
  def __chat_func_clean_up(gpt_response, prompt=""): 
    cleaned_dict = dict()
    cleaned = []
    for key, val in gpt_response.items(): 
      cleaned += [val]
    cleaned_dict["utterance"] = cleaned[0]
    cleaned_dict["end"] = True
    if "f" in str(cleaned[1]) or "F" in str(cleaned[1]): 
      cleaned_dict["end"] = False

    return cleaned_dict

  def __chat_func_validate(gpt_response, prompt=""): 
    print ("ugh...")
    try: 
      print("gpt_response", gpt_response)
      print(__chat_func_clean_up(gpt_response))
      return True
    except:
      return False 

  def get_fail_safe():
    cleaned_dict = dict()
    cleaned_dict["utterance"] = "..."
    cleaned_dict["end"] = False
    return cleaned_dict

  llm_param = {"max_new_tokens": 250, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
  }

  retrieved_str = ""
  for key, vals in retrieved.items(): 
    for v in vals: 
      retrieved_str += f"- {v.description}\n"

  prev_convo_insert = ""
  for i in curr_chat:
    prev_convo_insert += ": ".join(i) + "\n"

  perceived_features = ""
  for feature in employee_persona.scratch.features:
    if not feature[0] == "Group Identity":
      if isinstance(feature[1][0], (int, float)):
          perceived_features += f"{feature[0]}: {feature[1][0]} on a scale from {feature[1][1][0]} to {feature[1][1][1]}\n"
      else:
          perceived_features += f"{feature[0]}: {feature[1][0]}\n"
    else:
      if employee_persona.scratch.group_condition in [1,2,4,6]:
        perceived_features += f"Group Identity: {feature[1][0]}\n"

  ### Add information about self:
    ## Like interview prompting!

  ISS = ""
  ISS += f"-Age: {hiring_persona.scratch.age}\n"
  ISS += f"-Personality: {hiring_persona.scratch.innate}\n"
  ISS += f"-Short biography: {hiring_persona.scratch.learned}\n"
  ISS += f"-Living context: {hiring_persona.scratch.currently}\n" # summary about self
  if hiring_persona.scratch.group_condition in [1,2,4,6]:
    ISS += f"Group Identity: {hiring_persona.scratch.group_identity_text}\n"
  if hiring_persona.scratch.group_condition in [1,2,3,4,6]:
    ISS += f"Additional information: {hiring_persona.scratch.threat_text}\n"
  
  interview_prompt = f"You are {hiring_persona.scratch.name}. " # information about self
  interview_prompt += f"Here is some information about your personality, biography, and your current living context:\n"
  interview_prompt += f"{ISS}\n"

  interview_prompt  = f"Right now, you are interviewing {employee_persona.name} for the following role: {job_details_str}.\n"
  interview_prompt += f"Here is a summary of your relationship with {employee_persona.name}: {relationship}\n"
  interview_prompt += f"Here are some things you perceive of {employee_persona.name}: {perceived_features}\n"
  if prev_convo_insert == "": 
    interview_prompt += f"Now, start the interview!\n"
    interview_prompt += "Based on your personality, biography, current living context, what would you say to start the interview?\n"
  else:
    interview_prompt += f"Here is the interview with {employee_persona.name} so far: {prev_convo_insert}\n"
    interview_prompt += "Based on your personality, biography, current living context, and the conversation so far, what would you say next in the conversation?\n"

  interview_prompt += "Output format: Output a json of the following format:\n{\n"
  interview_prompt += f'"{hiring_persona.scratch.name}": "<{hiring_persona.scratch.name}\'s utterance>","Did the conversation end with {hiring_persona.scratch.name}\'s utterance?": "<json Boolean>"\n'
  interview_prompt += "}\n"
  interview_prompt += f"Make sure that the conversation is reasonable given your and {employee_persona.scratch.name}'s background and context!\n"
  interview_prompt += f"Keep the utterance below 150 words and ensure that the JSON structure is fully completed:"
  # interview_prompt += "What would you say next in the conversation?"
  failsafe = get_fail_safe()
  output = LLM_safe_generate_response_OLD(interview_prompt, 3, failsafe, __chat_func_validate, __chat_func_clean_up, llm_param, 1)
  print("interview output: ", output)

  if not "utterance" in output or not "end" in output:
    return failsafe["utterance"], failsafe["end"]

  return output["utterance"], output["end"]

def run_gpt_interviewee(hiring_persona, employee_persona, retrieved, curr_chat, job_details_str, relationship, test_input=None, verbose=False): 
  def __chat_func_clean_up(gpt_response, prompt=""): 
    cleaned_dict = dict()
    cleaned = []
    for key, val in gpt_response.items(): 
      cleaned += [val]
    
    cleaned_dict["utterance"] = cleaned[0]
    cleaned_dict["end"] = True
    
    if "f" in str(cleaned[1]) or "F" in str(cleaned[1]): 
      cleaned_dict["end"] = False

    return cleaned_dict

  def __chat_func_validate(gpt_response, prompt=""): 
    print ("ugh...")
    try: 
      print("gpt response", gpt_response)
      print (__chat_func_clean_up(gpt_response))
      return True
    except:
      return False 

  def get_fail_safe():
    cleaned_dict = dict()
    cleaned_dict["utterance"] = "..."
    cleaned_dict["end"] = False
    return cleaned_dict

  # gpt_param = {"engine": "text-davinci-003", "max_new_tokens": 50, 
  #              "temperature": 0.01, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop_strings": None}

  llm_param = {"max_new_tokens": 250, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
  }

  retrieved_str = ""
  for key, vals in retrieved.items(): 
    for v in vals: 
      retrieved_str += f"- {v.description}\n"

  prev_convo_insert = ""
  for i in curr_chat:
    prev_convo_insert += ": ".join(i) + "\n"

  perceived_features = ""
  for feature in hiring_persona.scratch.features:
    if not feature[0] == "Group Identity":
      if isinstance(feature[1][0], (int, float)):
          perceived_features += f"{feature[0]}: {feature[1][0]} on a scale from {feature[1][1][0]} to {feature[1][1][1]}\n"
      else:
          perceived_features += f"{feature[0]}: {feature[1][0]}\n"
    else:
      if hiring_persona.scratch.group_condition in [1,2,4,6]:
        perceived_features += f"Group Identity: {feature[1][0]}\n"

  ISS = ""
  ISS += f"-Age: {employee_persona.scratch.age}\n"
  ISS += f"-Personality: {employee_persona.scratch.innate}\n"
  ISS += f"-Short biography: {employee_persona.scratch.learned}\n"
  ISS += f"-Living context: {employee_persona.scratch.currently}\n" # summary about self
  if employee_persona.scratch.group_condition in [1,2,4,6]:
    ISS += f"Group Identity: {employee_persona.scratch.group_identity_text}\n"
  if employee_persona.scratch.group_condition in [1,2,3,4,6]:
    ISS += f"Additional information: {employee_persona.scratch.threat_text}\n"
  
  interview_prompt = f"You are {employee_persona.scratch.name}. " # information about self
  interview_prompt += f"Here is some information about your personality, biography, and your current living context:\n"
  interview_prompt += f"{ISS}\n"
      
  interview_prompt = f"Right now, you are being interviewed by {hiring_persona.scratch.name} for this job role: {job_details_str}.\n"
  interview_prompt += f"Here is a summary of your relationship with {hiring_persona.scratch.name}: {relationship}\n"
  interview_prompt += f"Here are some things you perceive of {hiring_persona.scratch.name}: {perceived_features}\n"
  if prev_convo_insert == "": 
    interview_prompt += f"Now, start the interview!\n"
  else:
    interview_prompt += f"Here is the interview with {hiring_persona.scratch.name} so far: {prev_convo_insert}\n"
  
  interview_prompt += "Based on your personality, biography, current living context, and the conversation so far, what would you say next in the conversation?\n"
  interview_prompt += "Output format: Output a json of the following format:\n{"
  interview_prompt += f'"{employee_persona.scratch.name}": "<{employee_persona.scratch.name}"\'s utterance>","Did the conversation end with {employee_persona.scratch.name}\'s utterance?": "<json Boolean>"'
  interview_prompt += "}\n"
  interview_prompt += "What would you say next in the conversation?\n"
  interview_prompt += f"Make sure that the conversation is reasonable given your and {hiring_persona.scratch.name}'s background and context!\n"
  interview_prompt += f"Keep the utterance below 150 words and ensure that the JSON structure is fully completed:"

  failsafe = get_fail_safe()
  output = LLM_safe_generate_response_OLD(interview_prompt, 3, failsafe, __chat_func_validate, __chat_func_clean_up, llm_param, 1)
  print("interview output: ", output)

  if not "utterance" in output or not "end" in output:
    return failsafe["utterance"], failsafe["end"]

  print (output) 

  return output["utterance"], output["end"]


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
    job_prompt += "\nOnly add the name of the location. Do not add addresses, towns, etc."
    
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



def run_gpt_prompt_decide_interview(hiring_persona, employee_persona, relationship, perceived_features, previous_convo, job_details_str):

  ## Determines based on ISS if someone can hire others (e.g., business owner, manager)
  llm_param = {"max_new_tokens": 75, "temperature": 0.01, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
    }
  
  ISS = ""
  ISS += f"-Age: {hiring_persona.scratch.age}\n"
  ISS += f"-Personality: {hiring_persona.scratch.innate}\n"
  ISS += f"-Short biography: {hiring_persona.scratch.learned}\n"
  if hiring_persona.scratch.group_condition in [1,2,4,6]:
    ISS += f"Group Identity: {hiring_persona.scratch.group_identity_text}\n"
  if hiring_persona.scratch.group_condition in [1,2,3,4,6]:
    ISS += f"Additional information: {hiring_persona.scratch.threat_text}\n"
  
  interview_prompt = f"You are {hiring_persona.scratch.name}. " # information about self
  interview_prompt += f"Here is some information about your personality and biography:\n"
  interview_prompt += f"{ISS}\n"

  interview_prompt = f"You are looking to hire an employee for your business. Here are the details about the role: {job_details_str}\n" # information about position

  # information about candidate
  interview_prompt += f"Based on the following information about {employee_persona.scratch.name}, determine if you want to interview them for the role.\n"
  interview_prompt += f"Here is a summary of your overall relationship with {employee_persona.scratch.name}: {relationship}\n"
  interview_prompt += f"Here is your previous conversation with {employee_persona.scratch.name}: {previous_convo}\n"
  interview_prompt += f"Here are some things you perceive of {employee_persona.scratch.name}: {perceived_features}\n"

  # prompt decision
  interview_prompt += f"Do you want to interview {employee_persona.scratch.name} for this position?\n"
  interview_prompt += f"Think step by step. Consider your relationship with {employee_persona.scratch.name}, the job requirements, and your own personality and biography.\n"
  interview_prompt += f"Respond with 'yes' or 'no' (<fill in a brief explanation why>). Add the explanation in parenthesis after your decision.\n"

  output = LLM_single_request(interview_prompt, llm_param)

  if "yes" in output.lower():
    interview_status = True
  else:
    interview_status = False
    
  return interview_status


def run_gpt_prompt_accept_interview(hiring_persona, employee_persona, relationship, previous_convo, perceived_features, job_details_str):

  ## Determines based on ISS if someone can hire others (e.g., business owner, manager)
  llm_param = {"max_new_tokens": 50, "temperature": 0.01, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
    }
  
  ISS = ""
  ISS += f"-Age: {employee_persona.scratch.age}\n"
  ISS += f"-Personality: {employee_persona.scratch.innate}\n"
  ISS += f"-Short biography: {employee_persona.scratch.learned}\n"
  ISS += f"-Living context: {employee_persona.scratch.currently}\n" # summary about self
  if employee_persona.scratch.group_condition in [1,2,4,6]:
    ISS += f"Group Identity: {employee_persona.scratch.group_identity_text}\n"
  if employee_persona.scratch.group_condition in [1,2,3,4,6]:
    ISS += f"Additional information: {employee_persona.scratch.threat_text}\n"

  interview_prompt = f"You are {employee_persona.scratch.name}. " # information about self
  interview_prompt += f"Here is some information about your personality and your current living context:\n"
  interview_prompt += f"{ISS}\n"

  interview_prompt  = f"Determine if you want to interview for the following job, based on your personality, biography, living context, the potential job's details, and your relationship with your potential boss.\n"
  interview_prompt += f"Here are the job details: {job_details_str}\n"
  interview_prompt += f"Here is a summary of your overall relationship with your potential boss, {hiring_persona.scratch.name}: {relationship}\n"
  interview_prompt += f"Here is your previous conversation with your potential boss, {hiring_persona.scratch.name}: {previous_convo}\n"
  interview_prompt += f"Here are some things you perceive of your potential boss, {hiring_persona.scratch.name}: {perceived_features}\n"
  interview_prompt += f"Do you want to interview for this job offered by {hiring_persona.scratch.name}?"
  interview_prompt += f"Think step by step. Consider your relationship with {hiring_persona.scratch.name}, the job requirements, and your own personality, biography, and living context.\n"
  interview_prompt += f"Respond with 'yes' or 'no' (<fill in a brief explanation why>). Add the explanation in parenthesis after your decision.\n"

  output = LLM_single_request(interview_prompt, llm_param)

  if "yes" in output.lower():
    accept_interview = True
  else:
    accept_interview = False

  return accept_interview


def run_gpt_prompt_summarize_interview(hiring_persona, employee_persona, interview_text, job_details_str):

  ## Determines based on ISS if someone can hire others (e.g., business owner, manager)
  llm_param = {"max_new_tokens": 250, "temperature": 0.01, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
    }

  interview_prompt  = f"Summarize the following interview in less than 3 sentences.\n" 
  interview_prompt += f"{hiring_persona.scratch.name} has interviewed {employee_persona.scratch.name} for this job {job_details_str}.\n"
  interview_prompt += f"Here is the interview: {interview_text}\n"
  interview_prompt += f"Summarize the interview in less than 3 sentences. Focus on relevant information to make a decision about the candidate and how well the interview went: \n"

  summary = LLM_single_request(interview_prompt, llm_param)
  return summary

def run_gpt_prompts_candidate_impressions(hiring_persona, employee_persona, relationship, interview_summary, perceived_features, job_details_str):

  ## Determines based on ISS if someone can hire others (e.g., business owner, manager)
  llm_param = {"max_new_tokens": 250, "temperature": 0.01, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
    }
  
  ISS = ""
  ISS += f"-Age: {hiring_persona.scratch.age}\n"
  ISS += f"-Personality: {hiring_persona.scratch.innate}\n"
  if hiring_persona.scratch.group_condition in [1,2,4,6]:
    ISS += f"Group Identity: {hiring_persona.scratch.group_identity_text}\n"
  if hiring_persona.scratch.group_condition in [1,2,3,4,6]:
    ISS += f"Additional information: {hiring_persona.scratch.threat_text}\n"
  
  interview_prompt = f"You are {hiring_persona.scratch.name}. " # information about self
  interview_prompt += f"Here is some information about yourself:\n"
  interview_prompt += f"{ISS}\n"
  
  impressions_prompt = f"You have interviewed {employee_persona.scratch.name} for this role: {job_details_str}.\n"
  impressions_prompt += f"Here is a relevant summary of the interview: {interview_summary}\n"
  impressions_prompt += f"Here is a summary of your overall relationship with {employee_persona.scratch.name}: {relationship}.\n"
  impressions_prompt += f"Here are some things you perceive of {employee_persona.scratch.name}: {perceived_features}.\n"
  impressions_prompt += f"Summarize your impression of {employee_persona.scratch.name} as a potential hire. Write 3 or less sentences:\n"

  impressions = LLM_single_request(impressions_prompt, llm_param)

################# Social ratings: Test offline first before server!
  social_ratings_prompt = f"You have interviewed {employee_persona.scratch.name} for this role: {job_details_str}.\n"
  social_ratings_prompt += f"Here is a relevant summary of the interview: {interview_summary}\n"
  social_ratings_prompt += f"Here is a summary of your overall relationship with {employee_persona.scratch.name}: {relationship}.\n"
  social_ratings_prompt += f"Here are some things you perceive of {employee_persona.scratch.name}: {perceived_features}.\n"
  social_ratings_prompt += """
  Please provide a JSON response rating the following dimensions for this candidate on a scale of 1 (extremely low) to 10 (extremely high):

  {
      "likability": "Rate how likeable and approachable the candidate is",
      "competence": "Assess the candidate's perceived professional capability",
      "trustworthiness": "Evaluate the candidate's perceived integrity and reliability", 
      "cultural_fit": "Measure how well the candidate aligns with organizational culture",
      "communication_effectiveness": "Rate the candidate's ability to communicate clearly and persuasively"
  }

  Respond ONLY with a JSON object with these exact keys, using numerical ratings from 1-10.
  """

  fail_safe_response = {"likability": 5, "competence": 5, "trustworthiness": 5, "cultural_fit": 5, "communication_effectiveness": 5}

  def __func_clean_up(social_ratings):
    social_ratings_list = []
    default_rating = 5

    # Define expected keys and order
    expected_keys = [
          "likability", 
          "competence", 
          "trustworthiness", 
          "cultural_fit", 
          "communication_effectiveness"
    ]

    for key in expected_keys:
      value = social_ratings.get(key, default_rating)  # Get value or default

      # Ensure the value is numeric and within range, otherwise use default
      if isinstance(value, (int, float)) and 1 <= value <= 10:
          social_ratings_list.append(value)
      else:
          print(f"Warning: Invalid or missing rating for '{key}', using default ({default_rating}).")
          social_ratings_list.append(default_rating)
    return social_ratings_list

  def __func_validate(output):
    try:
      __func_clean_up(output)
      return True
    except:
      print("falided validating social ratings")
      return False
  
  social_ratings_list = LLM_safe_generate_response_OLD(social_ratings_prompt, 3, fail_safe_response, __func_validate, __func_clean_up, llm_param, 0)

  ##################

  ratings_prompt = f"You have interviewed {employee_persona.scratch.name} for this role: {job_details_str}.\n"
  ratings_prompt += f"Here is a relevant summary of the interview: {interview_summary}\n"
  ratings_prompt += f"Here is a summary of your overall relationship with {employee_persona.scratch.name}: {relationship}.\n"
  ratings_prompt += f"Here are some things you perceive of {employee_persona.scratch.name}: {perceived_features}.\n"
  ratings_prompt += f"Here is your impression of {employee_persona.scratch.name} as a potential employee: {impressions}.\n"

  ### Change: 
    ### use objective metrics from org psych
    ### get additional impressions for social ratings: general liking/sympathy, ...
  ratings_prompt += f"Rate how good of an employee {employee_persona.scratch.name} would be, from 1 (unacceptable) to 10 (exceptional).\n"
  ratings_prompt += f"Return only a number between 1 and 10:\n"

  rating = LLM_single_request(ratings_prompt, llm_param)
  rating_nr = re.search(r'\d+', rating).group()
  try:
    if not rating_nr:
      rating_nr = 5
    rating_nr = int(rating_nr)
  except:
    rating_nr = 5
  return impressions, rating_nr, interview_summary, social_ratings_list

def run_gpt_prompt_consider_offering(hiring_persona, employee_persona, relationship, impressions, rating, interview_summary, job_details_str, perceived_features, social_ratings):

  ## Determines based on ISS if someone can hire others (e.g., business owner, manager)
  llm_param = {"max_new_tokens": 75, "temperature": 0.01, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
    }
  
  
  ISS = ""
  ISS += f"-Age: {hiring_persona.scratch.age}\n"
  ISS += f"-Personality: {hiring_persona.scratch.innate}\n"
  if employee_persona.scratch.group_condition in [1,2,4,6]:
    ISS += f"Group Identity: {hiring_persona.scratch.group_identity_text}\n"
  if employee_persona.scratch.group_condition in [1,2,3,4,6]:
    ISS += f"Additional information: {hiring_persona.scratch.threat_text}\n"
  
  interview_prompt  = f"You are {hiring_persona.scratch.name}. " # information about self
  interview_prompt += f"Here is some information about yourself:\n"
  interview_prompt += f"{ISS}\n"
  
  interview_prompt = f"You have interviewed {employee_persona.scratch.name} for this job: {job_details_str}.\n"
  interview_prompt += f"Decide if you want to make them a job offer.\n"
  interview_prompt += f"Here is a relevant summary of the interview: {interview_summary}\n" ## replace with interview summary?
  interview_prompt += f"Here is a summary of your overall relationship with the potential hire, {employee_persona.scratch.name}: {relationship}\n"
  interview_prompt += f"Here are some things you perceive of {employee_persona.scratch.name}: {perceived_features}\n"
  interview_prompt += f"Here is your impression of {employee_persona.scratch.name} as a potential hire: {impressions}\n"
  interview_prompt += f"This is your overall rating of {employee_persona.scratch.name} as a potential hire: {rating} on a scale from 1-10\n"
  interview_prompt += f"Do you want to make {employee_persona.scratch.name} an offer? Respond with 'yes' or 'no' (<fill in explanation why>). Add the explnation in parenthesis after your decision.\n"

  output = LLM_single_request(interview_prompt, llm_param)

  if "yes" in output.lower():
    offer_job = True
  else:
    offer_job = False

  return offer_job, interview_summary, impressions, rating, employee_persona, relationship, social_ratings

def run_gpt_prompt_employee_decision(hiring_persona, employee_persona, relationship, interview_text, perceived_features, job_details):

  job_details_str = ""
  for detail in hiring_persona.scratch.job_details:
    info = hiring_persona.scratch.job_details[detail]
    job_details_str += f"{detail}: {info}"

  ## Determines based on ISS if someone can hire others (e.g., business owner, manager)
  llm_param = {"max_new_tokens": 50, "temperature": 0.01, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
      "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
      "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
      "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
      "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
      #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
    }
  
  ISS = ""
  ISS += f"-Age: {employee_persona.scratch.age}\n"
  ISS += f"-Personality: {employee_persona.scratch.innate}\n"
  if employee_persona.scratch.group_condition in [1,2,4,6]:
    ISS += f"Group Identity: {employee_persona.scratch.group_identity_text}\n"
  if employee_persona.scratch.group_condition in [1,2,3,4,6]:
    ISS += f"Additional information: {employee_persona.scratch.threat_text}\n"
  
  interview_prompt = f"You are {employee_persona.scratch.name}. " # information about self
  interview_prompt += f"Here is some information about yourself:\n"
  interview_prompt += f"{ISS}\n"

  interview_prompt  = f"Decide if you want to accept the following job offer.\n"
  interview_prompt += f"Currently, you are: {employee_persona.scratch.learned}. The job might require to change that.\n"
  interview_prompt += f"Here are the offered job's details: {job_details_str}\n"
  interview_prompt += f"Here is a summary of your overall relationship with your potential boss, {hiring_persona.scratch.name}: {relationship}\n"
  interview_prompt += f"Here is your interview with {hiring_persona.scratch.name}: {interview_text}\n"
  interview_prompt += f"Here are some things you perceive of {hiring_persona.scratch.name}: {perceived_features}\n"
  interview_prompt += f"Do you want accept this job offered by {hiring_persona.scratch.name}? Respond with 'yes' or 'no' (<fill in explanation why>). Add the explnation in parenthesis after your decision.\n"

  output = LLM_single_request(interview_prompt, llm_param)

  if "yes" in output.lower():
    accept_job = True
  else:
    accept_job = False

  return accept_job 


def run_gpt_prompt_hiring_choice(hiring_persona):

  ## prompt a comparison
  ## potentially add info about own group membership here
  ISS = ""
  ISS += f"-Age: {hiring_persona.scratch.age}\n"
  ISS += f"-Personality: {hiring_persona.scratch.innate}\n"
  if hiring_persona.scratch.group_condition in [1,2,4,6]:
    ISS += f"Group Identity: {hiring_persona.scratch.group_identity_text}\n"
  if hiring_persona.scratch.group_condition in [1,2,3,4,6]:
    ISS += f"Additional information: {hiring_persona.scratch.threat_text}\n"
  
  interview_prompt = f"You are {hiring_persona.scratch.name}.\n" # information about self
  interview_prompt += f"Here is some information about yourself:\n"
  interview_prompt += f"{ISS}\n"
  
  hire_prompt  = f"You have interviewed {len(hiring_persona.scratch.offer_info)} candidates for a job.\n"
  hire_prompt += f"Read the information about each candidate and then choose the best one!\n"

  for i, candidate_name in enumerate(hiring_persona.scratch.offer_info):
    interview_gist, impressions, rating, _, employee_persona, _, _, = hiring_persona.scratch.offer_info[candidate_name]

    # focal_points = [f"{employee_persona.scratch.name}"]
    # retrieved = new_retrieve(hiring_persona, focal_points, 50)
    # relationship = generate_summarize_agent_relationship(hiring_persona, employee_persona, retrieved)
    perceived_features = ""
    for feature in employee_persona.scratch.features:
      if not feature[0] == "Group Identity":
        if isinstance(feature[1][0], (int, float)):
            perceived_features += f"{feature[0]}: {feature[1][0]} on a scale from {feature[1][1][0]} to {feature[1][1][1]}\n"
        else:
            perceived_features += f"{feature[0]}: {feature[1][0]}\n"
      else:
        if employee_persona.scratch.group_condition in [1,2,4,6]:
          perceived_features += f"Group Identity: {feature[1][0]}\n"
    
    ## Determines based on ISS if someone can hire others (e.g., business owner, manager)
    llm_param = {"max_new_tokens": 50, "temperature": 0.01, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
        "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
        "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
        "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
        "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
        #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
      }
    
    hire_prompt += f"Candidate {i+1}: {candidate_name}\n"
    hire_prompt += f"Relevant job interview summary: {interview_gist}\n"
    hire_prompt += f"Some perceived features: {perceived_features}\n"
    hire_prompt += f"Impressions as a potential hire: {impressions}\n"
    hire_prompt += f"Overall rating as a potential hire: {rating}\n\n"

  
  hire_prompt += f"Which candidate do you want to hire? Respond only with the number of the candidate!"
  output = LLM_single_request(hire_prompt, llm_param)
  choice = re.search(r'\d+', output).group()

  try:
    print("TEST WHO GOT HIRED. CANDIDATE: ", choice)
    if not choice:
      choice = 1
    choice = int(choice)
  except:
    choice = 1

  return choice


def run_gpt_prompt_reflect_convo(init_persona, target_persona, previous_convo):
    """
    Constructs the prompt for reflecting on the conversation and determines if either participant contains hateful or aggressive language.

    Args:
        init_persona: The initial persona in the conversation.
        target_persona: The target persona in the conversation.
        relationship: A summary of the relationship between init_persona and target_persona.
        previous_convo: The previous conversation as a string.
        job_details_str: A string describing the job details.

    Returns:
        dict: A dictionary indicating hate/aggression presence for each participant, e.g., {"init": True, "target": False}
    """
    llm_param = {
        "max_new_tokens": 300,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "do_sample": False,
        "seed": 42,
        "stream": False,
        "stop_strings": ["\n"]
    }
    
    # Persona Information Summary (ISS)   
    prompt = f"You are an objective hatespeech and agression analyzer assessing a conversation between two individuals: {init_persona.scratch.name} (init) and {target_persona.scratch.name} (target).\n"
    prompt += f"Conversation Transcript:\n{previous_convo}\n\n"
    prompt += f"Analyze the full transcript. If {init_persona.scratch.first_name} expressed hateful or aggressive language return 'init': True. If {target_persona.scratch.first_name} expressed any hateful or aggressive language return 'target': True.\n"
    prompt += "Respond with a JSON object in the following format: {\"init\": True/False, \"target\": True/False}. Ensure that the JSON is properly formatted."

    output = LLM_single_request(prompt, llm_param)
    
    # Attempt to parse the JSON output
    try:
        import json
        result = json.loads(output)
        # Ensure keys exist and are boolean
        init_result = bool(result.get("init", False))
        target_result = bool(result.get("target", False))
        return {"init": init_result, "target": target_result}
    except json.JSONDecodeError:
        # Handle parsing error, default to False
        return {"init": "", "target": ""}
    except Exception as e:
        # Handle other potential errors
        return {"init": "", "target": ""}

# modify for decomp
def run_gpt_prompt_reflect_action(init_persona, target_persona, actions_str):
    """
    Constructs the prompt for reflecting on the action schedule and determines if hate/aggression is present.

    Args:
        init_persona: The initial persona related to the actions.
        target_persona: The target persona related to the actions.
        relationship: A summary of the relationship between init_persona and target_persona.
        actions_str: The action schedule as a string.
        job_details_str: A string describing the job details (if relevant).

    Returns:
        dict: {"init": True/False, "target": True/False}
    """
    llm_param = {
        "max_new_tokens": 3,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "seed": 42,
        "stream": False,
        "stop_strings": None
    }
    
    if target_persona:
      # prompt = (
      #     f"Analyze the following set of actions or schedule related to {init_persona.scratch.name} (init) and {target_persona.scratch.name} (target).\n\n\
      #     Determine whether any action reflects hate or aggression (or intent).\n\
      #     Respond with a JSON object in the following format:\n\
      #     {{'init': true/false, 'target': true/false}}\n\
      #     Ensure that the JSON is properly formatted.\n\
      #     Here is the action:\n{actions_str}"
      # )
      prompt = f"""Analyze the following set of actions between a member of {init_persona.scratch.group_identity} and a member of {target_persona.scratch.group_identity} and determine whether any action is hateful, violent, or hurtful (e.g., attacking, insulting, harrassing, trolling, discriminating others).\n
              Here are the actions: {actions_str}
              Respond with 'yes' or 'no'\n"""
    else:
      prompt = f"""Analyze the following set of actions of a member of {init_persona.scratch.group_identity} and determine whether any action is hateful, violent, or hurtful (e.g., attacking, insulting, harrassing, trolling, discriminating others).\n
    Here are the actions: {actions_str}
    Respond with 'yes' or 'no'
    Important: Respond only with one word."""

      # prompt = (
      #     f"Analyze the following set of actions or schedule related to {init_persona.scratch.name}.\n\n\
      #     Determine whether any action reflects hate or aggression (or intent).\n\
      #     Respond with a JSON object in the following format:\n\
      #     {{'init': true/false, 'target': true/false}}\n\
      #     Ensure that the JSON is properly formatted.\n\
      #     Here is the action:\n{actions_str}"
      # )
    
    output = LLM_single_request(prompt, llm_param)
    
    # Attempt to parse the JSON output
    # try:
    #     import json
    #     result = json.loads(output)
    #     # Ensure keys exist and are boolean
    #     init_result = bool(result.get("init", False))
    #     target_result = bool(result.get("target", False))
    #     return {"init": init_result, "target": target_result}
    # except json.JSONDecodeError:
    #     # Handle parsing error, default to False
    #     return {"init": False, "target": False}
    # except Exception as e:
    #     # Handle other potential errors
    #     return {"init": False, "target": False}
    try:
      if "yes" in output.lower():
        return True
      else:
        return False
    except:
      print ("OUTPUT ERROR IN ACTION REFLECTION!")

def run_gpt_prompt_reflect_groups(init_persona, actions_str):
    """
    Constructs the prompt for reflecting on the action schedule and determines if hate/aggression is present.

    Args:
        init_persona: The initial persona related to the actions.
        target_persona: The target persona related to the actions.
        relationship: A summary of the relationship between init_persona and target_persona.
        actions_str: The action schedule as a string.
        job_details_str: A string describing the job details (if relevant).

    Returns:
        dict: {"init": True/False, "target": True/False}
    """
    llm_param = {
        "max_new_tokens": 50,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "seed": 42,
        "stream": False,
        "stop_strings": None
    }
    
    prompt = f"""Analyze the following set of actions of a member of {init_persona.scratch.group_identity}.\n\n
        First, determine whether any of the actions listed reflects an interaction between two or more people.\n
        If there are no clear indications, respond with false.\n
        Then, determine if the action is between people from different groups or the same group.\n
        If there are no clear indications, respond with unknown.\n
        Here is the action description: {actions_str}\n
        Respond with a JSON object in the following format:\n
        {{'interaction': true/false, 'intergroup': true/false/unknown}}\n
        Ensure that the JSON is properly formatted.\n"""

    
    output = LLM_single_request(prompt, llm_param)
    
    # Attempt to parse the JSON output
    try:
        import json
        result = json.loads(output)
        # Ensure keys exist and are boolean
        init_result = result.get("interaction", False)
        target_result = result.get("intergroup", "unknown")
        return {"interaction": init_result, "intergroup": target_result}
    except json.JSONDecodeError:
        # Handle parsing error, default to False
        return {"interaction": False, "intergroup": "unknown"}
    except Exception as e:
        # Handle other potential errors
        return {"interaction": False, "intergroup": "unknown"}
    
#### Add trustworthy and cooperation DVs / can be separate prompt within this function, then add it to the list as items at the end!
def run_gpt_robustness_check(persona, context, case): ## need to check persona consistency when having conversations and when making decisions
  if case == "hiring":
    hiring_persona = persona
    employee_persona, job_details_str, interview_summary, relationship, empl_impressions, empl_rating = context
    perceived_features = ""
    for feature in employee_persona.scratch.features:
      if not feature[0] == "Group Identity":
        if isinstance(feature[1][0], (int, float)):
            perceived_features += f"{feature[0]}: {feature[1][0]} on a scale from {feature[1][1][0]} to {feature[1][1][1]}\n"
        else:
            perceived_features += f"{feature[0]}: {feature[1][0]}\n"
      else:
        if employee_persona.scratch.group_condition in [1,2,4,6]:
          perceived_features += f"Group Identity: {feature[1][0]}\n"


    ## Determines based on ISS if someone can hire others (e.g., business owner, manager)
    llm_param = {"max_new_tokens": 250, "temperature": 0.01, "top_p": 1, "min_p": 0.1, "top_k": 40, "repetition_penalty": 1.15, 
        "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
        "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
        "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
        "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
        #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
      }
    
    ISS = ""
    ISS += f"-Age: {hiring_persona.scratch.age}\n"
    ISS += f"-Personality: {hiring_persona.scratch.innate}\n"
    if employee_persona.scratch.group_condition in [1,2,4,6]:
      ISS += f"Group Identity: {hiring_persona.scratch.group_identity_text}\n"
    if employee_persona.scratch.group_condition in [1,2,3,4,6]:
      ISS += f"Additional information: {hiring_persona.scratch.threat_text}\n"
    
    prompt_context  = f"You are {hiring_persona.scratch.name}. " # information about self
    prompt_context += f"Here is some information about yourself:\n"
    prompt_context += f"{ISS}\n"
    
    prompt_context = f"You have interviewed {employee_persona.scratch.name} for this job: {job_details_str}.\n"
    prompt_context += f"Decide if you want to make them a job offer.\n"
    prompt_context += f"Here is a relevant summary of the interview: {interview_summary}\n" ## replace with interview summary?
    prompt_context += f"Here is a summary of your overall relationship with the potential hire, {employee_persona.scratch.name}: {relationship}\n"
    prompt_context += f"Here are some things you perceive of {employee_persona.scratch.name}: {perceived_features}\n"
    prompt_context += f"Here is your impression of {employee_persona.scratch.name} as a potential hire: {empl_impressions}\n"
    prompt_context += f"This is your overall rating of {employee_persona.scratch.name} as a potential hire: {empl_rating} on a scale from 1-10\n"

  elif case == "convo":
    maze, target_persona, retrieved, curr_chat = context
      # Chat version optimized for speed via batch generation
    curr_context = (f"{persona.scratch.name} " + 
                f"was {persona.scratch.act_description} " + 
                f"when {persona.scratch.name} " + 
                f"saw {target_persona.scratch.name} " + 
                f"in the middle of {target_persona.scratch.act_description}.\n")
    curr_context += (f"{persona.scratch.name} " +
                f"is initiating a conversation with " +
                f"{target_persona.scratch.name}.")

    def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None):
      persona = init_persona
      prev_convo_insert = "\n"
      if persona.a_mem.seq_chat: 
        for i in persona.a_mem.seq_chat: 
          if i.object == target_persona.scratch.name: 
            v1 = int((persona.scratch.curr_time - i.created).total_seconds()/60)
            prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description}. This context takes place after that conversation.'
            break
      if prev_convo_insert == "\n": 
        prev_convo_insert = ""
      if persona.a_mem.seq_chat: 
        if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
          prev_convo_insert = ""

      curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
      curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
      curr_location = f"{curr_arena} in {curr_sector}"

      retrieved_str = ""
      for key, vals in retrieved.items(): 
        for v in vals: 
          retrieved_str += f"- {v.description}\n"


      convo_str = ""
      for i in curr_chat:
        convo_str += ": ".join(i) + "\n"
      if convo_str == "": 
        convo_str = "\nThe conversation has not started yet -- start it!"

      ISS = ""
      ISS += f"-Age: {init_persona.scratch.age}\n"
      ISS += f"-Personality: {init_persona.scratch.innate}\n"
      ISS += f"-Short biography: {init_persona.scratch.learned}\n"
      ISS += f"-Living context: {init_persona.scratch.currently}\n" # summary about self
      if init_persona.scratch.group_condition in [1,2,4,6]:
        ISS += f"Group Identity: {init_persona.scratch.group_identity_text}\n"
      if init_persona.scratch.group_condition in [1,2,3,4,6]:
        ISS += f"Additional information: {init_persona.scratch.threat_text}\n"

      init_iss = f"You are {init_persona.scratch.name}. Here is some information about your personality, biography, and living context:\n{ISS}\n"
      # init_iss = f"Here is a brief description of you, {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"

      # Do I need to take short biography out? And personality? Is this too much information?
      init_iss_target  = f"You are talking to {target_persona.scratch.name}. Here is a brief description of them:\n"
      init_iss_target += f"-Age: {target_persona.scratch.age}\n"
      init_iss_target += f"-Personality: {target_persona.scratch.innate}\n"
      init_iss_target += f"-Short biography: {target_persona.scratch.learned}\n"
      
      perceived_features = f"Here are some things you perceive of {target_persona.name}:\n"
      for feature in target_persona.scratch.features:
        if not feature[0] == "Group Identity":
          if isinstance(feature[1][0], (int, float)):
              perceived_features += f"{feature[0]}: {feature[1][0]} on a scale from {feature[1][1][0]} to {feature[1][1][1]}\n"
          else:
              perceived_features += f"{feature[0]}: {feature[1][0]}\n"
        else:
          if target_persona.scratch.group_condition in [1,2,4,6]:
            perceived_features += f"Group Identity: {feature[1][0]}\n"      
          
      prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
        curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
        convo_str, init_persona.scratch.name, target_persona.scratch.name,
        init_persona.scratch.name, init_persona.scratch.name,
        init_persona.scratch.name, perceived_features, init_iss_target
        ]
      return prompt_input

    prompt_template = "persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt" 
    prompt_input = create_prompt_input(maze, persona, target_persona, retrieved, curr_context, curr_chat) 
    prompt_context = generate_prompt(prompt_input, prompt_template).split("---\nTask")[0]

  elif case == "action":
    def create_prompt_input(persona):
        
      curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
      all_indices = []
      all_indices += [curr_f_org_index]
      if curr_f_org_index+1 <= len(persona.scratch.f_daily_schedule_hourly_org): 
        all_indices += [curr_f_org_index+1]
      if curr_f_org_index+2 <= len(persona.scratch.f_daily_schedule_hourly_org): 
        all_indices += [curr_f_org_index+2]

      summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. '
      summ_str += f'From '
      for index in all_indices: 
        if index < len(persona.scratch.f_daily_schedule_hourly_org): 
          start_min = 0
          for i in range(index): 
            start_min += persona.scratch.f_daily_schedule_hourly_org[i][1]
          end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
          start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") 
                        + datetime.timedelta(minutes=start_min)) 
          end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") 
                        + datetime.timedelta(minutes=end_min)) 
          start_time_str = start_time.strftime("%H:%M%p")
          end_time_str = end_time.strftime("%H:%M%p")
          summ_str += f"{start_time_str} ~ {end_time_str}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
      summ_str = summ_str[:-2] + "."

      prompt_input = []
      prompt_input += [persona.scratch.get_str_iss()]
      prompt_input += [summ_str]
      return prompt_input
      
    prompt_template = "persona/prompt_template/v2/task_decomp_v3.txt"
    prompt_input = create_prompt_input(persona)
    prompt_context = "".join(generate_prompt(prompt_input, prompt_template).split("In 5 minute increments,")[:-1])

  else:
    prompt_context = ""
  
  prompt = prompt_context + """
  ---
  Task: Given your background and the context you are in, how much do you agree with the following statements?

  Statements:
  1) I feel a strong bond with <Group 1>.
  2) Being a <Group 1> is an important reflection of who I am.
  3) I strongly identify with <Group 1>'s values.
  4) I feel committed to <Group 1>.
  5) I am willing to make personal sacrifices to support <Group 1> 
  6) Members of my group (<Group 1>) are more moral than members of the other group (<Group 2>).
  7) Members of my group (<Group 1>) are more trustworthy than members of the other group (<Group 2>).
  8) Members of my group (<Group 1>) are more competent than members of the other group (<Group 2>).
  9) Members of my group (<Group 1>) are more kind than members of the other group (<Group 2>).
  10) <Group 1> values are threatened by <Group 2>.
  11) <Group 1> traditions are threatened by <Group 2>.
  12) The physical safety of <Group 1> members is threatened by <Group 2>.
  13) The financial well-being of <Group 1> members is threatened by <Group 2>.

  Instructions:
  - Rate the extent that you agree with the statements using an integer from 1 (not at all) to 4 (neutral) to 7 (totally).
  - Do not make up facts that are not explicitly stated above (e.g., whether or not there is economic disparity between our groups), but subjective opinions are ok. 
  - Take the provided background and context as is. If there is no information related to statement rate accordingly (e.g. "1"). 
  - Focus on the situation described under "Context", if provided.

  Output format: Output a json of the following format: 
  {
  "1": "<Agreement from 1 (not at all) to 4 (neutral) to 7 (totally)> | <Single sentence explaining why>",
  "2": "<Agreement from 1 (not at all) to 4 (neutral) to 7 (totally)> | <Single sentence explaining why>",
  ...
  "N": "<Agreement from 1 (not at all) to 4 (neutral) to 7 (totally)> | <Single sentence explaining why>"
  }"""

  if persona.scratch.group_condition != 5:
    group = persona.scratch.group_identity
    if group == "Group A":
      outgroup = "Group B"
    else:
      outgroup = "Group A"
    prompt = prompt.replace("<Group 1>", f"{group}")\
                                    .replace("<Group 2>", f"{outgroup}")

  def __chat_func_clean_up(gpt_response, prompt=""): 
    response_list = []

    for q_nr, response in gpt_response.items():
      numeric, written = response.split(" | ") if " | " in response else (response, "NA")
      response_entry = {
              "item_nr": int(q_nr),  # Convert key to integer
              "numeric": int(numeric),  # Convert numeric rating to integer
              "written": written.strip()
          }
      response_list.append(response_entry)

    return response_list

  def __chat_func_validate(gpt_response, prompt=""): 
    print ("test robustness...")
    try: 
      print ("try formatting robustness check", __chat_func_clean_up(gpt_response))
      return True
    except:
      print("FAILED ROBUSTNESS CHECK")
      return False

  def get_fail_safe():
    cleaned_dict = {
      "1": "0 | NA",
      "2": "0 | NA",
      "3": "0 | NA",
      "4": "0 | NA",
      "5": "0 | NA",
      "6": "0 | NA",
      "7": "0 | NA",
      "8": "0 | NA",
      "9": "0 | NA",
      "10": "0 | NA",
      "11": "0 | NA",
      "12": "0 | NA",
      "13": "0 | NA"
    }
    return cleaned_dict

  llm_param = {"max_new_tokens": 500, "temperature": 0.01, "top_p": 1, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
        "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
        "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
        "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
        "min_length": 0, "no_repeat_ngram_size": 0, "stream": False, "stop_strings": None,
        #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
    }

  # print (prompt)
  fail_safe = get_fail_safe() 
  output = LLM_safe_generate_response_OLD(prompt, 3, fail_safe,
                      __chat_func_validate, __chat_func_clean_up, llm_param, 0)
  return output