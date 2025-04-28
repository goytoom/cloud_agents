"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: converse.py
Description: An extra cognitive module for generating conversations. 
"""
import math
import sys
import datetime
import random
import re
sys.path.append('../')

from global_methods import *

from persona.memory_structures.spatial_memory import *
from persona.memory_structures.associative_memory import *
from persona.memory_structures.scratch import *
from persona.cognitive_modules.retrieve import *
from persona.prompt_template.run_gpt_prompt import *

from tracker_instance import TRACKER  ## tested: Same tracker active in main file (reverie.py) and here!
# e.g., create an instance in main/reverie, save something, load here and display logs

def generate_agent_chat_summarize_ideas(init_persona, 
                                        target_persona, 
                                        retrieved, 
                                        curr_context): 
  all_embedding_keys = list()
  for key, val in retrieved.items(): 
    for i in val: 
      all_embedding_keys += [i.embedding_key]
  all_embedding_key_str =""
  for i in all_embedding_keys: 
    all_embedding_key_str += f"{i}\n"

  try: 
    summarized_idea = run_gpt_prompt_agent_chat_summarize_ideas(init_persona,
                        target_persona, all_embedding_key_str, 
                        curr_context)[0]
  except:
    summarized_idea = ""
  return summarized_idea


def generate_summarize_agent_relationship(init_persona, 
                                          target_persona, 
                                          retrieved): 
  all_embedding_keys = list()
  for key, val in retrieved.items(): 
    for i in val: 
      all_embedding_keys += [i.embedding_key]
  all_embedding_key_str =""
  for i in all_embedding_keys: 
    all_embedding_key_str += f"{i}\n"

  summarized_relationship = run_gpt_prompt_agent_chat_summarize_relationship(
                              init_persona, target_persona,
                              all_embedding_key_str)[0]
  return summarized_relationship


def generate_agent_chat(maze, 
                        init_persona, 
                        target_persona,
                        curr_context, 
                        init_summ_idea, 
                        target_summ_idea): 
  summarized_idea = run_gpt_prompt_agent_chat(maze, 
                                              init_persona, 
                                              target_persona,
                                              curr_context, 
                                              init_summ_idea, 
                                              target_summ_idea)[0]
  # for i in summarized_idea: 
  #   print (i)
  return summarized_idea


def agent_chat_v1(maze, init_persona, target_persona): 
  # Chat version optimized for speed via batch generation
  curr_context = (f"{init_persona.scratch.name} " + 
              f"was {init_persona.scratch.act_description} " + 
              f"when {init_persona.scratch.name} " + 
              f"saw {target_persona.scratch.name} " + 
              f"in the middle of {target_persona.scratch.act_description}.\n")
  curr_context += (f"{init_persona.scratch.name} " +
              f"is thinking of initating a conversation with " +
              f"{target_persona.scratch.name}.")
  
  ### Add Background of target persona! (some conversation are unrealistic, e.g., why would isabella have access to college stuff?)
    ## From you perspective? If add information about oneself (might already be there in 3rd person => keep 3rd person)
  ### Maybe add to template to have realistic conversations ("what would you say in this situation?")

  summarized_ideas = []
  part_pairs = [(init_persona, target_persona), 
                (target_persona, init_persona)]
  for p_1, p_2 in part_pairs: 
    focal_points = [f"{p_2.scratch.name}"]
    retrieved = new_retrieve(p_1, focal_points, 50)
    relationship = generate_summarize_agent_relationship(p_1, p_2, retrieved)
    focal_points = [f"{relationship}", 
                    f"{p_2.scratch.name} is {p_2.scratch.act_description}"]
    retrieved = new_retrieve(p_1, focal_points, 25)
    summarized_idea = generate_agent_chat_summarize_ideas(p_1, p_2, retrieved, curr_context)
    summarized_ideas += [summarized_idea]

  return generate_agent_chat(maze, init_persona, target_persona, 
                      curr_context, 
                      summarized_ideas[0], 
                      summarized_ideas[1])


def generate_one_utterance(maze, init_persona, target_persona, retrieved, curr_chat): 
  # Chat version optimized for speed via batch generation
  curr_context = (f"{init_persona.scratch.name} " + 
              f"was {init_persona.scratch.act_description} " + 
              f"when {init_persona.scratch.name} " + 
              f"saw {target_persona.scratch.name} " + 
              f"in the middle of {target_persona.scratch.act_description}.\n")
  curr_context += (f"{init_persona.scratch.name} " +
              f"is initiating a conversation with " +
              f"{target_persona.scratch.name}.")
  
  ### Add Background of target persona! (some conversation are unrealistic, e.g., why would isabella have access to college stuff?)
    ## From you perspective? If add information about oneself (might already be there in 3rd person => keep 3rd person)
  ### Maybe add to template to have realistic conversations ("what would you say in this situation?")

  print ("July 23 5")
  x = run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat)[0]

  print ("July 23 6")

  print ("adshfoa;khdf;fajslkfjald;sdfa HERE", x)

  return x["utterance"], x["end"]


### Log relevant convos
  ### Add reflection / classification of violence to the logs
  ### Log convos at 3 locations: pub, cafe, supermarket
  ### log approach/avoid decisions
def agent_chat_v2(maze, init_persona, target_persona): 
  curr_chat = []
  print ("July 23")

  for i in range(8): 
    focal_points = [f"{target_persona.scratch.name}"]
    retrieved = new_retrieve(init_persona, focal_points, 50)
    relationship = generate_summarize_agent_relationship(init_persona, target_persona, retrieved)
    print ("-------- relationshopadsjfhkalsdjf", relationship)
    last_chat = ""
    for i in curr_chat[-4:]:
      last_chat += ": ".join(i) + "\n"
    if last_chat: 
      focal_points = [f"{relationship}", 
                      f"{target_persona.scratch.name} is {target_persona.scratch.act_description}", 
                      last_chat]
    else: 
      focal_points = [f"{relationship}", 
                      f"{target_persona.scratch.name} is {target_persona.scratch.act_description}"]
    retrieved = new_retrieve(init_persona, focal_points, 15)
    utt, end = generate_one_utterance(maze, init_persona, target_persona, retrieved, curr_chat)

    curr_chat += [[init_persona.scratch.name, utt]]
    if end:
      break

    focal_points = [f"{init_persona.scratch.name}"]
    retrieved = new_retrieve(target_persona, focal_points, 50)
    relationship = generate_summarize_agent_relationship(target_persona, init_persona, retrieved)
    print ("-------- relationshopadsjfhkalsdjf", relationship)
    last_chat = ""
    for i in curr_chat[-4:]:
      last_chat += ": ".join(i) + "\n"
    if last_chat: 
      focal_points = [f"{relationship}", 
                      f"{init_persona.scratch.name} is {init_persona.scratch.act_description}", 
                      last_chat]
    else: 
      focal_points = [f"{relationship}", 
                      f"{init_persona.scratch.name} is {init_persona.scratch.act_description}"]
    retrieved = new_retrieve(target_persona, focal_points, 15)
    utt, end = generate_one_utterance(maze, target_persona, init_persona, retrieved, curr_chat)

    curr_chat += [[target_persona.scratch.name, utt]]
    if end:
      break


  ########### Functions for hiring ##############
  def _want_to_interview(hiring_persona, employee_persona, job_details_str, curr_chat):
    focal_points = [f"{employee_persona.scratch.name}"]
    retrieved = new_retrieve(hiring_persona, focal_points, 50)
    relationship = generate_summarize_agent_relationship(hiring_persona, employee_persona, retrieved)
    previous_convo = ""
    for row in curr_chat: 
      speaker = row[0]
      utt = row[1]
      previous_convo += f"{speaker}: {utt}\n"

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
    
    return run_gpt_prompt_decide_interview(hiring_persona, employee_persona, relationship, perceived_features, previous_convo, job_details_str)

  def accept_interview(hiring_persona, employee_persona, job_details_str, curr_chat):
    focal_points = [f"{hiring_persona.scratch.name}"]
    retrieved = new_retrieve(employee_persona, focal_points, 50)
    relationship = generate_summarize_agent_relationship(employee_persona, hiring_persona, retrieved)
    previous_convo = ""
    for row in curr_chat: 
      speaker = row[0]
      utt = row[1]
      previous_convo += f"{speaker}: {utt}\n"

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

    return run_gpt_prompt_accept_interview(hiring_persona, employee_persona, relationship, previous_convo, perceived_features, job_details_str)

  def run_interview(hiring_persona, employee_persona, job_details_str):
    curr_chat = []

    for i in range(8): 
      # interviewer
      focal_points = [f"{employee_persona.scratch.name}"]
      retrieved = new_retrieve(hiring_persona, focal_points, 50)
      relationship = generate_summarize_agent_relationship(hiring_persona, employee_persona, retrieved)
      print ("-------- relationshopadsjfhkalsdjf", relationship)
      last_chat = ""
      for i in curr_chat[-4:]:
        last_chat += ": ".join(i) + "\n"
      if last_chat: 
        focal_points = [f"{relationship}", 
                        f"{employee_persona.scratch.name} is {employee_persona.scratch.act_description}", 
                        last_chat]
      else: 
        focal_points = [f"{relationship}", 
                        f"{employee_persona.scratch.name} is {employee_persona.scratch.act_description}"]
      retrieved = new_retrieve(hiring_persona, focal_points, 15)
      utt, end = run_gpt_interviewer(hiring_persona, employee_persona, retrieved, curr_chat, job_details_str, relationship)

      curr_chat += [[hiring_persona.scratch.name, utt]]
      if end:
        break

      chat_so_far = ""
      for row in curr_chat: 
        speaker = row[0]
        utt = row[1]
        chat_so_far += f"{speaker}: {utt}\n"

      print("Interview so far: ")
      print(chat_so_far)

      # interviewee
      focal_points = [f"{hiring_persona.scratch.name}"]
      retrieved = new_retrieve(employee_persona, focal_points, 50)
      relationship = generate_summarize_agent_relationship(employee_persona, hiring_persona, retrieved)
      print ("-------- relationshopadsjfhkalsdjf", relationship)
      last_chat = ""
      for i in curr_chat[-4:]:
        last_chat += ": ".join(i) + "\n"
      if last_chat: 
        focal_points = [f"{relationship}", 
                        f"{hiring_persona.scratch.name} is {hiring_persona.scratch.act_description}", 
                        last_chat]
      else: 
        focal_points = [f"{relationship}", 
                        f"{hiring_persona.scratch.name} is {hiring_persona.scratch.act_description}"]
      retrieved = new_retrieve(employee_persona, focal_points, 15)
      utt, end = run_gpt_interviewee(hiring_persona, employee_persona, retrieved, curr_chat, job_details_str, relationship)

      curr_chat += [[employee_persona.scratch.name, utt]]
      if end:
        break

      chat_so_far = ""
      for row in curr_chat: 
        speaker = row[0]
        utt = row[1]
        chat_so_far += f"{speaker}: {utt}\n"

      print("Interview so far: ")
      print(chat_so_far)

    return curr_chat

  def employee_decision(hiring_persona, employee_persona, job_details, curr_chat):
    focal_points = [f"{hiring_persona.scratch.name}"]
    retrieved = new_retrieve(employee_persona, focal_points, 50)
    relationship = generate_summarize_agent_relationship(employee_persona, hiring_persona, retrieved)
    interview_text = ""
    for row in curr_chat: 
      speaker = row[0]
      utt = row[1]
      interview_text += f"{speaker}: {utt}\n"

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
    
    return run_gpt_prompt_employee_decision(hiring_persona, employee_persona, relationship, interview_text, perceived_features, job_details)

  def summarize_interview(hiring_persona, employee_persona, job_details_str, interview_chat):
    interview_text = ""
    for row in interview_chat: 
      speaker = row[0]
      utt = row[1]
      interview_text += f"{speaker}: {utt}\n"
    
    return run_gpt_prompt_summarize_interview(hiring_persona, employee_persona, interview_text, job_details_str)

  def candidate_impressions(hiring_persona, employee_persona, job_details_str, interview_chat):
    focal_points = [f"{employee_persona.scratch.name}"]
    retrieved = new_retrieve(hiring_persona, focal_points, 50)
    relationship = generate_summarize_agent_relationship(hiring_persona, employee_persona, retrieved)
    interview_text = ""
    for row in interview_chat: 
      speaker = row[0]
      utt = row[1]
      interview_text += f"{speaker}: {utt}\n"

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
    
    interview_summary = summarize_interview(hiring_persona, employee_persona, job_details_str, interview_chat)

    return run_gpt_prompts_candidate_impressions(hiring_persona, employee_persona, relationship, interview_summary, perceived_features, job_details_str)

  def consider_offering(hiring_persona, employee_persona, job_details_str, interview_chat):
    focal_points = [f"{hiring_persona.scratch.name}"]
    retrieved = new_retrieve(employee_persona, focal_points, 50)
    relationship = generate_summarize_agent_relationship(employee_persona, hiring_persona, retrieved)
    interview_text = ""
    for row in interview_chat: 
      speaker = row[0]
      utt = row[1]
      interview_text += f"{speaker}: {utt}\n"

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
       
    impressions, rating, interview_summary, social_ratings = candidate_impressions(hiring_persona, employee_persona, job_details_str, interview_chat)

    return run_gpt_prompt_consider_offering(hiring_persona, employee_persona, relationship, impressions, rating, interview_summary, job_details_str, perceived_features, social_ratings)

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

  ## after conversation run hiring check: X
  ## add variable for how many interviews/attempts of target X
  if init_persona.scratch.can_hire and target_persona.scratch.can_be_hired:
    hiring_persona = init_persona
    employee_persona = target_persona
  elif target_persona.scratch.can_hire and init_persona.scratch.can_be_hired:
      hiring_persona = target_persona
      employee_persona = init_persona
  else:
      hiring_persona = None
      employee_persona = None

  curr_chat_transcript = ""
  for row in curr_chat: 
    speaker = row[0]
    utt = row[1]
    curr_chat_transcript += f"{speaker}: {utt}\n"


  if hiring_persona: # maybe add outputs for debug (ideally extend frontend to include hiring interviews/status)
    print("DEBUG: HIRING SIMULATION")
    if employee_persona.scratch.name not in hiring_persona.scratch.interact_info:
      hiring_persona.scratch.interact_counter += 1 
      hiring_persona.scratch.interact_info[employee_persona.scratch.name] = "" # only count new people for interaction counter
      logging_info = {"interaction_count": hiring_persona.scratch.interact_counter, "interview_count": hiring_persona.scratch.interview_counter, "Decision_type": "Interacted", "Decision": True, "Init": hiring_persona, "Target": employee_persona, 
                    "Sim_Time": hiring_persona.scratch.curr_time, "Convo": curr_chat_transcript, "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": hiring_persona.scratch.group_condition,
                    "Init Group": hiring_persona.scratch.group_identity, "Target Group": employee_persona.scratch.group_identity, "convo_hate": "", "interview_hate": ""}
      TRACKER.log_decision(logging_info, hiring_persona.scratch.sim_nr)
    if hiring_persona.scratch.interact_counter <= 10:
      if employee_persona.scratch.name not in hiring_persona.scratch.interview_info: # if not offered an interview yet
        if not hiring_persona.scratch.offer_info or hiring_persona.scratch.interview_counter <= 5:  # approach 3 people (continue until at least one wants the job)
          if not hiring_persona.scratch.job_details:
            hiring_persona.scratch.job_details = run_gpt_get_job_details(hiring_persona)
            print("Job Details: ", hiring_persona.scratch.job_details)
          job_details_str = ""
          for detail in hiring_persona.scratch.job_details:
            info = hiring_persona.scratch.job_details[detail]
            job_details_str += f"{detail}: {info}\n"
          print("Job Details: ", job_details_str)
          
          int_status = _want_to_interview(hiring_persona, employee_persona, job_details_str, curr_chat) # check if employer wants to interview
          print("\nInterview candidate?", int_status, "\n")
          ## Save using TRACKER here:
            # repeat after each decision
          logging_info = {"interaction_count": hiring_persona.scratch.interact_counter, "interview_count": hiring_persona.scratch.interview_counter + int(int_status), "Decision_type": "Approach for Interview", "Decision": int_status, "Init": hiring_persona, "Target": employee_persona, 
                              "Sim_Time": hiring_persona.scratch.curr_time, "Convo": curr_chat_transcript, "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": hiring_persona.scratch.group_condition,
                              "Init Group": hiring_persona.scratch.group_identity, "Target Group": employee_persona.scratch.group_identity, "convo_hate": "", "interview_hate": ""}
          TRACKER.log_decision(logging_info, hiring_persona.scratch.sim_nr)

          if int_status:
            hiring_persona.scratch.interview_info[employee_persona.scratch.name] = "" # save that employee was offered an interview -> only offer once
            hiring_persona.scratch.interview_counter += 1 ## counter how many candidates were approached (make a decision after N)
            print("Number of interview approaches so far: ", hiring_persona.scratch.interview_counter)
            
            int_acceptance = accept_interview(hiring_persona, employee_persona, job_details_str, curr_chat) # check if employee wants to interview
            print("Accept interview?", int_acceptance)
            logging_info = {"interaction_count": hiring_persona.scratch.interact_counter, "interview_count": hiring_persona.scratch.interview_counter, "Decision_type": "Accept Interview", "Decision": int_acceptance, "Init": hiring_persona, "Target": employee_persona, 
                              "Sim_Time": hiring_persona.scratch.curr_time, "Convo": curr_chat_transcript, "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": hiring_persona.scratch.group_condition,
                              "Init Group": hiring_persona.scratch.group_identity, "Target Group": employee_persona.scratch.group_identity, "convo_hate": "", "interview_hate": ""}
            TRACKER.log_decision(logging_info, hiring_persona.scratch.sim_nr)
            if int_acceptance:
              interview_text = run_interview(hiring_persona, employee_persona, job_details_str) # run the interview
              print("Interview transcript: ")
              interview_transcript = ""
              for row in interview_text: 
                speaker = row[0]
                utt = row[1]
                interview_transcript += f"{speaker}: {utt}\n"
                ## add hate/aggression reflection of interview 
                ## -> add to logs
              print("Interview Text: ", interview_transcript)

              hate_reflection = reflect_convo(init_persona, target_persona, interview_transcript)
              init_hate_int = hate_reflection["init"]
              target_hate_int = hate_reflection["target"]
              interview_hate = any([init_hate_int, target_hate_int])
              
              job_acceptance = employee_decision(hiring_persona, employee_persona, hiring_persona.scratch.job_details, interview_text) # check if employee wants the job
              print("Employee wants the job? ", job_acceptance)
              logging_info = {"interaction_count": hiring_persona.scratch.interact_counter, "interview_count": hiring_persona.scratch.interview_counter, "Decision_type": "Accept Job", "Decision": job_acceptance, "Init": hiring_persona, "Target": employee_persona, 
                              "Sim_Time": hiring_persona.scratch.curr_time, "Convo": curr_chat_transcript, "Interview": interview_transcript, "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": hiring_persona.scratch.group_condition,
                              "Init Group": hiring_persona.scratch.group_identity, "Target Group": employee_persona.scratch.group_identity, "convo_hate": "", "interview_hate": interview_hate}
              TRACKER.log_decision(logging_info, hiring_persona.scratch.sim_nr)
              if job_acceptance:
                ## 1) Ask BO if they want to offer the job ==> DONE
                ## a) prompt to generate gist of conversation ==> DONE
                ## b) prompt to generate impression of candidate (suitable for job?) ==> DONE
                ## c) prompt to give rating 1-10 (e.g., ideal candidate) ==> DONE
                ## d) save in hiring agent (e.g., self.offer_info) ==> DONE
                consider_offering_job, interview_gist, empl_impressions, empl_rating, employee_persona, relationship, social_ratings = consider_offering(hiring_persona, employee_persona, 
                                                                                                        hiring_persona.scratch.job_details, interview_text) # check if employer wants to offer the job
                print("Interview summary: ", interview_gist)
                print("Interview impression: ", empl_impressions)
                print("Employee rating: ", empl_rating)
                print("Social Ratings: ", social_ratings)

                logging_info = {"interaction_count": hiring_persona.scratch.interact_counter, "interview_count": hiring_persona.scratch.interview_counter, "Decision_type": "Consider offering", "Decision": consider_offering_job, "Init": hiring_persona, "Target": employee_persona, 
                              "Sim_Time": hiring_persona.scratch.curr_time, "Convo": curr_chat_transcript, "Interview": interview_transcript, "target_impression": empl_impressions, "target_rating": empl_rating, "target_rating_soc": social_ratings, "Group Condition": hiring_persona.scratch.group_condition, 
                              "Init Group": hiring_persona.scratch.group_identity, "Target Group": employee_persona.scratch.group_identity, "convo_hate": "", "interview_hate": interview_hate}
                TRACKER.log_decision(logging_info, hiring_persona.scratch.sim_nr)


                ## robustness checks:
                robustness_context = [employee_persona, job_details_str, interview_gist, relationship, empl_impressions, empl_rating] ## add everything needed to construct context
                robustness_case = "hiring" # add different cases to check: Convo, hiring, acting
                robustness_items = run_gpt_robustness_check(hiring_persona, robustness_context, robustness_case) ## add code to construct context, run robustness checks and return

                # then log the outcome
                logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "robustness_hiring", "Decision": robustness_items, "Init": hiring_persona, "Target": employee_persona, 
                              "Sim_Time": hiring_persona.scratch.curr_time, "Convo": "", "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": hiring_persona.scratch.group_condition, 
                              "Init Group": hiring_persona.scratch.group_identity, "Target Group": employee_persona.scratch.group_identity, "convo_hate": "", "interview_hate": ""}
                TRACKER.log_decision(logging_info, hiring_persona.scratch.sim_nr)
                
                print("Boss wants to hire?", consider_offering_job)
                # If yes, save information
                if consider_offering_job:
                  hiring_persona.scratch.offer_info[employee_persona.scratch.name] = [interview_gist, empl_impressions, empl_rating, social_ratings, employee_persona, interview_transcript, interview_hate]

        ## Remove people who have been hired!
        if hiring_persona.scratch.offer_info:
          new_offer_info = {}
          for persona_name, hiring_information in hiring_persona.scratch.offer_info.items():
            _, _, _, _, empl_persona, _, _ = hiring_information
            if empl_persona.scratch.can_be_hired:
              new_offer_info.update({persona_name: hiring_information})

          # Update the offer_info list
          hiring_persona.scratch.offer_info = new_offer_info

        ## check if enough people have been approached/interviewed to make decision
        if hiring_persona.scratch.offer_info and hiring_persona.scratch.interview_counter >= 5:
          ## make decision
          hired_candidate, hired_candidate_persona = _hire(hiring_persona)
          _, empl_impressions, empl_rating, social_ratings, empl_persona, interview_transcript, interview_hate = hiring_persona.scratch.offer_info[hired_candidate_persona.scratch.name]
          print("Hiring Choice: ", hired_candidate)
          hiring_persona.scratch.can_hire = False # turn hiring status off
          hired_candidate_persona.scratch.can_be_hired = False
          logging_info = {"interaction_count": hiring_persona.scratch.interact_counter, "interview_count": hiring_persona.scratch.interview_counter, "Decision_type": "Hiring", "Decision": True, "Init": hiring_persona, "Target": empl_persona, 
                              "Sim_Time": hiring_persona.scratch.curr_time, "Convo": curr_chat_transcript, "Interview": interview_transcript, "target_impression": empl_impressions, "target_rating": empl_rating, "target_rating_soc": social_ratings, "Group Condition": hiring_persona.scratch.group_condition,
                              "Init Group": hiring_persona.scratch.group_identity, "Target Group": empl_persona.scratch.group_identity,
                              "convo_hate": "", "interview_hate": interview_hate}
          TRACKER.log_decision(logging_info, hiring_persona.scratch.sim_nr)
    else: # after interacting with at least 10 people try to hire or give up
      if hiring_persona.scratch.offer_info:
        ## make decision
        hired_candidate, hired_candidate_persona = _hire(hiring_persona)
        _, empl_impressions, empl_rating, social_ratings, empl_persona, interview_transcript, interview_hate = hiring_persona.scratch.offer_info[hired_candidate_persona.scratch.name]
        print("Hiring Choice: ", hired_candidate)
        hiring_persona.scratch.can_hire = False # turn hiring status off
        hired_candidate_persona.scratch.can_be_hired = False
        logging_info = {"interaction_count": hiring_persona.scratch.interact_counter, "interview_count": hiring_persona.scratch.interview_counter, "Decision_type": "Hiring", "Decision": True, "Init": hiring_persona, "Target": empl_persona, 
                      "Sim_Time": hiring_persona.scratch.curr_time, "Convo": curr_chat_transcript, "Interview": interview_transcript, "target_impression": empl_impressions, "target_rating": empl_rating, "target_rating_soc": social_ratings, "Group Condition": hiring_persona.scratch.group_condition,
                      "Init Group": hiring_persona.scratch.group_identity, "Target Group": empl_persona.scratch.group_identity, "convo_hate": "", "interview_hate": interview_hate}
        
      else: # if there's still no one, give up
        hiring_persona.scratch.can_hire = False
                  ############################To Do:############################
                  # Exit simulation after hiring to save time! ==> DONE
                    # Check that everyone has hired (no one has can_hire = True) ==> DONE (Main Loop)
                  # Get distribution of features (Bosses and Employees) ==> Done
                    # Save in results folder -> ==> DONE
                  # Set values for Bosses (instead of random)
                  # Set fixed distribution (e.g., majority/minority group!)
                  ############################To Do:############################

                  ## Save hiring persona information/features ==> Add tracker class, save information at every step ==> DONE
                    ## Decision type (offer interview, reject/take interview) 
                    ## BO/EMP features
                    ## Peripherals: Time, sim_nr (total and current)
                    ## Save in json periodically! (important at the end)
                  ## Save employee persona information/features ==> DONE
                  ## Save time of hiring ==> DONE
                  ## Save peripheral information ==> DONE
                    ## Who was approached? Who was not chosen for interview? Who rejected interview? Who was not considered for job? Who rejected job?
                    ## I.e., at every decision save information (features, time, number of candidate)
                    ## Could save in json file (probably easier to save in tracker)

          # To Do:
            # Make sure all code is updated (reverie, converse, run_gpt_prompt) ==> DONE
            # Extract the features of the hiring person ==> DONE
            # Extract the features of the hired person ==> DONE
            # Wrap in experiment: ==> DONE
              # Set number of time fixed (e.g., 12/24/48hrs in game)
              # Set up loop of reverie (start server, website, run everything, stop after time or condition, start next iteration!)
              # Stop once all BO have hired (print if not)
              # Save the combination of who hired whome somewhere: E.g., {Hiring: Persona A, Hired: Persona B}
              # Better: Save dict with all experiment parameters:
                # {Hiring_features: A, Hired_features: B, sim_time: X, hiring_time, interview: X, interview_gist: X, impressions: X}
            # Maybe: add hiring persona's features! ==> DONE
            # Maybe: track who was approached, offered, not hired (i.e., get information whether groups did not want the job!) ==> DONE

          ### TO DO:
            ### implement counter: approach 3 people, continue until at least one wants the job ==> Done
        
        ### add function to ask employee whether they would accept an offer, if yes proceed ==> DONE
        ### save interview convo ==> DONE
        ### add function to extract impression/summary ==> DONE
          ### prompt to give a summary of the interview ==> DONE
          ### prompt to give a summary of the impressions of the interviewee (e.g., ratings) ==> DONE
          ### save somewhere in agent (e.g., as a dict: persona_name: {interview: interview, impression: impression}) ==> DONE
          ### if interview counter = N, compare summaries and choose most suitable one. ==> DONE

    ### check if init/target persona is BO who is currently hiring and other is EMPL, if yes proceed ==> DONE
    ### define function check if BO wants to hire based on conversation and retrieved experience/impression of EMP, if yes proceed ==> DONE
    ### define function summarize job details (job role name, job role description, daily routine/hours ==> similar to ISS) ==> DONE
    ### define function to check if EMP wants to get new job based on current circumstances and job details, if yes proceed ==> DONE
    
    
  print ("July 23 PU")
  for row in curr_chat: 
    print (row)
  print ("July 23 FIN")

  robustness_case = "convo" # add different cases to check: Convo, hiring, acting
  robustness_context = [maze, target_persona, retrieved, curr_chat]
  robustness_items = run_gpt_robustness_check(init_persona, robustness_context, robustness_case) ## add code to construct context, run robustness checks and return

  # then log the outcome
  logging_info = {"interaction_count": "", "interview_count": "", "Decision_type": "robustness_convo", "Decision": robustness_items, "Init": init_persona, "Target": target_persona, 
                "Sim_Time": init_persona.scratch.curr_time, "Convo": curr_chat_transcript, "Interview": "", "target_impression": "", "target_rating": "", "target_rating_soc": "", "Group Condition": init_persona.scratch.group_condition, 
                "Init Group": init_persona.scratch.group_identity, "Target Group": target_persona.scratch.group_identity, "convo_hate": "", "interview_hate": ""}
  TRACKER.log_decision(logging_info, init_persona.scratch.sim_nr)

  ### log general convo 
  # -> add reflection about hate/aggression

  return curr_chat


def generate_summarize_ideas(persona, nodes, question): 
  statements = ""
  for n in nodes:
    statements += f"{n.embedding_key}\n"
  summarized_idea = run_gpt_prompt_summarize_ideas(persona, statements, question)[0]
  return summarized_idea


def generate_next_line(persona, interlocutor_desc, curr_convo, summarized_idea):
  # Original chat -- line by line generation 
  prev_convo = ""
  for row in curr_convo: 
    prev_convo += f'{row[0]}: {row[1]}\n'

  next_line = run_gpt_prompt_generate_next_convo_line(persona, 
                                                      interlocutor_desc, 
                                                      prev_convo, 
                                                      summarized_idea)[0]  
  return next_line


def generate_inner_thought(persona, whisper):
  inner_thought = run_gpt_prompt_generate_whisper_inner_thought(persona, whisper)[0]
  return inner_thought

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


def generate_poig_score(persona, event_type, description): 
  if debug: print ("GNS FUNCTION: <generate_poig_score>")

  if "is idle" in description: 
    return 1

  if event_type == "event" or event_type == "thought": 
    return run_gpt_prompt_event_poignancy(persona, description)[0]
  elif event_type == "chat": 
    return run_gpt_prompt_chat_poignancy(persona, 
                           persona.scratch.act_description)[0]


def load_history_via_whisper(personas, whispers):
  for count, row in enumerate(whispers): 
    persona = personas[row[0]]
    whisper = row[1]

    thought = generate_inner_thought(persona, whisper)

    created = persona.scratch.curr_time
    expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
    s, p, o = generate_action_event_triple(thought, persona)
    keywords = set([s, p, o])
    thought_poignancy = generate_poig_score(persona, "event", whisper)
    thought_embedding_pair = (thought, get_embedding(thought))
    persona.a_mem.add_thought(created, expiration, s, p, o, 
                              thought, keywords, thought_poignancy, 
                              thought_embedding_pair, None)


def open_convo_session(persona, convo_mode): 
  if convo_mode == "analysis": 
    curr_convo = []
    interlocutor_desc = "Interviewer"

    while True: 
      line = input("Enter Input: ")
      if line == "end_convo": 
        break

      if int(run_gpt_generate_safety_score(persona, line)[0]) >= 8: 
        print (f"{persona.scratch.name} is a computational agent, and as such, it may be inappropriate to attribute human agency to the agent in your communication.")        

      else: 
        retrieved = new_retrieve(persona, [line], 50)[line]
        summarized_idea = generate_summarize_ideas(persona, retrieved, line)
        curr_convo += [[interlocutor_desc, line]]

        next_line = generate_next_line(persona, interlocutor_desc, curr_convo, summarized_idea)
        curr_convo += [[persona.scratch.name, next_line]]


  elif convo_mode == "whisper": 
    whisper = input("Enter Input: ")
    thought = generate_inner_thought(persona, whisper)

    created = persona.scratch.curr_time
    expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
    s, p, o = generate_action_event_triple(thought, persona)
    keywords = set([s, p, o])
    thought_poignancy = generate_poig_score(persona, "event", whisper)
    thought_embedding_pair = (thought, get_embedding(thought))
    persona.a_mem.add_thought(created, expiration, s, p, o, 
                              thought, keywords, thought_poignancy, 
                              thought_embedding_pair, None)



### Fill out reflection modules
  ### Take in convo/decomp plan -> return if action contains hate, potentially on scale!
def reflect_convo(init_persona, target_persona, curr_chat):
    """
    Takes in a conversation and outputs whether each participant contains hateful or aggressive language
    or expresses hate or aggressive intent.

    Args:
        init_persona: The initial persona in the conversation.
        target_persona: The target persona in the conversation.
        job_details_str: A string describing the job details.
        curr_chat: A list of tuples representing the conversation. Each tuple contains (speaker, utterance).

    Returns:
        dict: A dictionary indicating hate/aggression presence for each participant, e.g., {"init": True, "target": False}
    """
    
    return run_gpt_prompt_reflect_convo(init_persona, target_persona, curr_chat)



def reflect_action(init_persona, target_persona, action_str, decomp_str=None):
    """
    Evaluates an action schedule to determine if it reflects hate or aggression by either participant.

    Args:
        init_persona: The initial persona related to the actions.
        target_persona: The target persona related to the actions.
        job_details_str: A string describing the job details (if relevant).
        action_str: Main action for the hour
        decomp_str: A list of actions or a to be evaluated (decomposed from main action).

    Returns:
        dict: A dictionary indicating whether each participant's actions reflect hate/aggression.
              Format: {"init": True/False, "target": True/False}
    """    
    if decomp_str:
      actions_total = f"{action_str}:\n"
      for action, dur in decomp_str:
          actions_total += f"- {action}\n"
    else:
      actions_total = action_str
    
    return run_gpt_prompt_reflect_action(init_persona, target_persona, actions_total)

def reflect_groups(init_persona, action_str, decomp_str=None):
    """
    Evaluates an action schedule to determine if it reflects hate or aggression by either participant.

    Args:
        init_persona: The initial persona related to the actions.
        target_persona: The target persona related to the actions.
        job_details_str: A string describing the job details (if relevant).
        action_str: Main action for the hour
        decomp_str: A list of actions or a to be evaluated (decomposed from main action).

    Returns:
        dict: A dictionary indicating whether each participant's actions reflect hate/aggression.
              Format: {"init": True/False, "target": True/False}
    """    
    if decomp_str:
      actions_total = f"{action_str}:\n"
      for action, dur in decomp_str:
          actions_total += f"- {action}\n"
    else:
      actions_total = action_str
    
    return run_gpt_prompt_reflect_groups(init_persona, actions_total)

