"""
Original Author: Joon Sung Park (joonspk@stanford.edu)
Modifications: Suhaib Abdurahman (EMAIL)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
Modifier-Description: Replaces OpenAI API with a local Mistral derivative
"""
import json
import random
#import openai
import time 
import requests

from utils import *
# openai.api_key = openai_api_key

# def process_sse_stream(url, headers, data):
#     with requests.post(url, headers=headers, json=data, verify=False, stream=True) as response:
#         buffer = ''
#         for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
#             buffer += chunk
#             while '\r\n\r\n' in buffer:
#                 raw_event, buffer = buffer.split('\r\n\r\n', 1)
#                 try:
#                     # Parse the SSE event to extract the JSON data
#                     json_start = raw_event.find('{')  # Find the start of the JSON object
#                     if json_start != -1:
#                         json_data = json.loads(raw_event[json_start:])
#                         # Iterate through the choices and print the content
#                         for choice in json_data.get("choices", []):
#                             delta = choice.get("delta", {})
#                             content = delta.get("content", "")
#                             print(content, end='')  # Print continuously on the same line
#                 except json.JSONDecodeError:
#                     print("Error decoding JSON from event:", raw_event)
#         print()  # Ensure there's a newline at the end of the stream


def LLM_single_request(prompt): 
    """
    Given a prompt and a dictionary of MISTRAL parameters, make a request to the local API
    server and returns the response. 
    ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
    RETURNS: 
    a str of Local LLM (here MISTRAL-7B) response. 
    """
    params = {"max_tokens": 512, "temperature": 0.7, "top_p": 0.9, "min_p": 0, "top_k": 40, "repetition_penalty": 1.15, 
          "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty_range": 1024, "typical_p": 1, "tfs": 1, 
          "top_a": 0, "epsilon_cutoff": 0, "eta_cutoff": 0, "guidance_scale": 1, "mirostat_mode": 0, "mirostat_tau": 5, 
          "mirostat_eta": 0.1, "smoothing_factor": 0, "do_sample": True, "seed": 42, "encoder_repetition_penalty": 1, 
          "min_length": 0, "no_repeat_ngram_size": 0, "stream": False,
          #"num_beams": 1, "penalty_alpha": 0, "length_penalty": 1, "early_stopping": false, 
         }
    
    data = {
            "mode": "instruct",
            "messages": [{"role": "user", "content": prompt}],
            "instruction_template": "ChatML",
        }
    
    data = data | params
    
    url = "http://127.0.0.1:5000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # temp_sleep()
    try: 
        response = requests.post(url, headers=headers, json=data, verify=False)
        assistant_message = response.json()['choices'][0]['message']['content']
      
        return assistant_message
    
    except: 
        print ("MISTRAL ERROR")
        return "MISTRAL ERROR"

prompt = """
---
Character 1: Maria Lopez is working on her physics degree and streaming games on Twitch to make some extra money. She visits Hobbs Cafe for studying and eating just about everyday.
Character 2: Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities.

Past Context: 
138 minutes ago, Maria Lopez and Klaus Mueller were already conversing about conversing about Maria's research paper mentioned by Klaus This context takes place after that conversation.

Current Context: Maria Lopez was attending her Physics class (preparing for the next lecture) when Maria Lopez saw Klaus Mueller in the middle of working on his research paper at the library (writing the introduction).
Maria Lopez is thinking of initating a conversation with Klaus Mueller.
Current Location: library in Oak Hill College

(This is what is in Maria Lopez's head: Maria Lopez should remember to follow up with Klaus Mueller about his thoughts on her research paper. Beyond this, Maria Lopez doesn't necessarily know anything more about Klaus Mueller) 

(This is what is in Klaus Mueller's head: Klaus Mueller should remember to ask Maria Lopez about her research paper, as she found it interesting that he mentioned it. Beyond this, Klaus Mueller doesn't necessarily know anything more about Maria Lopez) 

Here is their conversation. 

Maria Lopez: "
---
Output the response to the prompt above in json. The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"]. Output multiple utterances in ther conversation until the conversation comes to a natural conclusion.
Example output json:
{"output": "[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]"}
"""

print (LLM_single_request(prompt))












