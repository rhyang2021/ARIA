import os
import traceback
from openai import OpenAI, AzureOpenAI
import requests
from requests.exceptions import Timeout
from typing import Dict, List
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import pdb

def llm_openai(prompt: List[Dict[str, str]], model: str):
    """Get completion from the GPT model."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages = prompt,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return None


def vllm(prompt: List[Dict[str, str]], model: str, port=8031, temperature=0) -> str:
    model_id = "model_path"
    base_url = f"http://hostname:{port}/v1"

    client = OpenAI(
            base_url=base_url,
            api_key="EMPTY")

    for i in range(5):
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=prompt,
                temperature=temperature,
                )
            text = completion.choices[0].message.content
            return text.strip()
        
        except Exception as e:
            print(f'ERROR: {str(e)}')
            print(f'Retrying ({i + 1}/5), wait for {2 ** (i + 1)} sec...')
            time.sleep(2 ** (i + 1))
    return ""

def hf_request(prompt, port, temperature):
    url = f'http://hostname:{port}/generate'
    # request
    data = {
        'index': 1,
        'messages': prompt,
        'temperature': temperature
        }
    # POST
    response = requests.post(url, json=data)
    print(response.json())
    
    return response.json()['response']
