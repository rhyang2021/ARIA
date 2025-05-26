import random
from typing import Optional, Dict
import time
from openai import OpenAI
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import pdb
from transformers import T5Tokenizer, T5ForConditionalGeneration
from llm_base import llm_azure, llm_openai, llm_gpt, vllm
import concurrent.futures

INSTRUCTION_GUESS_MY_CITY = """ 
Let's play a game of Guess My City.
In each round, you must ask me a question to guess the city I'm thinking of. Continue asking until you guess the correct city.

Your question format must **follow this format exactly**: 
Question: <your question>

For example: 
Question: What is the primary language spoken in your city?
Question: Is your city located on the east coast of the country?

{history}
Now, please proceed by asking your next question:
"""

CITY_LIST = ['Seoul, South Korea',
 'Sao Paulo, Brazil',
 'Bombay, India',
 'Jakarta, Indonesia',
 'Karachi, Pakistan',
 'Moscow, Russia',
 'Istanbul, Turkey',
 'Shanghai, China',
 'Tokyo, Japan',
 'Bangkok, Thailand',
 'Beijing, China',
 'Delhi, India',
 'London, UK',
 'Cairo, Egypt',
 'Tehran, Iran',
 'Bogota, Colombia',
 'Bandung, Indonesia',
 'Tianjin, China',
 'Lima, Peru',
 'Lahore, Pakistan',
 'Bogor, Indonesia',
 'Santiago, Chile',
 'Shenyang, China',
 'Calcutta, India',
 'Wuhan, China',
 'Sydney, Australia',
 'Guangzhou, China',
 'Singapore, Singapore',
 'Madras, India',
 'Baghdad, Iraq',
 'Pusan, South Korea',
 'Yokohama, Japan',
 'Dhaka, Bangladesh',
 'Berlin, Germany',
 'Alexandria, Egypt',
 'Bangalore, India',
 'Malang, Indonesia',
 'Hyderabad, India',
 'Chongqing, China',
 'Haerbin, China',
 'Ankara, Turkey',
 'Buenos Aires, Argentina',
 'Chengdu, China',
 'Ahmedabad, India',
 'Casablanca, Morocco',
 'Chicago, USA',
 'Xian, China',
 'Madrid, Spain',
 'Surabaya, Indonesia',
 'Pyong Yang, North Korea',
 'Nanjing, China',
 'Kinshaha, Congo',
 'Rome, Italy',
 'Taipei, China',
 'Osaka, Japan',
 'Kiev, Ukraine',
 'Yangon, Myanmar',
 'Toronto, Canada',
 'Zibo, China',
 'Dalian, China',
 'Taega, South Korea',
 'Addis Ababa, Ethopia',
 'Jinan, China',
 'Salvador, Brazil',
 'Inchon, South Korea',
 'Semarang, Indonesia',
 'Giza, Egypt',
 'Changchun, China',
 'Havanna, Cuba',
 'Nagoya, Japan',
 'Belo Horizonte, Brazil',
 'Paris, France',
 'Tashkent, Uzbekistan',
 'Fortaleza, Brazil',
 'Sukabumi, Indonesia',
 'Cali, Colombia',
 'Guayaquil, Ecuador',
 'Qingdao, China',
 'Izmir, Turkey',
 'Cirebon, Indonesia',
 'Taiyuan, China',
 'Brasilia, Brazil',
 'Bucuresti, Romania',
 'Faisalabad, Pakistan',
 'Medan, Indonesia',
 'Houston, USA',
 'Mashhad, Iran',
 'Medellin, Colombia',
 'Kanpur, India',
 'Budapest, Hungary',
 'Caracas, Venezuela']

INITIAL_STR = "Questions:\n"

PROMPT_TEMPLATE = 'You are playing a game called Guess My City with me. The rule of Guess My City is that you are given a hidden city, and I am guessing what the city is within twenty questions. For every question, you can give a free-form brief answer (e.g., "Yes, it is located on the east coast of the country.", "The primary language spoken in the city is Spanish."). Your answer should never include the name of the hidden city. Now the hidden word given to you is "{word}", and the question for the current round is "{question}". Your response is:'


class GuessMyCityEnv():
    def __init__(
        self, 
        # word_list,  
        max_conversation_length: int=20,
    ):
        self.city_list = CITY_LIST
        self.max_conversation_length = max_conversation_length
        self.random = random.Random(None)
        self.count = 0
        self.curr_word = None
        self.history = ''
        self.done = True

    def generate_answers(self, curr_words, questions):
        inputs = [f"The object is {curr_word}." + question for  curr_word, question in zip(curr_words, questions)]
        encoder_ids = self.tokenizer(inputs ,padding=True, return_tensors='pt').to(self.model.device)
        return self.tokenizer.batch_decode(self.model.generate(input_ids=encoder_ids['input_ids'], attention_mask=encoder_ids['attention_mask'],\
                                                                max_new_tokens=64, do_sample = False), skip_special_tokens= True)
                
    def is_correct(self, question):
        #check for the last word
        # cut out punctuations at the end
        while len(question) > 0 and not question[-1].isalpha():
            question = question[:-1]

        if len(question) == 0:
            return False
        # this is the name of the city
        word = self.curr_word.lower().split(",")[0]
        return word in question.lower()

    def _step(self, question):
        question = question.strip() if len(question.split("Question:")) <= 1 else question.split("Question:")[-1].strip()
        question_instruction = [{"role": "user", "content": PROMPT_TEMPLATE.format(word=self.curr_word.lower().split(",")[0], question=question)}]
        answer = llm_openai(prompt=question_instruction, model="gpt-4")
        
        if self.done:
            return None
        if self.curr_word.lower().split(",")[0] in answer.lower():
            answer = "I can't answer that question."
        self.count+=1
        self.history += f"Question: {question}. Answer: {answer}\n"
        self.message = [{"role": "user", "content": INSTRUCTION_GUESS_MY_CITY.format(history=self.history)}]

        done = self.is_correct(question)
        reward = -1
        #if correct reward is -1
        if done:
            reward = 0
        self.done = done or self.count == self.max_conversation_length
        return  self.message, answer, reward, self.done
        
    def reset(self, idx : Optional[int]=None, curr_word: Optional[str]=None):
        self.count = 0 
        if idx is not None:
            self.curr_word = self.city_list[idx]
        if curr_word is not None:
            self.curr_word = curr_word
        else:
            self.curr_word = self.random.choice(self.city_list)
        self.history = 'Here is the game history: \n'
        self.message = [{"role": "user", "content": INSTRUCTION_GUESS_MY_CITY.format(history='')}]
        self.done = False
        return self.message
        # return (Text(INITIAL_STR, is_action=False),)


class BatchedGuessMyCityEnv():
    def __init__(
        self, 
        env_load_path: str,
        device,
        cache_dir: str,
        max_conversation_length: int=20,
        bsize: int=32,
    ):
        self.env_list = [GuessMyCityEnv(max_conversation_length) for _ in range(bsize)]
        self.bsize = bsize
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", cache_dir=cache_dir)
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", cache_dir=cache_dir).to(device)
        self.model.load_state_dict(torch.load(env_load_path)['model_state_dict'])

    def generate_answers(self, questions):
        curr_words = [env.curr_word for env in self.env_list]
        inputs = [f"Your home town is {curr_word}." + question for  curr_word, question in zip(curr_words, questions)]
        encoder_ids = self.tokenizer(inputs ,padding=True, return_tensors='pt').to(self.model.device)
        return self.tokenizer.batch_decode(self.model.generate(input_ids=encoder_ids['input_ids'], 
                                            attention_mask=encoder_ids['attention_mask'],
                                            max_new_tokens=64, do_sample = False), 
                                            skip_special_tokens= True)

    def reset(self, idx: Optional[int] = None):
        return [env.reset(idx) for env in self.env_list]
    
    def step(self, questions):
        answers = self.generate_answers(questions)
        # print("Step once!")
        with concurrent.futures.ThreadPoolExecutor() as executor: 
            jobs = [executor.submit(env._step, q, a) for env, q, a in zip(self.env_list, questions, answers)]
            results = [job.result() for job in jobs]
        return results
