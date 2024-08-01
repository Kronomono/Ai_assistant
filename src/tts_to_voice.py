import re
import json
import os
from dotenv import load_dotenv
import logging
import requests
import torch
from llama_cpp import Llama
from llm_axe.core import internet_search, read_website
from memory import Memory
from datetime import datetime
load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
class LlamaWrapper:
    def __init__(self, llama_model):
        self.llama = llama_model
    def ask(self, prompts, format="", temperature=0.7):
        prompt = " ".join([p["content"] for p in prompts])
        logger.debug(f"Sending prompt to Llama: {prompt}")
        output = self.llama(prompt, max_tokens=2048, temperature=temperature, top_p=0.9, echo=False)
        logger.debug(f"Raw output from Llama: {output}")
        response = output['choices'][0]['text'].strip()
        logger.debug(f"Stripped response: {response}")
        return self.parse_json_response(response) if format == "json" else response
    def parse_json_response(self, response):
        try:
            return json.dumps(json.loads(response))
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.dumps(json.loads(json_match.group()))
                except json.JSONDecodeError:
                    pass
            return json.dumps({"response": response})
class LLMWrapper:
    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None
        self.llama_wrapper = None
        self.memory = Memory()
        self.name = os.getenv('NAME_OF_BOT')  
        self.role = os.getenv('ROLE_OF_BOT')  
    def initialize(self):
        if self.llm is None:
            logger.info("Initializing LLM...")
            gpu_layers = -1 if torch.cuda.is_available() else 0
            logger.info(f"Using GPU layers: {gpu_layers}")
            self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_batch=512, n_gpu_layers=gpu_layers, verbose=True)
            self.llama_wrapper = LlamaWrapper(self.llm)
            logger.info("LLM initialized successfully.")
    def ask(self, prompts: list, format: str = "", temperature: float = 0.7):
        self.initialize()
        return self.llama_wrapper.ask(prompts, format, temperature)
    
    def close(self):
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.llama_wrapper = None
            logger.info("LLM resources released.")
    def classify_query(self, query):
        classification_prompt = f"""
        As {self.name}, a {self.role}, classify the following query into one of these categories:
        1. 'local': Can be handled with existing information or {self.role} capabilities
        2. 'online': - Requires up-to-date or specific information from the internet (eg. current news events, weather, stock prices etc.)
            - Ask for specific web content or links
            - Requires information about recent or upcoming events
	@@ -143,7 +143,7 @@ def perform_local_query(self, user_input, context, current_time):
1. Maintain the persona of {self.name}, the {self.role}.
2. If asked about your capabilities or identity, respond accordingly.
3. If you don't have enough information to answer the query, state that clearly.
4. Response as a {self.role}, and be confident in your responses.
5. Always provide a substantive response.
6. Use the current date and time information when relevant to the query.
	@@ -230,7 +230,7 @@ def extract_information(self, content, question, current_time):
llm_wrapper = LLMWrapper(os.getenv("LLM_MODEL_PATH"))

if __name__ == "__main__":
    prompt = "Hello Akane, I am your creator Matthew. Its nice to meet you."
    print(f"Processing prompt: {prompt}")
    response = llm_wrapper.generate_response(prompt)
    print("\nGenerated output:")
    print(response)
    print("\nDebug Information:")
    print(f"LLM initialized: {llm_wrapper.llm is not None}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    llm_wrapper.close()