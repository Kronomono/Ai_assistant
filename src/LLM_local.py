import os
from dotenv import load_dotenv
import logging
import torch
from llama_cpp import Llama
from memory import Memory
from datetime import datetime
from personality_utils import load_embedded_personality
from hyperdb import HyperDB

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMWrapper:
    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None
        self.memory = Memory()
        self.name = os.getenv('NAME_OF_BOT')
        self.role = os.getenv('ROLE_OF_BOT')
        self.personality, self.personality_embedding = load_embedded_personality()
        self.db = HyperDB()

    def initialize(self):
        if self.llm is None:
            logger.info("Initializing LLM...")
            gpu_layers = -1 if torch.cuda.is_available() else 0 
            logger.info(f"Using GPU layers: {gpu_layers}")
            self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_batch=512, n_gpu_layers=gpu_layers, verbose=True)
            logger.info("LLM initialized successfully.")

    def ask(self, prompts, format="", temperature=0.7):
        self.initialize()
        prompt = " ".join([p["content"] for p in prompts])
        logger.debug(f"Sending prompt to Llama: {prompt}")
        logger.debug(f"Temperature: {temperature}")
        output = self.llm(prompt, max_tokens=2048, temperature=temperature, top_p=0.9, echo=False)
        logger.debug(f"Raw output from Llama: {output}")
        response = output['choices'][0]['text'].strip()
        logger.debug(f"Stripped response: {response}")
        return response
    
    def close(self):
        if self.llm is not None:
            del self.llm
            self.llm = None
            logger.info("LLM resources released.")

    def generate_response(self, user_input):
        self.initialize()

        context = self.memory.get_relevant_context(user_input)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        response = self.perform_local_query(user_input, context, current_time)

        self.memory.add_conversation([
            {"role": "user", "content": user_input, "timestamp": current_time},
            {"role": self.role, "content": response, "timestamp": current_time}
        ])

        logger.debug(f"DEBUG: LLM generated response: {response}")
        
        return response
    
    def perform_local_query(self, user_input, context, current_time):
        prompt = f"""As {self.name}, a {self.role}, respond to the following input:

        Personality:
        {self.personality}

        Context from previous conversations:
        {context}

        Current date and time: {current_time}

        User input:
        {user_input}

        Instructions:
        1. Respond authentically as {self.name}, based on the personality description provided.
        2. Use the context from previous conversations to maintain continuity.
        3. If you don't have enough information to answer the query, state that clearly.
        4. Use the current date and time information when relevant to the query.

        Response:"""

        local_response = self.ask([{'role': 'user', 'content': prompt}], temperature=0.5)
        logger.debug(f"Local response: {local_response}")
        
        return local_response.strip()

# Create a global instance
llm_wrapper = LLMWrapper(os.getenv("LLM_MODEL_PATH"))

if __name__ == "__main__":
    prompt = "Hey, can you tell me about yourself?" 
    print(f"Processing prompt: {prompt}")
    response = llm_wrapper.generate_response(prompt)
    print("\nGenerated output:")
    print(response)

    print("\nDebug Information:")
    print(f"LLM initialized: {llm_wrapper.llm is not None}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated()}")
    llm_wrapper.close()