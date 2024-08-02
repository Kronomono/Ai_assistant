import re
import os
from dotenv import load_dotenv
import logging
import requests
import torch
from llama_cpp import Llama
from llm_axe.core import internet_search, read_website
from memory import Memory
from datetime import datetime
from personality_utils import load_embedded_personality
from hyperdb import HyperDB, get_embedding


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
            gpu_layers =  -1 if torch.cuda.is_available() else 0 
            logger.info(f"Using GPU layers: {gpu_layers}")
            self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_batch=512, n_gpu_layers=gpu_layers, verbose=True)
            logger.info("LLM initialized successfully.")

    def ask(self, prompts, format="", temperature=0.7):
        self.initialize()
        prompt = " ".join([p["content"] for p in prompts])
        logger.debug(f"Sending prompt to Llama: {prompt}")
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

    def classify_query(self, query):
        classification_prompt = f"""
        As {self.name}, a {self.role}, classify the following query into one of these categories. Just a single classification:
        1. 'local': Can be handled with existing information or {self.role} capabilities. Or requires information about the user that can be obtained from memory.
        2. 'online': - Requires up-to-date or specific information from the internet (eg. current news events, weather, stock prices etc.)
            - Ask for specific web content or links
            - Requires information about recent or specific events
            - Involves searching for or comparing products or services
            - Needs information about a specific person, place or thing, that is not general knowledge, and is not about you or the user. (eg. "Who is the CEO of Google?", "What is the capital of France?")
        

        Query: "{query}"

        Classification (local/online):
        Explanation:
        """
        response = self.ask([{'role': 'user', 'content': classification_prompt}], temperature=0.3).strip()
        
        classification = 'local'
        explanation = ''
        
        if 'online' in response.lower():
            classification = 'online'
       
        
        explanation_match = re.search(r'Explanation:(.*)', response, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        logger.info(f"Query classification: {classification}")
        logger.info(f"Classification explanation: {explanation}")
        
        return classification, explanation

    def generate_response(self, user_input):
        self.initialize()

        query_type, explanation = self.classify_query(user_input)
        context = self.memory.get_relevant_context(user_input)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if query_type == "online":
            logger.info(f"Online query detected: {explanation}")
            response = self.perform_online_search(user_input, current_time)
        
        else:
            logger.info(f"Local query detected: {explanation}")
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
        logger.debug(f"Local/Personal response: {local_response}")
        
        return local_response.strip()

    def perform_online_search(self, user_input, current_time):
        try:
            url = self.extract_url(user_input)
            online_response = self.process_specific_url(url, user_input, current_time) if url else self.enhanced_online_search(user_input, current_time)
            
            if online_response:
                logger.info("Valid online information found.")
                return online_response
            else:
                logger.info("No valid online information found.")
                return self.perform_local_query(user_input, "", current_time)
        except Exception as e:
            logger.error(f"Error during online search: {e}", exc_info=True)
            return self.perform_local_query(user_input, "", current_time)

    @staticmethod
    def extract_url(text):
        url_match = re.search(r'https?://\S+', text)
        return url_match.group() if url_match else None

    def process_specific_url(self, url, user_input, current_time):
        try:
            page_content = self.safe_read_website(url)
            if page_content:
                extracted_info = self.extract_information(page_content, user_input, current_time)
                logger.info(f"Extracted information from {url}: {extracted_info}")
                if extracted_info and "No relevant information found" not in extracted_info:
                    return f"Based on {url}, {extracted_info}"
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
        return None

    def enhanced_online_search(self, user_input, current_time):
        logger.info("Performing general search...")
        search_results = internet_search(user_input)
        if search_results:
            for result in search_results[:5]:
                response = self.process_specific_url(result.get('url', ''), user_input, current_time)
                if response:
                    return response
        return None

    @staticmethod
    def safe_read_website(url):
        try:
            return read_website(url)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error reading website {url}: {e}")
            return None

    def extract_information(self, content, question, current_time):
        prompt = f"""Extract the most relevant information from the following content to answer this question: "{question}"

        Content:
        {content[:5000]}

        Current date and time: {current_time}

        Instructions:
        1. Extract only factual information directly related to the question.
        2. If you find a clear, direct answer, state it concisely and completely.
        3. Include specific details such as numbers, names, or other relevant data if mentioned.
        4. If a complete list or description is available, provide it in full.
        5. If no relevant information is found, state "No relevant information found."
        6. Do not infer or generate information not present in the content.
        7. Use the current date and time information when relevant to the query.

        Extracted Information:"""

        return self.ask([{'role': 'user', 'content': prompt}], temperature=0.3).strip()

# Create a global instance
llm_wrapper = LLMWrapper(os.getenv("LLM_MODEL_PATH"))

if __name__ == "__main__":
    prompt = "Hey Akane your smart could you use this link to tell me what number cosmog is in the national pokedex? https://bulbapedia.bulbagarden.net/wiki/Cosmog_(Pok%C3%A9mon)" 
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
