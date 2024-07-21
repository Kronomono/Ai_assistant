import re
import json
import logging
import requests
from llama_cpp import Llama
from llm_axe.core import internet_search, read_website
from memory import Memory  # Import the Memory class

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LlamaWrapper:
    def __init__(self, llama_model):
        self.llama = llama_model

    def ask(self, prompts, format="", temperature=0.7):
        prompt = " ".join([p["content"] for p in prompts])
        output = self.llama(prompt, max_tokens=2048, temperature=temperature, top_p=0.9, echo=False)
        response = output['choices'][0]['text'].strip()

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
        self.memory = Memory()  # Initialize the Memory instance
        self.assistant_name = "Akane"
        self.assistant_role = "virtual assistant"

    def initialize(self):
        if self.llm is None:
            self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_batch=512, n_gpu_layers=-1, verbose=False)
            self.llama_wrapper = LlamaWrapper(self.llm)

    def ask(self, prompts: list, format: str = "", temperature: float = 0.7):
        self.initialize()
        return self.llama_wrapper.ask(prompts, format, temperature)

    def classify_query(self, query):
        classification_prompt = f"""
        As {self.assistant_name}, a {self.assistant_role}, classify the following query into one of these categories:
        1. 'personal': Related to your identity, capabilities, or assistant-specific information
        2. 'local': simple tasks that can be handled with existing information
        3. 'online': 
            - Requires up-to-date or specific information from the internet (eg. current events, weather, stock prices etc.)
            - Ask for specific web content or links
            - Requires information about recent or upcoming events or has the date/time in the query
            - Involves searching for or comparing products or services
            - Needs information about a specfic person, place or thing, that is not general knowledge, and is not about you or the user. (eg. "Who is the CEO of Google?", "What is the capital of France?")

        Query: "{query}"

        Classification (personal/local/online):
        Explanation:
        """
        response = self.ask([{'role': 'user', 'content': classification_prompt}], temperature=0.3).strip()
        
        classification = 'local'  # Default to local
        explanation = ''
        
        if 'personal' in response.lower():
            classification = 'personal'
        elif 'online' in response.lower():
            classification = 'online'
        
        explanation_match = re.search(r'Explanation:(.*)', response, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        logger.info(f"Query classification: {classification}")
        logger.info(f"Classification explanation: {explanation}")
        
        return classification, explanation

    def generate_response(self, user_input):
        if self.llm is None:
            self.initialize()

        query_type, explanation = self.classify_query(user_input)

        # Retrieve relevant context from memory
        context = self.memory.get_relevant_context(user_input)
        
        # Prepare the prompt with context
        prompt_with_context = f"""As {self.assistant_name}, a {self.assistant_role}, respond to the following input:

Context from previous conversations:
{context}

User input:
{user_input}

Response:"""

        if query_type == "personal":
            logger.info(f"Personal query detected: {explanation}")
            response = self.perform_personal_query(prompt_with_context)
        elif query_type == "online":
            logger.info(f"Online query detected: {explanation}")
            response = self.perform_online_search(prompt_with_context)
        else:
            logger.info(f"Local query detected: {explanation}")
            response = self.perform_local_query(prompt_with_context)

        # Save the interaction to memory
        self.memory.add_to_conversation("user", user_input)
        self.memory.add_to_conversation("assistant", response)
        self.memory.save_current_conversation()

        # Append the new conversation to the existing memory file
        try:
            self.memory.append_to_file("conversation_history.gz")
        except Exception as e:
            logger.error(f"Error appending to memory file: {e}")
            # Optionally, you can add a fallback method here, such as:
            # self.memory.save_to_file("conversation_history_new.gz")

        return response

    def perform_personal_query(self, prompt):
        personal_response = self.get_personal_response(prompt)
        logger.info("Personal response generated.")
        return f"[As {self.assistant_name}] {personal_response}"

    def perform_local_query(self, prompt):
        local_response = self.get_local_knowledge_response(prompt)
        logger.info("Local knowledge response generated.")
        return f"[Using local knowledge] {local_response}"

    def perform_online_search(self, user_input):
        try:
            url = self.extract_url(user_input)
            online_response = self.process_specific_url(url, user_input) if url else self.enhanced_online_search(user_input)
            
            if online_response:
                logger.info("Valid online information found.")
                return f"[Using online search] {online_response}"
            else:
                logger.info("No valid online information found.")
                return self.perform_local_query(user_input)  # Fallback to local query if online search fails
        except Exception as e:
            logger.error(f"Error during online search: {e}", exc_info=True)
            return self.perform_local_query(user_input)  # Fallback to local query if online search fails

    @staticmethod
    def extract_url(text):
        url_match = re.search(r'https?://\S+', text)
        return url_match.group() if url_match else None

    def process_specific_url(self, url, user_input):
        try:
            page_content = self.safe_read_website(url)
            if page_content:
                extracted_info = self.extract_information(page_content, user_input)
                logger.info(f"Extracted information from {url}: {extracted_info}")
                if extracted_info and "No relevant information found" not in extracted_info:
                    return f"Based on {url}, {extracted_info}"
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
        return None

    def enhanced_online_search(self, user_input):
        logger.info("Performing general search...")
        search_results = internet_search(user_input)
        if search_results:
            for result in search_results[:5]:  # Check top 5 results
                response = self.process_specific_url(result.get('url', ''), user_input)
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

    def extract_information(self, content, question):
        prompt = f"""Extract the most relevant information from the following content to answer this question: "{question}"

Content:
{content[:5000]}  # Limit content to 5000 characters

Instructions:
1. Extract only factual information directly related to the question.
2. If you find a clear, direct answer, state it concisely and completely.
3. Include specific details such as numbers, names, or other relevant data if mentioned.
4. If a complete list or description is available, provide it in full.
5. If no relevant information is found, state "No relevant information found."
6. Do not infer or generate information not present in the content.

Extracted Information:"""

        return self.ask([{'role': 'user', 'content': prompt}], temperature=0.3).strip()

    def get_local_knowledge_response(self, prompt):
        local_prompt = f"""As {self.assistant_name}, a {self.assistant_role}, answer the following question to the best of your ability:

{prompt}

Instructions:
1. If you are certain about the answer, provide it directly and completely.
2. If you have relevant information but are not certain, provide the information and clearly state your level of confidence.
3. Include specific details, numbers, or names if the question asks for them.
4. If you don't have enough information to answer accurately, say "I don't have enough reliable information to answer this question accurately."
5. Do not make up or guess at information you're not confident about. It's better to express uncertainty than to provide potentially incorrect information.

Answer:"""

        local_response = self.ask([{'role': 'user', 'content': local_prompt}], temperature=0.3)
        logger.debug(f"Local knowledge response: {local_response}")
        
        return local_response.strip() or "I don't have enough reliable information to answer this question accurately."

    def get_personal_response(self, prompt):
        personal_prompt = f"""As {self.assistant_name}, a {self.assistant_role}, respond to the following:

{prompt}

Instructions:
1. Maintain the persona of {self.assistant_name}, the {self.assistant_role}.
2. If asked about your capabilities or identity, respond accordingly.
3. If you don't have specific information about past conversations, politely explain that you don't have access to that information.
4. Be helpful and friendly in your responses.

Response:"""

        personal_response = self.ask([{'role': 'user', 'content': personal_prompt}], temperature=0.5)
        logger.debug(f"Personal response: {personal_response}")
        
        return personal_response.strip()

    def save_memory(self, filename="conversation_history.gz"):
        """Append the current state of the memory to the existing file."""
        self.memory.append_to_file(filename)
        logger.info(f"Memory appended to {filename}")

    def load_memory(self, filename="conversation_history.gz"):
        """Load the state of the memory from a file."""
        self.memory.load_from_file(filename)
        logger.info(f"Memory loaded from {filename}")

# Create a global instance
llm_wrapper = LLMWrapper("llama_model/dolphin-2.9-llama3-8b-q8_0.gguf")

if __name__ == "__main__":
    # Load existing memory at the start
    llm_wrapper.load_memory()

    prompt = ""
    print(f"Processing prompt: {prompt}")
    response = llm_wrapper.generate_response(prompt)
    print("\nGenerated output:")
    print(response)

    print("\nDebug Information:")
    print(f"LLM initialized: {llm_wrapper.llm is not None}")