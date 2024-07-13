import re
import json
import logging
import requests
from llama_cpp import Llama
from llm_axe.core import internet_search, read_website

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

    def initialize(self):
        if self.llm is None:
            self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_batch=512, n_gpu_layers=-1, verbose=False)
            self.llama_wrapper = LlamaWrapper(self.llm)

    def ask(self, prompts: list, format: str = "", temperature: float = 0.7):
        self.initialize()
        return self.llama_wrapper.ask(prompts, format, temperature)

    def classify_query(self, query):
        classification_prompt = f"""
        Classify the following query as 'online' or 'local' based on these criteria:

        Online:
        - Requires up-to-date information (e.g., current events, weather, stock prices)
        - Asks for specific web content or links
        - Requires information about recent or upcoming events
        - Involves searching for or comparing products or services
        - Needs information about a specific person, place, or thing that may not be in a general knowledge base

        Local:
        - General knowledge questions
        - Mathematical calculations or problem-solving
        - Language-related tasks (translation, grammar, etc.)
        - Coding help or explanations
        - Hypothetical scenarios or creative tasks
        - Personal advice or opinions (that don't require current data)

        Query: "{query}"

        Classification (online/local):
        Explanation:
        """
        response = self.ask([{'role': 'user', 'content': classification_prompt}], temperature=0.3).strip()
        
        # Extract classification and explanation
        classification = 'local'  # Default to local
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
        if self.llm is None:
            self.initialize()

        query_type, explanation = self.classify_query(user_input)

        if query_type == "online":
            logger.info(f"Online query detected: {explanation}")
            return self.perform_online_search(user_input)
        else:
            logger.info(f"Local query detected: {explanation}")
            return self.perform_local_query(user_input)

    def perform_local_query(self, user_input):
        local_response = self.get_local_knowledge_response(user_input)
        
        if "I don't have enough reliable information" not in local_response:
            logger.info("Local knowledge response found.")
            return f"[Using local knowledge] {local_response}"

        logger.info("Local knowledge response insufficient, attempting online search...")
        return self.perform_online_search(user_input)

    def perform_online_search(self, user_input):
        try:
            url = self.extract_url(user_input)
            online_response = self.process_specific_url(url, user_input) if url else self.enhanced_online_search(user_input)
            
            if online_response:
                logger.info("Valid online information found.")
                return f"[Using online search] {online_response}"
            else:
                logger.info("No valid online information found.")
        except Exception as e:
            logger.error(f"Error during online search: {e}", exc_info=True)
        return "I couldn't retrieve information from the internet."

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

    def get_local_knowledge_response(self, user_input):
        local_prompt = f"""You are a knowledgeable assistant with expertise in various fields. Please answer the following question to the best of your ability:

Question: {user_input}

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

# Create a global instance
llm_wrapper = LLMWrapper("llama_model/dolphin-2.9-llama3-8b-q8_0.gguf")

if __name__ == "__main__":
    prompt = "You are to role play as Akane Kurokawa from oshi no ko. I am your biggest fan asking for an autograph and picture. Please respond."
    print(f"Processing prompt: {prompt}")
    response = llm_wrapper.generate_response(prompt)
    print("\nGenerated output:")
    print(response)

    print("\nDebug Information:")
    print(f"LLM initialized: {llm_wrapper.llm is not None}")