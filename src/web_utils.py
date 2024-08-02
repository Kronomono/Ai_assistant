import re
import logging
import requests
from llm_axe.core import internet_search, read_website

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_url(text):
    url_match = re.search(r'https?://\S+', text)
    return url_match.group() if url_match else None

def safe_read_website(url):
    try:
        return read_website(url)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error reading website {url}: {e}")
        return None

def process_specific_url(llm_instance, url, user_input, current_time):
    try:
        page_content = safe_read_website(url)
        if page_content:
            extracted_info = extract_information(llm_instance, page_content, user_input, current_time)
            logger.info(f"Extracted information from {url}: {extracted_info}")
            if extracted_info and "No relevant information found" not in extracted_info:
                return f"Based on {url}, {extracted_info}"
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
    return None

def enhanced_online_search(llm_instance, user_input, current_time):
    logger.info("Performing general search...")
    search_results = internet_search(user_input)
    if search_results:
        for result in search_results[:5]:
            response = process_specific_url(llm_instance, result.get('url', ''), user_input, current_time)
            if response:
                return response
    return None

def extract_information(llm_instance, content, question, current_time):
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

    return llm_instance.ask([{'role': 'user', 'content': prompt}], temperature=0.3).strip()