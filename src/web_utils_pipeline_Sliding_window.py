import requests
import json
import time
import logging
import os
from crawl4ai.chunking_strategy import SlidingWindowChunking
from hyperdb import HyperDB, get_embedding

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a directory for storing website chunks
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), 'website_chunks')
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Initialize HyperDB for storing website chunks
website_chunks_db = HyperDB(embedding_function=get_embedding)

def information_dump(query, max_results=3):
    url = "http://localhost:8000/search_and_crawl"
    payload = {
        "query": query,
        "max_results": max_results
    }
    headers = {
        "Content-Type": "application/json"
    }

    logger.info(f"Sending request to {url} with payload: {payload}")
    response = requests.post(url, json=payload, headers=headers)
    logger.info(f"Received response with status code: {response.status_code}")
    logger.debug(f"Response content: {response.text}")

    if response.status_code == 200:
        data = response.json()
        request_id = data.get('request_id')
        if request_id:
            logger.info(f"Request queued successfully. Request ID: {request_id}")
            return poll_for_results(url, request_id)
        else:
            logger.error("Error: No request_id received")
            return None
    else:
        logger.error(f"Error: {response.status_code} - {response.text}")
        return None

def poll_for_results(base_url, request_id):
    while True:
        status_url = f"{base_url}/status/{request_id}"
        logger.info(f"Polling for results at {status_url}")
        response = requests.get(status_url)
        logger.debug(f"Received response with status code: {response.status_code}")
        logger.debug(f"Response content: {response.text}")

        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'completed':
                logger.info("Request completed. Returning results.")
                return data['result']
            elif data['status'] == 'queued':
                logger.info(f"Request still in queue. Position: {data.get('queue_position', 'unknown')}")
            elif data['status'] == 'processing':
                logger.info("Request is being processed.")
            else:
                logger.warning(f"Unexpected status: {data['status']}")
        else:
            logger.error(f"Error checking status: {response.status_code} - {response.text}")
        time.sleep(5)  # Wait for 5 seconds before polling again

def reword_question(llm_instance, question):
    prompt = f"""Reword the following question into a concise search query suitable for an information retrieval system. 
    Focus on extracting key concepts and entities, removing unnecessary words, and formatting it as a short, direct query.

    Original question: "{question}"

    Instructions:
    1. Identify the main topic and any specific entities mentioned.
    2. Remove filler words, pronouns, and conversational language.
    3. Keep only essential adjectives or qualifiers that narrow down the search.
    4. Format the result as 3-6 words if possible, but prioritize including all key information.
    5. Do not add any information not present in the original question.
    6. Respond only with the reworded query, nothing else.

    Reworded query:"""

    reworded_query = llm_instance.ask([{'role': 'user', 'content': prompt}], temperature=0.3).strip()
    return reworded_query

def chunk_and_store_content(content):
    chunker = SlidingWindowChunking(window_size=200, step=100)  # Adjust these values as needed
    chunks = []
    
    if isinstance(content, dict) and 'results' in content:
        results = content['results']
    elif isinstance(content, list):
        results = content
    else:
        logger.error(f"Unexpected content structure: {type(content)}")
        return

    for result in results:
        text_chunks = chunker.chunk(result['content'])
        for chunk in text_chunks:
            chunks.append({
                'chunk': chunk,
                'url': result['url']
            })
    
    website_chunks_db.add(chunks)
    
    # Save the chunks to a file
    file_path = os.path.join(CHUNKS_DIR, f"chunks_{int(time.time())}.gz")
    website_chunks_db.save(file_path)

def query_website_chunks(query, top_k=10):
    # Load all chunk files
    for file_name in os.listdir(CHUNKS_DIR):
        file_path = os.path.join(CHUNKS_DIR, file_name)
        website_chunks_db.load(file_path)
    
    # Query the loaded chunks
    results = website_chunks_db.query(query, top_k=top_k)
    return results

def extract_information(llm_instance, question, current_time, max_results=3):
    reworded_query = reword_question(llm_instance, question)
    print(f"Reworded query: {reworded_query}")
    
    content = information_dump(reworded_query, max_results)
    if not content:
        return "Failed to retrieve information."
    
    if isinstance(content, list) and len(content) == 0:
        return "No relevant information found."
    
    # Chunk and store the content
    chunk_and_store_content(content)
    
    # Query the stored chunks
    relevant_info = query_website_chunks(reworded_query, top_k=10)
    
    if not relevant_info:
        return "No relevant information found across all chunks."
    
    # Combine relevant chunks
    combined_info = "\n".join([info[0]['chunk'] for info in relevant_info])
    
    # Final summarization
    summary_prompt = f"""Provide a comprehensive answer to the question based on the following extracted information: "{question}"
    Extracted Information:
    {combined_info}
    Current date and time: {current_time}
    Instructions:
    1. Include ALL relevant details from the extracted information in your answer.
    2. Ensure you mention ALL types, categories, or classifications if they are present in the information.
    3. If there are multiple pieces of information, combine them into a coherent answer.
    4. If the information is incomplete or uncertain, state that clearly.
    5. Use the current date and time information when relevant to the query.
    6. Make sure to directly and fully address the main focus of the question in your answer.
    7. If there's any contradiction in the information, mention it and provide all versions.
    8. Do not include or mention the source URLs in your answer.
    Comprehensive Answer:"""

    try:
        final_answer = llm_instance.ask([{'role': 'user', 'content': summary_prompt}], temperature=0.3).strip()
    except Exception as e:
        print(f"Error generating final answer: {e}")
        final_answer = "Unable to generate a final answer due to content length. Please try asking about specific aspects of the topic."

    # Add source information
    used_urls = set(info[0]['url'] for info in relevant_info)
    sources = "\n\nSources:\n" + "\n".join(used_urls)
    return final_answer + sources

# You can add any additional utility functions or classes here if needed