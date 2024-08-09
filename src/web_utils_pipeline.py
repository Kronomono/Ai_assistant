import requests
import json
import time
import logging
from crawl4ai.chunking_strategy import SlidingWindowChunking

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

def chunk_content(content, window_size=500, step=250):
    chunker = SlidingWindowChunking(window_size=window_size, step=step)
    all_chunks = []
    for result in content['results']:
        chunks = chunker.chunk(result['content'])
        for chunk in chunks:
            all_chunks.append({
                "url": result['url'],
                "chunk": chunk
            })
    return all_chunks

def extract_information(llm_instance, question, current_time, max_results=3):
    reworded_query = reword_question(llm_instance, question)
    print(f"Reworded query: {reworded_query}")
    
    content = information_dump(reworded_query, max_results)
    if not content:
        return "Failed to retrieve information."
    
    if not content.get('results'):
        return "No relevant information found."
    
    chunks = chunk_content(content)
    total_chunks = len(chunks)
    print(f"Total chunks: {total_chunks}")
    
    all_relevant_info = []
    used_urls = set()
    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{total_chunks}")
        prompt = f"""Extract ALL relevant information from the following content to answer this question: "{question}"
        Content chunk from {chunk['url']}:
        {chunk['chunk']}
        Current date and time: {current_time}
        Instructions:
        1. Extract ALL factual information that could be relevant to the question, even if it seems redundant.
        2. Include ALL specific details such as types, categories, numbers, names, or other relevant data mentioned.
        3. Pay special attention to the first few sentences of the content, as they often contain key information.
        4. If multiple pieces of information are found, list them all separately.
        5. If no relevant information is found in this chunk, state "No relevant information found in this chunk."
        6. Do not infer or generate information not present in the content.
        7. Use the current date and time information when relevant to the query.
        Extracted Information:"""

        try:
            chunk_info = llm_instance.ask([{'role': 'user', 'content': prompt}], temperature=0.3).strip()
            if "No relevant information found in this chunk" not in chunk_info:
                all_relevant_info.append(chunk_info)
                used_urls.add(chunk['url'])
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue
    
    if not all_relevant_info:
        return "No relevant information found across all chunks."
    
    # Combine all relevant information
    combined_info = "\n".join(all_relevant_info)
    
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
    sources = "\n\nSources:\n" + "\n".join(used_urls)
    return final_answer + sources

# You can add any additional utility functions or classes here if needed