import requests
import json
from crawl4ai.chunking_strategy import SlidingWindowChunking

def information_dump(query, max_results=3):
    url = "http://localhost:8000/search_and_crawl"
    payload = {
        "query": query,
        "max_results": max_results
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()

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
    content = information_dump(reworded_query, max_results)
    chunks = chunk_content(content)
    
    all_relevant_info = []
    for chunk in chunks:
        prompt = f"""Extract the most relevant information from the following content to answer this question: "{question}"
        Content chunk from {chunk['url']}:
        {chunk['chunk']}
        Current date and time: {current_time}
        Instructions:
        1. Extract only factual information directly related to the question.
        2. If you find relevant information, state it concisely and completely.
        3. Include specific details such as numbers, names, or other relevant data if mentioned.
        4. Pay special attention to the first few sentences of the content, as they often contain key information.
        5. If no relevant information is found in this chunk, state "No relevant information found in this chunk."
        6. Do not infer or generate information not present in the content.
        7. Use the current date and time information when relevant to the query.
        Extracted Information:"""

        try:
            chunk_info = llm_instance.ask([{'role': 'user', 'content': prompt}], temperature=0.3).strip()
            if "No relevant information found in this chunk" not in chunk_info:
                all_relevant_info.append(chunk_info)
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue

        # Break the loop if we have found enough relevant information
        if len(all_relevant_info) >= 3:
            break
    
    if not all_relevant_info:
        return "No relevant information found across all chunks."
    
    # Combine all relevant information
    combined_info = "\n".join(all_relevant_info)
    
    # Final summarization
    summary_prompt = f"""Summarize the following extracted information to answer the question: "{question}"
    Extracted Information:
    {combined_info}
    Current date and time: {current_time}
    Instructions:
    1. Provide a concise and complete answer based on the extracted information.
    2. Include specific details such as numbers, names, or other relevant data if mentioned.
    3. If the information is incomplete or uncertain, state that clearly.
    4. Use the current date and time information when relevant to the query.
    5. Make sure to directly address the main focus of the question in your answer.
    Final Answer:"""

    try:
        final_answer = llm_instance.ask([{'role': 'user', 'content': summary_prompt}], temperature=0.3).strip()
    except Exception as e:
        print(f"Error generating final answer: {e}")
        final_answer = "Unable to generate a final answer due to content length. Please try asking about specific aspects of the topic."

    return final_answer