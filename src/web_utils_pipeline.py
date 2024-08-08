import requests
import json


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

def extract_information(llm_instance, question, current_time, max_results=3):
    reworded_query = reword_question(llm_instance,question)
    content = information_dump(reworded_query, 3)
    
    prompt = f"""Extract the most relevant information from the following content to answer this question: "{question}"
    Content:
    {json.dumps(content, indent=2)[:5000]}
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
