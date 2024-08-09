import os
import time
import logging
import re
import uvicorn
import asyncio
from typing import Dict, List
from uuid import uuid4
from duckduckgo_search import DDGS
from crawl4ai import WebCrawler
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
import aiohttp
from aiohttp import ClientSession

# Load .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize rate limiter
def rate_limit_key_func(request: Request):
    access_token = request.headers.get("access-token")
    if access_token == os.environ.get('ACCESS_TOKEN'):
        return None
    return get_remote_address(request)

limiter = Limiter(key_func=rate_limit_key_func)
app.state.limiter = limiter

# Pydantic model for request validation
class SearchRequest(BaseModel):
    query: str
    max_results: int = 5

# Queue for managing requests
request_queue = asyncio.Queue()

# Dictionary to store request status and results
request_status: Dict[str, Dict] = {}

# Semaphore to control concurrent requests
sem = asyncio.Semaphore(1)  # Adjust this value based on your rate limit

async def process_queue():
    while True:
        request_id, request_data = await request_queue.get()
        request_status[request_id]['status'] = 'processing'
        async with sem:
            result = await process_request(request_data)
            request_status[request_id]['status'] = 'completed'
            request_status[request_id]['result'] = result
        request_queue.task_done()

async def process_request(request_data):
    query = request_data['query']
    max_results = request_data['max_results']
    
    crawler = WebCrawler()
    crawler.warmup()
    
    url = extract_url(query)
    if url:
        logger.info(f"URL detected in query: {url}")
        urls = [url]
    else:
        logger.info(f"No URL detected. Performing DuckDuckGo search for query: {query}")
        urls = await perform_search(query, max_results)
        
        if not urls:
            # Fallback: Try search without quotes
            fallback_query = query.replace('"', '')
            logger.info(f"No results found. Trying fallback search: {fallback_query}")
            urls = await perform_search(fallback_query, max_results)
            
           
    
    if not urls:
        logger.warning(f"No results found for any query attempts")
        return {"results": [], "message": "No results found"}
    
    crawled_results = []
    async with ClientSession() as session:
        for url in urls:
            try:
                logger.info(f"Crawling URL: {url}")
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        result = crawler.run(url=url, html_content=content)
                        if result.success:
                            crawled_results.append({"url": url, "content": result.extracted_content})
                            logger.info(f"Successfully crawled {url}")
                        else:
                            logger.warning(f"Failed to crawl {url}: {result.error}")
                    else:
                        logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                await asyncio.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}", exc_info=True)
    
    logger.info(f"Crawling completed. Number of results: {len(crawled_results)}")
    return {"results": crawled_results}

async def perform_search(query, max_results):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='wt-wt', safesearch='off', max_results=max_results))
        logger.info(f"DuckDuckGo search completed for '{query}'. Raw results: {results}")
        urls = [result['href'] for result in results]
        logger.info(f"Extracted URLs: {urls}")
        return urls
    except Exception as e:
        logger.error(f"Error performing DuckDuckGo search: {e}", exc_info=True)
        return []

def extract_url(text: str) -> str:
    url_pattern = r'https?://\S+'
    match = re.search(url_pattern, text)
    return match.group(0) if match else None

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/search_and_crawl")
@limiter.limit("5/minute")
async def search_and_crawl(request: Request, search_request: SearchRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received query: {search_request.query}")
    request_id = str(uuid4())
    await request_queue.put((request_id, {"query": search_request.query, "max_results": search_request.max_results}))
    request_status[request_id] = {'status': 'queued'}
    return JSONResponse(content={"message": "Request queued for processing", "request_id": request_id})

@app.get("/search_and_crawl/status/{request_id}")
async def get_status(request_id: str):
    if request_id not in request_status:
        raise HTTPException(status_code=404, detail="Request not found")
    
    status = request_status[request_id]['status']
    if status == 'completed':
        result = request_status[request_id]['result']
        del request_status[request_id]  # Clean up completed request
        return {"status": status, "result": result}
    elif status == 'queued':
        queue_position = list(request_status.keys()).index(request_id) + 1
        return {"status": status, "queue_position": queue_position}
    else:
        return {"status": status}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()