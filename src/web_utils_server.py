import os
import time
import logging
import re
import uvicorn
from typing import List, Dict
from duckduckgo_search import DDGS
from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import CosineStrategy
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel


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

# Dictionary to store last request times for each client
last_request_times = {}
last_rate_limit = {}

def get_rate_limit():
    limit = os.environ.get('ACCESS_PER_MIN', "5")
    return f"{limit}/minute"

# Custom rate limit exceeded handler
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    if request.client.host not in last_rate_limit or time.time() - last_rate_limit[request.client.host] > 60:
        last_rate_limit[request.client.host] = time.time()
    retry_after = 60 - (time.time() - last_rate_limit[request.client.host])
    reset_at = time.time() + retry_after
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded",
            "limit": str(exc.limit.limit),
            "retry_after": retry_after,
            'reset_at': reset_at,
            "message": f"You have exceeded the rate limit of {exc.limit.limit}."
        }
    )

app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)

# Middleware for token-based bypass and per-request limit
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        SPAN = int(os.environ.get('ACCESS_TIME_SPAN', 10))
        access_token = request.headers.get("access-token")
        if access_token == os.environ.get('ACCESS_TOKEN'):
            return await call_next(request)
        
        client_ip = request.client.host
        current_time = time.time()
        
        # Check time since last request
        if client_ip in last_request_times:
            time_since_last_request = current_time - last_request_times[client_ip]
            if time_since_last_request < SPAN:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Too many requests",
                        "message": f"Rate limit exceeded. Please wait {SPAN} seconds between requests.",
                        "retry_after": max(0, SPAN - time_since_last_request),
                        "reset_at": current_time + max(0, SPAN - time_since_last_request),
                    }
                )
        
        last_request_times[client_ip] = current_time
        return await call_next(request)

app.add_middleware(RateLimitMiddleware)

def extract_url(text: str) -> str:
    url_pattern = r'https?://\S+'
    match = re.search(url_pattern, text)
    return match.group(0) if match else None

def setup_cosine_strategy(query: str) -> CosineStrategy:
    return CosineStrategy(
        semantic_filter=query,
        word_count_threshold=20,
        max_dist=0.2,
        linkage_method='ward',
        top_k=3,
        model_name='BAAI/bge-small-en-v1.5'
    )

# Pydantic model for request validation
class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/search_and_crawl")
@limiter.limit(get_rate_limit())
async def search_and_crawl(request: Request, search_request: SearchRequest):
    logger.info(f"Processing query: {search_request.query}")
    
    crawler = WebCrawler()
    crawler.warmup()
    
    strategy = setup_cosine_strategy(search_request.query)
    
    url = extract_url(search_request.query)
    if url:
        logger.info(f"URL detected in query: {url}")
        urls = [url]
    else:
        logger.info("No URL detected. Performing DuckDuckGo search.")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(search_request.query, region='wt-wt', safesearch='off', max_results=search_request.max_results))
            urls = [result['href'] for result in results]
            logger.info(f"Search completed. Found {len(urls)} results.")
        except Exception as e:
            logger.error(f"Error performing DuckDuckGo search: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    crawled_results = []
    for url in urls:
        try:
            logger.info(f"Crawling URL: {url}")
            result = crawler.run(url=url, extraction_strategy=strategy)
            if result.success:
                crawled_results.append({"url": url, "content": result.extracted_content})
            else:
                logger.warning(f"Failed to crawl {url}")
            time.sleep(3)  # Rate limiting
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
    
    return JSONResponse(content={"results": crawled_results})

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
   run_server()