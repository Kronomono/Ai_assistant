import time
import logging
import re
from typing import List, Dict
from duckduckgo_search import DDGS
from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import CosineStrategy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_url(text: str) -> str:
    url_pattern = r'https?://\S+'
    match = re.search(url_pattern, text)
    return match.group(0) if match else None

def setup_cosine_strategy(query: str) -> CosineStrategy:
    """
    Set up and return a CosineStrategy instance.
    
    Args:
        query (str): The search query to use as a semantic filter.
    
    Returns:
        CosineStrategy: Configured CosineStrategy instance.
    """
    return CosineStrategy(
        semantic_filter=query,
        word_count_threshold=10,
        max_dist=0.2,
        linkage_method='ward',
        top_k=3,
        model_name='BAAI/bge-small-en-v1.5'
    )

def search_and_crawl(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Check if the query contains a URL. If it does, crawl that URL.
    Otherwise, perform a DuckDuckGo search, then crawl the resulting URLs.
    Use CosineStrategy to extract relevant content.
    
    Args:
        query (str): The search query or URL.
        max_results (int): Maximum number of results to return for a search. Defaults to 5.
    
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing URLs and their extracted content.
    """
    logger.info(f"Processing query: {query}")
    
    crawler = WebCrawler()
    crawler.warmup()
    
    strategy = setup_cosine_strategy(query)
    
    url = extract_url(query)
    if url:
        logger.info(f"URL detected in query: {url}")
        urls = [url]
    else:
        logger.info("No URL detected. Performing DuckDuckGo search.")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region='wt-wt', safesearch='off', max_results=max_results))
            urls = [result['href'] for result in results]
            logger.info(f"Search completed. Found {len(urls)} results.")
        except Exception as e:
            logger.error(f"Error performing DuckDuckGo search: {e}")
            return []
    
    crawled_results = []
    for url in urls:
        try:
            logger.info(f"Crawling URL: {url}")
            result = crawler.run(url=url, extraction_strategy=strategy)
            if result.success:
                crawled_results.append({"url": url, "content": result.extracted_content})
            else:
                logger.warning(f"Failed to crawl {url}")
            time.sleep(5)  # Rate limiting
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
    
    return crawled_results

# Example usage
if __name__ == "__main__":
    # Test with a URL in the query
    url_query = "Hey Akane search up https://stackoverflow.com/questions/5575569/is-there-any-free-custom-search-api-like-google-custom-search"
    url_results = search_and_crawl(url_query)
    
    print("Results for URL query:")
    for result in url_results:
        print(f"URL: {result['url']}")
        print(result['content'])
        print("\n---\n")
    
    # Test with a regular search query
    search_query = "cosmog moves"
    search_results = search_and_crawl(search_query)
    
    print("Results for search query:")
    for result in search_results:
        print(f"URL: {result['url']}")
        print(result['content'])
        print("\n---\n")