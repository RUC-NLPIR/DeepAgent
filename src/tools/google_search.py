import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Union
import aiohttp
import asyncio
import chardet
import random


# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)

error_indicators = [
    'limit exceeded',
    'Error fetching',
    'Account balance not enough',
    'Invalid bearer token',
    'HTTP error occurred',
    'Error: Connection error occurred',
    'Error: Request timed out',
    'Unexpected error',
    'Please turn on Javascript',
    'Enable JavaScript',
    'port=443',
    'Please enable cookies',
]

invalid_search_queries = [
    "and end with",
    "search query",
    "query",
    "your query here",
    "your query",
    "your search query",
]


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 5000) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:1000000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def extract_text_from_url(
    url: str,
    use_jina: bool = False,
    jina_api_key: Optional[str] = None,
    snippet: Optional[str] = None,
    keep_links: bool = False
) -> tuple[str, str]:
    """
    Synchronous version of extract_text_from_url_async.
    Extract text from a URL (webpage or PDF). If a snippet is provided, extract the context related to it.
    Returns a tuple: (extracted_text, full_text)
    """
    try:
        if use_jina:
            # Jina extraction
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers, timeout=30)
            text = response.text
            if not keep_links:
                pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                text = re.sub(pattern, "", text)
            text = text.replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            if 'pdf' in url:
                # Use PDF handler; keep return signature consistent
                text = extract_pdf_text(url)
            else:
                response = requests.get(url, timeout=30)
                # Detect and handle encoding
                content_type = response.headers.get('content-type', '').lower()
                if 'charset' in content_type:
                    charset = content_type.split('charset=')[-1]
                    html = response.content.decode(charset, errors='replace')
                else:
                    content = response.content
                    try:
                        import chardet
                        detected = chardet.detect(content)
                        encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
                    except Exception:
                        encoding = 'utf-8'
                    html = content.decode(encoding, errors='replace')

                # Check for error indicators
                has_error = (any(indicator.lower() in html.lower() for indicator in error_indicators) and len(html.split()) < 64) or len(html) < 50 or len(html.split()) < 20
                if has_error:
                    error_msg = "Error extracting content: Content contains error indicators"
                    return error_msg, error_msg
                else:
                    try:
                        soup = BeautifulSoup(html, 'lxml')
                    except Exception:
                        soup = BeautifulSoup(html, 'html.parser')

                    if keep_links:
                        for element in soup.find_all(['script', 'style', 'meta', 'link']):
                            element.decompose()

                        text_parts = []
                        for element in soup.body.descendants if soup.body else soup.descendants:
                            if isinstance(element, str) and element.strip():
                                cleaned_text = ' '.join(element.strip().split())
                                if cleaned_text:
                                    text_parts.append(cleaned_text)
                            elif getattr(element, "name", None) == 'a' and element.get('href'):
                                href = element.get('href')
                                link_text = element.get_text(strip=True)
                                if href and link_text:
                                    if href.startswith('/'):
                                        base_url = '/'.join(url.split('/')[:3])
                                        href = base_url + href
                                    elif not href.startswith(('http://', 'https://')):
                                        href = url.rstrip('/') + '/' + href
                                    text_parts.append(f"[{link_text}]({href})")
                        text = ' '.join(text_parts)
                        text = ' '.join(text.split())
                    else:
                        text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            extracted_text = context if success else text[:10000]
            return extracted_text, text
        else:
            extracted_text = text[:10000]
            return extracted_text, text

    except Exception as e:
        error_msg = f"Error fetching {url}: {str(e)}"
        return error_msg, error_msg

def fetch_page_content(urls, max_workers=32, use_jina=False, jina_api_key=None, snippets: Optional[dict] = None, show_progress=False, keep_links=False):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        jina_api_key (str): API key for Jina.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.
        show_progress (bool): Whether to show progress bar with tqdm.
        keep_links (bool): Whether to keep links in the extracted text.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, jina_api_key, snippets.get(url) if snippets else None, keep_links): url
            for url in urls
        }
        completed_futures = concurrent.futures.as_completed(futures)
        if show_progress:
            completed_futures = tqdm(completed_futures, desc="Fetching URLs", total=len(urls))
            
        for future in completed_futures:
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            # time.sleep(0.1)  # Simple rate limiting
    return results


def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = full_text
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_relevant_info(search_results):
    """
    Extract relevant information from Google Serper search results.

    Args:
        search_results (dict): JSON response from the Google Serper API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    
    if 'webPages' in search_results and 'value' in search_results['webPages']:
        for id, result in enumerate(search_results['webPages']['value']):
            info = {
                'id': id + 1,  # Increment id for easier subsequent operations
                'title': result.get('name', ''),
                'link': result.get('link', ''),
                'date': result.get('date', '').split('T')[0],
                'snippet': result.get('snippet', ''),  # Remove HTML tags
                # Add context content to the information
                'context': ''  # Reserved field to be filled later
            }
            useful_info.append(info)
    
    return useful_info


class RateLimiter:
    def __init__(self, rate_limit: int, time_window: int = 60):
        """
        初始化速率限制器
        
        Args:
            rate_limit: 在时间窗口内允许的最大请求数
            time_window: 时间窗口大小(秒)，默认60秒
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """获取一个令牌，如果没有可用令牌则等待"""
        async with self.lock:
            while self.tokens <= 0:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.rate_limit,
                    self.tokens + (time_passed * self.rate_limit / self.time_window)
                )
                self.last_update = now
                if self.tokens <= 0:
                    await asyncio.sleep(random.randint(5, 30))  # 等待xxx秒后重试
            
            self.tokens -= 1
            return True

# 创建全局速率限制器实例
jina_rate_limiter = RateLimiter(rate_limit=130)  # 每分钟xxx次，避免报错

async def extract_text_from_url_async(url: str, session: aiohttp.ClientSession, use_jina: bool = False, 
                                    jina_api_key: Optional[str] = None, snippet: Optional[str] = None, 
                                    keep_links: bool = False) -> tuple[str, str]:
    """Async version of extract_text_from_url"""
    try:
        if use_jina:
            # 在调用jina之前获取令牌
            await jina_rate_limiter.acquire()
            
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            async with session.get(f'https://r.jina.ai/{url}', headers=jina_headers, ssl=False) as response:
                text = await response.text()
                if not keep_links:
                    pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                    text = re.sub(pattern, "", text)
                text = text.replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            if 'pdf' in url:
                # Use async PDF handling; keep return signature consistent
                text = await extract_pdf_text_async(url, session)
            else:
                async with session.get(url) as response:
                    # 检测和处理编码
                    content_type = response.headers.get('content-type', '').lower()
                    if 'charset' in content_type:
                        charset = content_type.split('charset=')[-1]
                        html = await response.text(encoding=charset)
                    else:
                        # 如果没有指定编码，先用bytes读取内容
                        content = await response.read()
                        # 使用chardet检测编码
                        detected = chardet.detect(content)
                        encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
                        html = content.decode(encoding, errors='replace')
                    
                    # 检查是否有错误指示
                    has_error = (any(indicator.lower() in html.lower() for indicator in error_indicators) and len(html.split()) < 64) or len(html) < 50 or len(html.split()) < 20
                    if has_error:
                        error_msg = "Error extracting content: Content contains error indicators"
                        return error_msg, error_msg
                    else:
                        try:
                            soup = BeautifulSoup(html, 'lxml')
                        except Exception:
                            soup = BeautifulSoup(html, 'html.parser')

                        if keep_links:
                            # Similar link handling logic as in synchronous version
                            for element in soup.find_all(['script', 'style', 'meta', 'link']):
                                element.decompose()

                            text_parts = []
                            for element in soup.body.descendants if soup.body else soup.descendants:
                                if isinstance(element, str) and element.strip():
                                    cleaned_text = ' '.join(element.strip().split())
                                    if cleaned_text:
                                        text_parts.append(cleaned_text)
                                elif element.name == 'a' and element.get('href'):
                                    href = element.get('href')
                                    link_text = element.get_text(strip=True)
                                    if href and link_text:
                                        if href.startswith('/'):
                                            base_url = '/'.join(url.split('/')[:3])
                                            href = base_url + href
                                        elif not href.startswith(('http://', 'https://')):
                                            href = url.rstrip('/') + '/' + href
                                        text_parts.append(f"[{link_text}]({href})")

                            text = ' '.join(text_parts)
                            text = ' '.join(text.split())
                        else:
                            text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            extracted_text = context if success else text[:10000]
            return extracted_text, text
        else:
            extracted_text = text[:10000]
            return extracted_text, text

    except Exception as e:
        error_msg = f"Error fetching {url}: {str(e)}"
        return error_msg, error_msg

async def fetch_page_content_async(urls: List[str], use_jina: bool = False, jina_api_key: Optional[str] = None, 
                                 snippets: Optional[Dict[str, str]] = None, show_progress: bool = False,
                                 keep_links: bool = False, max_concurrent: int = 32) -> Dict[str, tuple[str, str]]:
    """Asynchronously fetch content from multiple URLs."""
    # Robustness: accept a single URL string
    if isinstance(urls, str):
        urls = [urls]
    async def process_urls():
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=240)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for url in urls:
                task = extract_text_from_url_async(
                    url, 
                    session, 
                    use_jina, 
                    jina_api_key,
                    snippets.get(url) if snippets else None,
                    keep_links
                )
                tasks.append(task)
            
            if show_progress:
                results = []
                for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching URLs"):
                    result = await task
                    results.append(result)
            else:
                results = await asyncio.gather(*tasks)
            
            return {url: result for url, result in zip(urls, results)}  # 返回字典而不是协程对象

    return await process_urls()  # 确保等待异步操作完成

async def extract_pdf_text_async(url: str, session: aiohttp.ClientSession) -> str:
    """
    Asynchronously extract text from a PDF.

    Args:
        url (str): URL of the PDF file.
        session (aiohttp.ClientSession): Aiohttp client session.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        async with session.get(url, timeout=30) as response:  # Set timeout to 20 seconds
            if response.status != 200:
                return f"Error: Unable to retrieve the PDF (status code {response.status})"
            
            content = await response.read()
            
            # Open the PDF file using pdfplumber
            with pdfplumber.open(BytesIO(content)) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text
            
            # Limit the text length
            cleaned_text = full_text
            return cleaned_text
    except asyncio.TimeoutError:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def google_serper_search(query: str, api_key: str, timeout: int = 5):
    """
    Perform a search using the Google Serper API.

    Args:
        query (str): Search query.
        api_key (str): API key for Google Serper API.
        timeout (int or float or tuple): Request timeout in seconds.

    Returns:
        dict: JSON response of the search results. Returns empty dict if request fails.
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=timeout)
            response.raise_for_status()  # Raise exception if the request failed
            search_results = response.json()
            relevant_info = extract_relevant_info_serper(search_results)
            return relevant_info
        except Timeout:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Google Serper API request timed out ({timeout} seconds) for query: {query} after {max_retries} retries")
                return {}
            print(f"Google Serper API Timeout occurred, retrying ({retry_count}/{max_retries})...")
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Google Serper API Request Error occurred: {e} after {max_retries} retries")
                return {}
            print(f"Google Serper API Request Error occurred, retrying ({retry_count}/{max_retries})...")
        time.sleep(1)  # Wait 1 second between retries
    
    return {}

def extract_relevant_info_serper(search_results):
    """
    Extract relevant information from Google Serper search results.

    Args:
        search_results (dict): JSON response from the Google Serper API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    if 'organic' in search_results:
        for i, result in enumerate(search_results['organic']):
            # Try to extract domain for site_name, or leave empty
            site_name = ''
            try:
                from urllib.parse import urlparse
                site_name = urlparse(result.get('link', '')).netloc
            except Exception:
                pass

            info = {
                'id': i + 1,
                'title': result.get('title', ''),
                'url': result.get('link', ''),
                'site_name': site_name, # Serper doesn't directly provide siteName, try to parse from URL
                'date': result.get('date', ''), # Serper might not always provide date
                'snippet': result.get('snippet', ''),
            }
            useful_info.append(info)
    return useful_info



async def google_serper_search_async(query: str, api_key: str, timeout: int = 20):
    """
    Perform an asynchronous search using the Google Serper API.

    Args:
        query (str): Search query.
        api_key (str): API key for Google Serper API.
        timeout (int): Request timeout in seconds for each attempt.

    Returns:
        dict: JSON response of the search results. Returns empty dict if all retries fail.
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers_serper = {  # Use a different name to avoid conflict with global headers
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    max_retries = 5
    retry_count = 0
    
    # Create a timeout object for aiohttp
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession() as session:
        while retry_count < max_retries:
            try:
                async with session.post(url, headers=headers_serper, data=payload, timeout=client_timeout, ssl=False) as response:
                    response.raise_for_status()  # Raise AIOHTTPError for bad status (4xx or 5xx)
                    search_results = await response.json()
                    relevant_info = extract_relevant_info_serper(search_results)
                    return relevant_info
            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Google Serper API request timed out ({timeout} seconds) for query: {query} after {max_retries} retries")
                    return {}
                print(f"Google Serper API Timeout occurred, retrying ({retry_count}/{max_retries})...")
            except aiohttp.ClientError as e: # Covers ConnectionError, ClientResponseError, etc.
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Google Serper API Request Error occurred: {e} after {max_retries} retries")
                    return {}
                print(f"Google Serper API Request Error occurred ({e}), retrying ({retry_count}/{max_retries})...")
            
            if retry_count < max_retries:
                await asyncio.sleep(1)  # Wait 1 second between retries (non-blocking)
    
    return {}


def get_openai_function_web_search() -> dict:
    """Return the OpenAI tool/function definition for web_search."""
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform a web search and return the raw search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string."
                    }
                },
                "required": ["query"]
            }
        }
    }


def get_openai_function_browse_pages() -> dict:
    """Return the OpenAI tool/function definition for browse_pages."""
    return {
        "type": "function",
        "function": {
            "name": "browse_pages",
            "description": "Fetch the content of multiple webpages or PDFs. Returns a mapping from page URL to page content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "description": "A list of URLs to fetch content from.",
                        "items": {"type": "string", "format": "uri"},
                        "minItems": 1
                    }
                },
                "required": ["urls"]
            }
        }
    }

import asyncio

async def main():
    # Example usage
    # Define the query to search
    query = "Structure of dimethyl fumarate"

    # Set your API key for Google Serper API
    SERPER_API_KEY = "your_serper_api_key_here"
    JINA_API_KEY = "your_jina_api_key_here"

    print("Performing Google Serper Search...")
    # search_results = await google_serper_search_async(query, SERPER_API_KEY)
    search_results = google_serper_search(query, SERPER_API_KEY)
    print(search_results)
    
    if not search_results:
        print("No search results to process.")
        return

    print("Fetching and extracting context for each snippet...")
    # Build a single list of URLs and fetch once
    urls = [info['url'] for info in search_results]
    # full_texts = await fetch_page_content_async(urls, use_jina=True, jina_api_key=JINA_API_KEY, show_progress=True)
    full_texts = fetch_page_content(urls, use_jina=False, jina_api_key=JINA_API_KEY, show_progress=True)
    for url, text in full_texts.items():
        print('---')
        print(url, text[:1000])

if __name__ == "__main__":
    asyncio.run(main())
