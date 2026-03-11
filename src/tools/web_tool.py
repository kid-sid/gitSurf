import os
import requests
from bs4 import BeautifulSoup
from typing import Optional

class WebSearchTool:
    """
    Allows the agent to search the web and fetch content from URLs.
    Useful for looking up documentation or solving version-specific errors.
    """

    def __init__(self, tavily_api_key: Optional[str] = None):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")

    def search(self, query: str, num_results: int = 5) -> str:
        """
        Performs a web search using Tavily API if available.
        """
        if not self.tavily_api_key:
            return "[Error] TAVILY_API_KEY not found. Web search is disabled."

        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": num_results
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for res in data.get("results", []):
                results.append(f"Title: {res['title']}\nURL: {res['url']}\nSnippet: {res['content']}\n")
            
            return "\n---\n".join(results)
        except Exception as e:
            return f"[Error] Web search failed: {e}"

    def fetch_url(self, url: str) -> str:
        """
        Fetches the content of a URL and converts it to basic text/markdown.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            text = soup.get_text(separator='\n')
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)
            
            return clean_text[:10000] 
        except Exception as e:
            return f"[Error] Failed to fetch URL {url}: {e}"

if __name__ == "__main__":
    # Simple Test
    tool = WebSearchTool()
    print("Testing WebSearchTool Fetch...")
    # Fetching a simple page
    content = tool.fetch_url("https://example.com")
    print(f"Content length: {len(content)}")
    print(f"Preview: {content[:100]}")
