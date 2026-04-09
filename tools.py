import os
from models import SearchResult

try:
    from duckduckgo_search import DDGS
    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

class WebSearchTool:
    """Web search tool using DuckDuckGo."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> list[SearchResult]:
        """Search the web for the given query
        Returns a list of SearchResult objects."""

        if not HAS_DUCKDUCKGO:
            print("Warning: duckduckgo-search not installed. Using mock results.")
            return self._mock_search(query)
        
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    results.append(SearchResult(
                        title=r.get("title","No title"),
                        url=r.get("href",r.get("link","")),
                        snippet=r.get("body",r.ge)
                    ))
            return results
        except Exception as e:
            print(f"Search error:{e}")
            return self._mock_search(query)

    def _mock_search(self, query: str) -> list[SearchResult]:
        """Return mock search results for testing."""
        return [
            SearchResult(
                title=f"Wikipedia: {query}",
                url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                snippet=f"This is a Wikipedia article about {query}."
            ),
            SearchResult(
                title=f"Britannica: {query} Overview",
                url=f"https://www.britannica.com/topic/{query.replace(' ', '-')}",
                snippet=f"Encyclopedic overview of {query} covering key aspects."
            ),
            SearchResult(
                title=f"Science Direct: {query} Research",
                url=f"https://www.sciencedirect.com/topics/{query.replace(' ', '-')}",
                snippet=f"Academic research and studies related to {query}."
            ),
        ]

class SerpAPITool:
    """Alternative web search tool using SerpAPI (requires API key).
    Set SERPAPI_KEY environment variable."""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.api_key = os.getenv("SERPAPI_KEY")

    def search(self, query: str) -> list[SearchResult]:
        """Search the web using SerpAPI."""
        if not self.api_key:
            print("Warning: SERPAPI_KEY not set. Cannot use SerpAPI.")
            return []
        
        if not HAS_HTTPX:
            print("Warning: httpx not installed. Cannot use SerpAPI.")
            return []

        try:
            params = {
                'q': query,
                'api_key': self.api_key,
                'num': self.max_results
            }
            response = httpx.get('https://serpapi.com/search', params=params)
            data = response.json()
            
            results = []
            for r in data.get('organic_results', [])[:self.max_results]:
                results.append(SearchResult(
                    title=r.get('title', ''),
                    url=r.get('link', ''),
                    snippet=r.get('snippet', '')
                ))
            return results
        except Exception as e:
            print(f"SerpAPI error: {e}")
            return []
        
class TavilySearchTool:
    """Search using Tavily API (Optimized for AI/LLM use)
    Set TAVILY_API_KEY environment variable."""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.api_key = os.getenv("TAVILY_API_KEY")

    def search(self, query: str) -> list[SearchResult]:
        """Search the web using Tavily API."""
        if not self.api_key:
            print("Warning: TAVILY_API_KEY not set. Cannot use Tavily.")
            return []
        
        if not HAS_HTTPX:
            print("Warning: httpx not installed. Cannot use Tavily.")
            return []

        try:
            response = httpx.post(
                'https://api.tavily.com/search',
                json={
                    'api_key': self.api_key,
                    'query': query,
                    'max_results': self.max_results,
                    'include_answer': False
                }
            )
            data = response.json()
            
            results = []
            for r in data.get('results', []):
                results.append(SearchResult(
                    title=r.get('title', ''),
                    url=r.get('url', ''),
                    snippet=r.get('content', '')[:300]
                ))
            return results
        except Exception as e:
            print(f"Tavily error: {e}")
            return []

def get_search_tool(provider: str = 'duckduckgo', max_results: int = 5):
    """Factory function to get the appropriate search tool."""
    providers = {
        'duckduckgo': WebSearchTool,
        'serpapi': SerpAPITool,
        'tavily': TavilySearchTool,
    }
    
    tool_class = providers.get(provider.lower(), WebSearchTool)
    return tool_class(max_results=max_results)
