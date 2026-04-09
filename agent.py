import os
import time
import json
from datetime import datetime
from typing import Optional

import instructor
from openai import OpenAI
from mistralai import Mistral

from models import AgentResponse, QuerySession, SearchResult
from tools import get_search_tool
from prompt_loader import PromptLoader

class ResearchAgent:
    """Agentic research assistant that answers questions using web search.
    Supports multiple LLM providers and prompt templates for comparison."""

    SUPPORTED_PROVIDERS = ["openai", "mistral", "ollama"]

    def __init__(
            self,
            provider: str = "mistral",
            model: Optional[str] = None,
            search_provider: str = "duckduckgo",
            max_search_results: int = 5,
    ):
        self.provider = provider.lower()
        self.model = model or self._default_model()
        self.prompt_loader = PromptLoader()
        self.search_tool = get_search_tool(search_provider, max_search_results)
        self.client = self._init_client()
        self.sessions: list[QuerySession] = []

    def _default_model(self) -> str:
        """Get default model based on provider."""
        defaults = {
            "openai": "gpt-oss",
            "mistral": "mistral-large-latest",
            "ollama": "llama3.2"
        }
        return defaults.get(self.provider, "mistral-large-latest")
    
    def __init__client(self):
        """Initialize the LLM client with Instructor"""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            client = OpenAI(api_key=api_key)
            return instructor.from_openai(client)
        
        elif self.provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable not set")
            client = Mistral(api_key=api_key)
            return instructor.from_mistral(client)
        
        elif self.provider == "ollama":
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            return instructor.from_openai(client)
    
    def list_prompts(self) -> list[str]:
        """List available prompt templates."""
        return self.prompt_loader.list_prompts()

    #/*
    # TODO: Hel[per function for promting - strucured or raw responses
    #*/

    def _get_raw_client(self):
        """Get raw client without Instructor wrapper."""
        if self.provider == 'openai':
            return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif self.provider == 'mistral':
            return Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
        elif self.provider == 'ollama':
            return OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    

    def compare_prompts(
        self, 
        question: str, 
        prompt_names: Optional[list[str]] = None
    ) -> dict:
        """
        Compare multiple prompts on the same question.
        
        Args:
            question: The question to test
            prompt_names: List of prompt names to compare (default: all)
            
        Returns:
            Comparison results dictionary
        """
        if prompt_names is None:
            prompt_names = self.list_prompts()
        
        results = {
            'question': question,
            'model': f"{self.provider}/{self.model}",
            'timestamp': datetime.now().isoformat(),
            'comparisons': []
        }
        
        for prompt_name in prompt_names:
            print(f"Testing prompt: {prompt_name}...")
            session = self.ask(question, prompt_name)
            
            comparison = {
                'prompt': prompt_name,
                'execution_time': session.execution_time_seconds,
                'success': session.response is not None,
            }
            
            if session.response:
                comparison['answer_length'] = len(session.response.answer)
                comparison['num_references'] = len(session.response.references)
                comparison['confidence'] = session.response.confidence
                comparison['reference_urls'] = [
                    ref.url for ref in session.response.references
                ]
            else:
                comparison['error'] = session.raw_response
            
            results['comparisons'].append(comparison)
        
        return results

    def export_sessions(self, filepath: str = 'sessions.json'):
        """Export all sessions to a JSON file."""
        data = [session.model_dump() for session in self.sessions]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Exported {len(self.sessions)} sessions to {filepath}")

    def print_response(self, session: QuerySession):
        """Pretty print a session response."""
        print("\n" + "="*60)
        print(f"Question: {session.question}")
        print(f"Prompt: {session.prompt_used} | Model: {session.model_used}")
        print(f"Time: {session.execution_time_seconds:.2f}s")
        print("="*60)
        
        if session.response:
            print(f"\n📝 ANSWER:\n{session.response.answer}")
            print(f"\n⭐ Confidence: {session.response.confidence}")
            
            if session.response.key_points:
                print("\n🔑 KEY POINTS:")
                for point in session.response.key_points:
                    print(f"  • {point}")
            
            print("\n📚 REFERENCES:")
            for i, ref in enumerate(session.response.references, 1):
                print(f"  [{i}] {ref.title}")
                print(f"      {ref.url}")
        else:
            print(f"\n❌ Error or raw response:\n{session.raw_response}")
        
        print("\n" + "="*60)

