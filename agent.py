import os
import time
import json
from datetime import datetime
from typing import Optional

from openai import OpenAI
import mistralai

from models import AgentResponse, QuerySession, SearchResult
from tools import get_search_tool
from prompt_loader import PromptLoader


def _get_mistral_class():
    """Get the Mistral client class from mistralai with compatibility across package versions."""
    try:
        from mistralai import Mistral
        return Mistral
    except ImportError:
        from mistralai.client import Mistral
        return Mistral


def _import_instructor():
    """Import instructor after patching mistralai for compatibility with newer versions."""
    try:
        if not hasattr(mistralai, "Mistral"):
            mistralai.Mistral = _get_mistral_class()
    except Exception:
        pass

    import instructor
    return instructor

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
    
    def _init_client(self):
        """Initialize the LLM client with Instructor."""
        instructor = _import_instructor()

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
            MistralClient = _get_mistral_class()
            client = MistralClient(api_key=api_key)
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


    def ask(
        self, 
        question: str, 
        prompt_name: str = 'structured',
        use_search: bool = True
    ) -> QuerySession:
        """
        Ask a question and get a structured response with references.
        
        Args:
            question: The question to answer
            prompt_name: Name of the prompt template to use
            use_search: Whether to perform web search
            
        Returns:
            QuerySession with the response and metadata
        """
        start_time = time.time()
        
        prompt_template = self.prompt_loader.load_prompt(prompt_name)
        
        search_results: list[SearchResult] = []
        if use_search:
            search_results = self.search_tool.search(question)
        
        search_results_text = self.search_tool.format_results(search_results)
        
        user_prompt = prompt_template.format_user_prompt(
            question=question,
            search_results=search_results_text
        )
        
        messages = [
            {"role": "system", "content": prompt_template.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=AgentResponse,
            )
            
            session = QuerySession(
                question=question,
                prompt_used=prompt_name,
                model_used=f"{self.provider}/{self.model}",
                response=response,
                search_results=search_results,
                execution_time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            session = QuerySession(
                question=question,
                prompt_used=prompt_name,
                model_used=f"{self.provider}/{self.model}",
                raw_response=f"Error: {str(e)}",
                search_results=search_results,
                execution_time_seconds=time.time() - start_time
            )
        
        self.sessions.append(session)
        return session

    def ask_raw(
        self, 
        question: str, 
        prompt_name: str = 'structured',
        use_search: bool = True
    ) -> QuerySession:
        """
        Ask a question and get raw text response (no structured output).
        Useful for comparing structured vs unstructured responses.
        """
        start_time = time.time()
        
        prompt_template = self.prompt_loader.load_prompt(prompt_name)
        
        search_results: list[SearchResult] = []
        if use_search:
            search_results = self.search_tool.search(question)
        
        search_results_text = self.search_tool.format_results(search_results)
        
        user_prompt = prompt_template.format_user_prompt(
            question=question,
            search_results=search_results_text
        )
        
        messages = [
            {"role": "system", "content": prompt_template.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        raw_client = self._get_raw_client()
        
        try:
            response = raw_client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            raw_response = response.choices[0].message.content
            
            session = QuerySession(
                question=question,
                prompt_used=prompt_name,
                model_used=f"{self.provider}/{self.model}",
                raw_response=raw_response,
                search_results=search_results,
                execution_time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            session = QuerySession(
                question=question,
                prompt_used=prompt_name,
                model_used=f"{self.provider}/{self.model}",
                raw_response=f"Error: {str(e)}",
                search_results=search_results,
                execution_time_seconds=time.time() - start_time
            )
        
        self.sessions.append(session)
        return session

    def _get_raw_client(self):
        """Get raw client without Instructor wrapper."""
        if self.provider == 'openai':
            return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif self.provider == 'mistral':
            MistralClient = _get_mistral_class()
            return MistralClient(api_key=os.getenv('MISTRAL_API_KEY'))
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

def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n🔬 Research Agent - Prompt Testing Tool")
    print("="*50)
    
    provider = os.getenv('LLM_PROVIDER', 'mistral')
    
    try:
        agent = ResearchAgent(provider=provider)
        print(f"✅ Initialized with {provider}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Set required API keys in environment variables.")
        return
    
    print(f"\nAvailable prompts: {', '.join(agent.list_prompts())}")
    print("\nCommands:")
    print("  ask <prompt> <question> - Ask with specific prompt")
    print("  compare <question>      - Compare all prompts")
    print("  prompts                 - List prompts")
    print("  export                  - Export sessions")
    print("  quit                    - Exit")

    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                break
            
            elif user_input.lower() == 'prompts':
                for name in agent.list_prompts():
                    info = agent.prompt_loader.get_prompt_info(name)
                    print(f"  • {name} (v{info['version']})")
            
            elif user_input.lower() == 'export':
                agent.export_sessions()
            
            elif user_input.lower().startswith('compare '):
                question = user_input[8:].strip()
                if question:
                    results = agent.compare_prompts(question)
                    print("\n📊 COMPARISON RESULTS:")
                    for comp in results['comparisons']:
                        status = "✅" if comp['success'] else "❌"
                        refs = comp.get('num_references', 0)
                        time_s = comp.get('execution_time', 0)
                        print(f"  {status} {comp['prompt']}: {refs} refs, {time_s:.2f}s")
            
            elif user_input.lower().startswith('ask '):
                parts = user_input[4:].strip().split(' ', 1)
                if len(parts) == 2:
                    prompt_name, question = parts
                    if prompt_name in agent.list_prompts():
                        session = agent.ask(question, prompt_name)
                        agent.print_response(session)
                    else:
                        print(f"Unknown prompt: {prompt_name}")
                        print(f"Available: {', '.join(agent.list_prompts())}")
                else:
                    print("Usage: ask <prompt_name> <question>")
            
            else:
                session = agent.ask(user_input, 'structured')
                agent.print_response(session)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    interactive_mode()