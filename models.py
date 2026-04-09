from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
import uuid

class Reference(BaseModel):
    """
    A single reference with unique URL
    """
    title: str = Field(..., description="The title of the reference")
    url: str = Field(..., description="The unique URL of the reference")
    snippet: Optional[str] = Field(None, description="A brief snippet or summary of the reference")

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class AgentResponse(BaseModel):
    """
    Structured respons from the research agent.
    """
    answer: str = Field(..., description="The comprehensive answer to the question")
    confidence: str = Field(
        default="medium",
        description="The confidence level of the answer (low, medium, high)"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Key takeaways as bullet points"
    )
    reference: list[Reference] = Field(
        ...,
        min_length=1,
        description="List of unique references supporting the answer"
    )

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        valid = ["high", "medium", "low"]
        if v.lower() not in valid:
            return "medium"
        return v.lower()
    
    @field_validator('references')
    @classmethod
    def validate_references(cls, v: list[Reference]) -> list[Reference]:
        urls = [ref.url for ref in v]
        if len(urls) != len(set(urls)):
            raise ValueError('All reference URLs must be unique')
        return v

class SearchResult(BaseModel):
    """A single search result from web search."""
    title: str
    url: str
    snippet: str
    
    def __str__(self) -> str:
        return f"[{self.title}]({self.url})\n{self.snippet}"

class QuerySession(BaseModel):
    """Tracks a query session with metadata."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str
    prompt_used: str
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.now)
    response: Optional[AgentResponse] = None
    raw_response: Optional[str] = None
    search_results: Optional[list[SearchResult]] = Field(default_factory=list)
    execution_time_seconds: Optional[float] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
class PromptTemplate(BaseModel):
    """Parsed prompt template from XML file"""
    name: str
    version: str
    system_prompt: str
    user_template: str
    file_path: str

    def format_user_prompt(self, question: str, search_results: str) -> str:
        """Format the user template with provided values"""
        return self.user_template.format(question=question, search_results=search_results)

class ComparisonResult(BaseModel):
    """Result of comparing multiple prompts/models."""
    question: str
    sessions: list[QuerySession]
    best_prompt: Optional[str] = None
    notes: Optional[str] = None
    