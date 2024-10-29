# This script will hold the schemas for the output we want
from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description='Critique of what is missing')
    superfluous: str = Field(description='Critique of what is superfluous') # Superfluous means unnecessary information
    suitability: str = Field(description='Critique of the suitability of the roadmap for the user')

class AnswerQuestion(BaseModel):
    """Answer the question"""
    answer: str = Field(description='Roadmap based on the given topic and user characteristics.')
    reflection: Reflection = Field(description='Your reflection on the initial answer')
    search_queries: List[str] = Field(description="3-5 search queries for researching improvements to address the critique of your current roadmap.")


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
