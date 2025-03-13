import os
from pydantic_ai.models.openai import OpenAIModel

OPENAI_MODEL = OpenAIModel(model_name='gpt-4o-2024-11-20')

OLLAMA_MODEL = OpenAIModel(model_name='llama3-groq-tool-use', base_url='http://localhost:11434/v1')
