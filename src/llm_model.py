# t1.py
import json
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from TB_kg import Neo4jKnowledgeBase

class LLMClient:
    def __init__(self, api_key, base_url=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def llm_call(self, messages, model=LLM_MODEL, temperature=None, max_tokens=None,
                 response_format=None, expect_json=False):
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )

            content = response.choices[0].message.content.strip()

            if expect_json:
                try:
                    return json.loads(content)
                except Exception as e:
                    print(f"⚠️ JSON parsing failed: {e}\nContent: {content}")
                    return {}
            return content

        except Exception as e:
            print(f"⚠️ LLM call failed: {e}")
            return {}

llm = LLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

KB = Neo4jKnowledgeBase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
KB.client = llm
KB.LLM_MODEL = LLM_MODEL
KB.LLM_TEMPERATURE = LLM_TEMPERATURE
KB.LLM_MAX_TOKENS = LLM_MAX_TOKENS

CONVERSATION_HISTORY = []
