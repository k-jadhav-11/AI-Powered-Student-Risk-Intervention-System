
import os
from typing import Iterator, Literal

import tiktoken
from dotenv import load_dotenv

load_dotenv()

Provider = Literal["openai", "gemini"]

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# The maximum number of tokens allowed for a single (system, user) prompt pair.
TOKEN_BUDGET_PER_QUERY = 4000


def configure_langsmith() -> bool:
    """
    Configure LangSmith tracing using environment variables.
    Returns True if LangSmith is enabled, otherwise False.
    """
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "student-risk-assistant")
    endpoint = os.environ.get("LANGSMITH_ENDPOINT")
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    return True


LANGSMITH_ENABLED = configure_langsmith()


def get_llm_config() -> dict:
    """
    Load LLM provider, API keys, and model settings from environment variables.
    Uses OpenAI as the default provider when no provider is specified.
    """
    return {
        "provider": os.environ.get("LLM_PROVIDER", "openai"),
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "gemini_api_key": os.environ.get("GEMINI_API_KEY"),
        "openai_model": os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        "gemini_model": os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    }


def is_configured(provider: Provider | None = None) -> bool:
    """
    Check whether the selected LLM provider has a valid API key.
    Returns True if the provider is configured, otherwise False.
    """
    cfg = get_llm_config()
    provider = provider or cfg["provider"]
    if provider == "openai":
        return bool(cfg["openai_api_key"])
    if provider == "gemini":
        return bool(cfg["gemini_api_key"])
    return False


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in the given text using tiktoken.
    Uses an approximate character-based count if tokenization fails.
    """
    try:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def check_token_budget(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> dict:
    """
    Calculate the total tokens used by the system and user prompts.
    Check whether the total token count is within the allowed budget.
    """
    total = count_tokens(system_prompt, model) + count_tokens(user_prompt, model)
    return {
        "total_tokens": total,
        "budget": TOKEN_BUDGET_PER_QUERY,
        "within_budget": total <= TOKEN_BUDGET_PER_QUERY,
    }


class LLMError(Exception):
    """
    Custom error for LLM configuration or API failures.
    Allows the application to handle LLM errors gracefully.
    """
    pass


def complete(system_prompt: str, user_prompt: str, provider: Provider | None = None) -> str:
    """ Send a prompt to OpenAI or Gemini and return the response.
    Raises LLMError if the API call fails or configuration is missing. """
    cfg = get_llm_config()
    provider = provider or cfg["provider"]

    if provider == "openai":
        if not cfg["openai_api_key"]:
            raise LLMError("OPENAI_API_KEY is not set in .env")
        try:
            import openai
            client = openai.OpenAI(api_key=cfg["openai_api_key"])
            response = client.chat.completions.create(
                model=cfg["openai_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI call failed: {e}") from e

    if provider == "gemini":
        if not cfg["gemini_api_key"]:
            raise LLMError("GEMINI_API_KEY is not set in .env")
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=cfg["gemini_api_key"])
            response = client.models.generate_content(
                model=cfg["gemini_model"],
                contents=user_prompt,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            )
            return response.text
        except Exception as e:
            raise LLMError(f"Gemini call failed: {e}") from e

    raise LLMError(f"Unknown provider: {provider!r} (expected 'openai' or 'gemini')")


def get_chat_model(provider: Provider | None = None):
    """
    Create a LangChain chat model for the selected LLM provider.
    Returns either a ChatOpenAI or ChatGoogleGenerativeAI model.
    """
    cfg = get_llm_config()
    provider = provider or cfg["provider"]

    if provider == "openai":
        if not cfg["openai_api_key"]:
            raise LLMError("OPENAI_API_KEY is not set in .env")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(openai_api_key=cfg["openai_api_key"], model_name=cfg["openai_model"])

    if provider == "gemini":
        if not cfg["gemini_api_key"]:
            raise LLMError("GEMINI_API_KEY is not set in .env")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(google_api_key=cfg["gemini_api_key"], model=cfg["gemini_model"])

    raise LLMError(f"Unknown provider: {provider!r} (expected 'openai' or 'gemini')")


def stream_complete(system_prompt: str, user_prompt: str, provider: Provider | None = None) -> Iterator[str]:
    """
    Stream the LLM response in small chunks as they are generated.
    Handles streaming errors without losing an already generated response.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    model = get_chat_model(provider)  # raises LLMError here if unconfigured
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    any_yielded = False
    try:
        for chunk in model.stream(messages):
            text = _extract_chunk_text(chunk.content)
            if text:
                any_yielded = True
                yield text
    except Exception as e:
        if not any_yielded:
            raise LLMError(f"Streaming call failed: {e}") from e
        # Real content already reached the caller -- don't discard a
        # successful answer over a post-stream finalization/tracing
        # failure. Stop iterating cleanly instead of raising.
        return


def _extract_chunk_text(content) -> str:
    """
    Extract plain text from different LangChain content formats.
    Combines text blocks into a single string for the chat interface.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    pieces.append(text)
            elif isinstance(item, list):
                pieces.append(_extract_chunk_text(item))
        return "".join(pieces)
    return ""
