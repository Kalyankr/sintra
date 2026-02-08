import os
from functools import lru_cache
from typing import TYPE_CHECKING, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from sintra.profiles.models import LLMConfig, LLMProvider, ModelRecipe

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

# Type alias for LLM with structured output
StructuredLLM = Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, "ChatOllama"]

# Map providers to their required environment variables
PROVIDER_API_KEYS = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.GOOGLE: "GOOGLE_API_KEY",
}

# Cache for LLM instances — avoids re-creating expensive clients per call
_llm_cache: dict[tuple[str, str, float], BaseChatModel] = {}


class MissingAPIKeyError(Exception):
    """Raised when a required API key is not set."""

    pass


def _get_base_llm(config: LLMConfig) -> BaseChatModel:
    """Returns a base LLM instance without structured output binding.

    Caches instances by (provider, model_name, temperature) to avoid
    re-creating expensive API clients on every node invocation.

    Args:
        config: LLM configuration with provider, model name, and temperature.

    Returns:
        A base LLM instance.

    Raises:
        MissingAPIKeyError: If the required API key is not set.
        ValueError: If the provider is not supported.
    """
    cache_key = (config.provider.value, config.model_name, config.temperature)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    # Check for required API key (Ollama doesn't need one)
    if config.provider in PROVIDER_API_KEYS:
        env_var = PROVIDER_API_KEYS[config.provider]
        if not os.environ.get(env_var):
            raise MissingAPIKeyError(
                f"{env_var} is not set. "
                f"Add it to your .env file or export it in your shell."
            )

    if config.provider == LLMProvider.OPENAI:
        llm = ChatOpenAI(model=config.model_name, temperature=config.temperature)

    elif config.provider == LLMProvider.ANTHROPIC:
        llm = ChatAnthropic(model=config.model_name, temperature=config.temperature)

    elif config.provider == LLMProvider.GOOGLE:
        # Normalize model name - some LangChain versions require 'models/' prefix
        model_name = config.model_name
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=config.temperature)

    elif config.provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            num_ctx=4096,
            format="json",
        )

    else:
        raise ValueError(f"Provider {config.provider} not supported.")

    _llm_cache[cache_key] = llm
    return llm


def get_architect_llm(config: LLMConfig) -> StructuredLLM:
    """Returns an LLM instance with structured output capabilities.

    Args:
        config: LLM configuration with provider, model name, and temperature.

    Returns:
        An LLM instance bound to ModelRecipe schema for structured output.

    Raises:
        MissingAPIKeyError: If the required API key is not set.
        ValueError: If the provider is not supported.
    """
    llm = _get_base_llm(config)
    # Bind the Pydantic model to ensure the LLM returns a ModelRecipe object
    return llm.with_structured_output(ModelRecipe)


def get_tool_enabled_llm(config: LLMConfig, tools: list) -> BaseChatModel:
    """Returns an LLM instance with tool-calling capabilities.

    Args:
        config: LLM configuration with provider, model name, and temperature.
        tools: List of tools to bind to the LLM.

    Returns:
        An LLM instance with tools bound.

    Raises:
        MissingAPIKeyError: If the required API key is not set.
        ValueError: If the provider is not supported.
    """
    llm = _get_base_llm(config)
    return llm.bind_tools(tools)


def get_critic_llm(config: LLMConfig) -> BaseChatModel:
    """Returns an LLM instance for the critic agent.

    The critic uses a simpler interface without structured output
    to provide free-form feedback and routing decisions.

    Args:
        config: LLM configuration with provider, model name, and temperature.

    Returns:
        A base LLM instance for critic reasoning.
    """
    return _get_base_llm(config)
