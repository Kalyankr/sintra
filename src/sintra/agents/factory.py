from typing import Union

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from sintra.profiles.models import LLMConfig, LLMProvider, ModelRecipe

# Type alias for LLM with structured output
StructuredLLM = Union[
    ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, "ChatOllama"
]


def get_architect_llm(config: LLMConfig) -> StructuredLLM:
    """Returns an LLM instance with structured output capabilities.
    
    Args:
        config: LLM configuration with provider, model name, and temperature.
    
    Returns:
        An LLM instance bound to ModelRecipe schema for structured output.
    
    Raises:
        ValueError: If the provider is not supported.
    """
    llm: StructuredLLM

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

    # Bind the Pydantic model to ensure the LLM returns a ModelRecipe object
    return llm.with_structured_output(ModelRecipe)
