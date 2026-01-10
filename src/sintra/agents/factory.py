from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from sintra.profiles.models import LLMConfig, LLMProvider, ModelRecipe

# sintra/agents/factory.py


def get_architect_llm(config: LLMConfig):
    """Returns an LLM instance with structured output capabilities."""

    if config.provider == LLMProvider.OPENAI:
        llm = ChatOpenAI(model=config.model_name, temperature=config.temperature)

    elif config.provider == LLMProvider.ANTHROPIC:
        llm = ChatAnthropic(model=config.model_name, temperature=config.temperature)

    elif config.provider == LLMProvider.GOOGLE:
        # Ensure the model name doesn't double-prefix
        model_name = config.model_name
        if not model_name.startswith("models/"):
            # Some LangChain versions prefer the prefix, some don't.
            # 'gemini-1.5-flash-latest' is usually the most stable.
            pass

        llm = ChatGoogleGenerativeAI(model=model_name, temperature=config.temperature)

    elif config.provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=config.model_name, temperature=config.temperature)

    else:
        raise ValueError(f"Provider {config.provider} not supported.")

    # Bind the Pydantic model to ensure the LLM returns a ModelRecipe object
    return llm.with_structured_output(ModelRecipe)
