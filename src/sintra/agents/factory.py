from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from sintra.profiles.models import LLMConfig, LLMProvider, ModelRecipe


def get_architect_llm(config: LLMConfig):
    """Returns an LLM instance with structured output capabilities."""

    if config.provider == LLMProvider.OPENAI:
        llm = ChatOpenAI(model=config.model_name, temperature=config.temperature)
    elif config.provider == LLMProvider.ANTHROPIC:
        llm = ChatAnthropic(model=config.model_name, temperature=config.temperature)
    elif config.provider == LLMProvider.GOOGLE:
        llm = ChatGoogleGenerativeAI(
            model=config.model_name, temperature=config.temperature
        )
    else:
        raise ValueError(f"Provider {config.provider} not supported.")

    # Bind the Pydantic model to ensure structured output
    return llm.with_structured_output(ModelRecipe)
