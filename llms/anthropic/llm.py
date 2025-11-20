from typing import AsyncGenerator
from .models import ClaudeAgentOptions, AgentDefinition
from llms.llm import LLM as BaseLLM


class LLM(BaseLLM):
    def __init__(self, model: AnthropicLLMModel) -> None:
        self.model = model
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def astream(self, prompt: str) -> AsyncGenerator[str]:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                agents={
                    "agent": AgentDefinition(tools=self.tools),
                },
            ),
        ):
            yield message.content