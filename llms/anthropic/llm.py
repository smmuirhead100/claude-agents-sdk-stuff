import os
from typing import AsyncGenerator, List

from anthropic import AsyncAnthropic

from agents.chat_context import ChatMessage
from agents.tools import Tool, ToolCall
from llms.anthropic.utils import anthropic_chunk_to_str_or_tool_call, chat_messages_to_anthropic_system_and_messages, tool_to_anthropic_tool
from .models import AnthropicLLMModel
from llms.llm import LLM as BaseLLM
from dotenv import load_dotenv

load_dotenv()


class LLM(BaseLLM):
    def __init__(self, model: AnthropicLLMModel = AnthropicLLMModel.CLAUDE_4_5_SONNET) -> None:
        self.model = model
        self.client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    async def astream(
        self,
        messages: list[ChatMessage],
        tools: List[Tool],
    ) -> AsyncGenerator[str | ToolCall]:
        system, messages = chat_messages_to_anthropic_system_and_messages(messages)
        stream = await self.client.messages.create(
            max_tokens=1024,
            system=system,
            messages=messages,
            model=self.model,
            stream=True,
            tools=[tool_to_anthropic_tool(t) for t in tools]
        )
        async for chunk in stream:
            converted_chunk = anthropic_chunk_to_str_or_tool_call(chunk)
            if converted_chunk:
                yield converted_chunk
            
