import json
import os
from typing import AsyncGenerator, List, Optional

from anthropic import AsyncAnthropic, types

from agents.core.chat_context import ChatMessage
from agents.core.tools import Tool, ToolCall
from llms.anthropic.utils import chat_messages_to_anthropic_system_and_messages, tool_to_anthropic_tool
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
        current_tool_call: Optional[ToolCall] = None
        current_tool_call_args: Optional[str] = None
        async for chunk in stream:
            if isinstance(chunk, types.RawContentBlockStartEvent):
                content_block = chunk.content_block
                if isinstance(content_block, types.ToolUseBlock):
                    if current_tool_call:
                        raise ValueError("Received a ToolUseBlock but current_tool_call already exists.")
                    current_tool_call = ToolCall(id=content_block.id, name=content_block.name)
            elif isinstance(chunk, types.RawContentBlockDeltaEvent):
                if isinstance(chunk.delta, types.TextDelta):
                    yield chunk.delta.text
                elif isinstance(chunk.delta, types.InputJSONDelta):
                    delta = chunk.delta.partial_json
                    if not current_tool_call:
                        raise ValueError("Received tool call delta, but no current_tool_call exists.")
                    current_tool_call_args = f"{current_tool_call_args}{delta}" if current_tool_call_args else delta
            elif isinstance(chunk, types.RawMessageDeltaEvent):
                if chunk.delta.stop_reason == "tool_use":
                    if not current_tool_call:
                        raise ValueError("Received stop tool_use event, but no current_tool_call to yield.")
                    current_tool_call.args = json.loads(current_tool_call_args)
                    yield current_tool_call
                    current_tool_call, current_tool_call_args = None, None
