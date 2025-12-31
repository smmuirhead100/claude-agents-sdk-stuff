import json
import os
from typing import AsyncGenerator, List

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

        # Track multiple tool calls in parallel by their ID
        tool_calls: dict[str, ToolCall] = {}
        tool_call_args: dict[str, str] = {}
        index_to_tool_id: dict[int, str] = {}
        tool_call_order: list[str] = []
        current_tool_id: str | None = None

        async for chunk in stream:
            if isinstance(chunk, types.RawContentBlockStartEvent):
                content_block = chunk.content_block
                if isinstance(content_block, types.ToolUseBlock):
                    current_tool_id = content_block.id
                    tool_calls[current_tool_id] = ToolCall(id=content_block.id, name=content_block.name)
                    tool_call_args[current_tool_id] = ""
                    index = getattr(chunk, 'index', None)
                    if index is not None:
                        index_to_tool_id[index] = content_block.id
                    tool_call_order.append(content_block.id)
            elif isinstance(chunk, types.RawContentBlockDeltaEvent):
                if isinstance(chunk.delta, types.TextDelta):
                    yield chunk.delta.text
                elif isinstance(chunk.delta, types.InputJSONDelta):
                    delta = chunk.delta.partial_json
                    # Try to get the index from the delta event first
                    index = getattr(chunk, 'index', None)
                    tool_id = None
                    if index is not None and index in index_to_tool_id:
                        tool_id = index_to_tool_id[index]
                    elif current_tool_id:
                        # Fallback: use the current tool ID we're processing
                        tool_id = current_tool_id
                    elif tool_call_order:
                        # Last resort: use the most recently added tool call
                        tool_id = tool_call_order[-1]
                    
                    if tool_id and tool_id in tool_call_args:
                        tool_call_args[tool_id] = (
                            tool_call_args.get(tool_id, "") + delta
                        )
            elif isinstance(chunk, types.RawContentBlockStopEvent):
                # Block is done, clear current tool ID
                current_tool_id = None
            elif isinstance(chunk, types.RawMessageDeltaEvent):
                if chunk.delta.stop_reason == "tool_use":
                    # Yield all completed tool calls
                    for tool_id in tool_call_order:
                        if tool_id in tool_calls:
                            tool_call = tool_calls[tool_id]
                            # Handle case where tool call has no arguments or empty string
                            args_str = tool_call_args.get(tool_id, "")
                            if args_str and args_str.strip():
                                try:
                                    tool_call.args = json.loads(args_str)
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, use empty dict
                                    tool_call.args = {}
                            else:
                                tool_call.args = {}
                            yield tool_call
                    tool_calls.clear()
                    tool_call_args.clear()
                    tool_call_order.clear()
