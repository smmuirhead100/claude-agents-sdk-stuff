from typing import AsyncGenerator

from pydantic import BaseModel, ConfigDict
from agents.chat_context import ChatMessage
from agents.tools import Tool, ToolCall
from llms.llm import LLM


class AgentWithToolsOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    llm: LLM
    tools: list[Tool]


class AgentWithTools:
    def __init__(self, options: AgentWithToolsOptions) -> None:
        self.options = options

    async def astream(self, messages: list[ChatMessage]) -> AsyncGenerator[str]:
        stream = self.options.llm.astream(
            messages=messages,
            tools=self.options.tools,
        )
        async for chunk in stream:
            if isinstance(chunk, ToolCall):
                raise NotImplementedError("Tools not handled.")
            yield chunk
