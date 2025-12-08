from typing import AsyncGenerator, List
import inspect

from pydantic import create_model
from agents.core.chat_context import ChatMessage
from agents.core.tools import Tool, ToolCall
from llms.llm import LLM

_IS_TOOL = "is_tool"


class AgentWithTools:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self._tools = self._get_tools_from_decorated_methods()
        print([t.model_dump() for t in self._tools])

    async def astream(self, messages: List[ChatMessage]) -> AsyncGenerator[str | ToolCall]:
        stream = self._llm.astream(
            messages=messages,
            tools=self._tools,
        )
        async for chunk in stream:
            if isinstance(chunk, ToolCall):
                tool_call_response = await self._execute_tool_call(tool_call=chunk)
                chunk.response = tool_call_response
            yield chunk

    async def _execute_tool_call(self, tool_call: ToolCall) -> str:
        method_name = tool_call.name
        method = getattr(self, method_name)
        if not method:
            raise ValueError(f"Method '{method_name}' not found on {self.__class__.__name__}")
        result = await method(**tool_call.args)
        return str(result)

    def _get_tools_from_decorated_methods(self) -> List[Tool]:
        tools = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, _IS_TOOL):
                sig = inspect.signature(attr)
                fields = {name: (param.annotation, ...) for name, param in sig.parameters.items() if name != "self"}
                input_schema = create_model(f"{attr.__name__}Input", **fields) if fields else create_model(f"{attr.__name__}Input")
                tools.append(Tool(
                    name=attr.__name__,
                    description=attr.__doc__ or "",
                    input_schema=input_schema,
                ))
        return tools
