# Abstract base class for LLMs
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

from agents.chat_context import ChatMessage
from agents.tools import Tool, ToolCall


class LLM(ABC):
    @abstractmethod
    async def astream(
        self,
        messages: list[ChatMessage],
        tools: List[Tool],
    ) -> AsyncGenerator[str | ToolCall]:
        raise NotImplementedError("Subclasses must implement this method!")
