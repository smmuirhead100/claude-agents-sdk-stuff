# Abstract base class for LLMs
from abc import ABC, abstractmethod
from typing import AsyncGenerator

from agents.chat_context import ChatMessage
from agents.tools import ToolCall


class LLM(ABC):
    @abstractmethod
    async def astream(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
    ) -> AsyncGenerator[str | ToolCall]:
        raise NotImplementedError("Subclasses must implement this method!")
