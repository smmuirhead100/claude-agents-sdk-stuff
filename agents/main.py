import asyncio
import os
import shutil
from agents.builtins.agent_with_bash import AgentWithBash
from agents.core.chat_context import ChatMessage, ChatRole
from agents.core.tools import ToolCall
from llms.anthropic.models import AnthropicLLMModel
from llms.anthropic.llm import LLM


async def run():
    llm = LLM(model=AnthropicLLMModel.CLAUDE_4_5_SONNET.value)
    instructions = "You are a testing agent being tested by the developer right now."
    agent = AgentWithBash(llm=llm, instructions=instructions)

    print("Chat with the agent! Type 'exit' or 'quit' to end the conversation.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            message = ChatMessage(role=ChatRole.USER, content=user_input)

            print("Agent: ", end="", flush=True)
            assistant_content_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            async for chunk in agent.astream(chat_message=message):
                if isinstance(chunk, ToolCall):
                    tool_calls.append(chunk)
                else:
                    print(chunk, end="", flush=True)
                    assistant_content_parts.append(chunk)
            print()
    finally:
        # Clean up the workspace
        if hasattr(agent, 'work_dir') and os.path.exists(agent.work_dir):
            shutil.rmtree(agent.work_dir)


if __name__ == "__main__":
    asyncio.run(run())
