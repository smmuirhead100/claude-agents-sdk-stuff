import asyncio
from agents.agent_with_tools import AgentWithTools, AgentWithToolsOptions
from agents.chat_context import ChatMessage, ChatRole
from llms.anthropic.models import AnthropicLLMModel
from llms.anthropic.llm import LLM


async def run():
    llm = LLM(model=AnthropicLLMModel.CLAUDE_4_5_SONNET)
    options = AgentWithToolsOptions(llm=llm, tools=[])
    agent = AgentWithTools(options=options)
    async for chunk in agent.astream(messages=[ChatMessage(role=ChatRole.SYSTEM, content="You are a silly agent.")]):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
