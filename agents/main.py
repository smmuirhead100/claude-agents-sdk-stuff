import asyncio
from agents.agent_with_tools import AgentWithTools, AgentWithToolsOptions
from agents.chat_context import ChatMessage, ChatRole
from agents.tools import ToolCall
from llms.anthropic.models import AnthropicLLMModel
from llms.anthropic.llm import LLM


async def run():
    llm = LLM(model=AnthropicLLMModel.CLAUDE_4_5_SONNET.value)
    options = AgentWithToolsOptions(llm=llm, tools=[])
    agent = AgentWithTools(options=options)

    # Initialize messages with system message
    messages: list[ChatMessage] = [
        ChatMessage(role=ChatRole.SYSTEM, content="You are a silly agent.")
    ]

    print(
        "Chat with the agent! Type 'exit' or 'quit' to end the conversation.\n"
    )

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to history
        messages.append(ChatMessage(role=ChatRole.USER, content=user_input))

        # Stream agent response
        print("Agent: ", end="", flush=True)
        assistant_content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        async for chunk in agent.astream(messages=messages):
            if isinstance(chunk, ToolCall):
                tool_calls.append(chunk)
                # Tool calls are handled internally, but we might want to
                # display them. For now, we'll just collect them.
            else:
                # It's a string chunk
                print(chunk, end="", flush=True)
                assistant_content_parts.append(chunk)

        print()  # New line after response

        # Add assistant response to history
        assistant_content = "".join(assistant_content_parts)
        if assistant_content:
            messages.append(
                ChatMessage(
                    role=ChatRole.ASSISTANT, content=assistant_content
                )
            )

        # If there were tool calls, we might want to add them to the
        # conversation. This depends on how you want to handle tool calls
        # in the message history. For now, the tool calls are executed and
        # their responses are included in the LLM's context.


if __name__ == "__main__":
    asyncio.run(run())
