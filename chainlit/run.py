import chainlit as cl
from biomni.agent import A1_HITS
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import os

llm = "gemini-2.5-pro"
agent = A1_HITS(
    path="/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data",
    llm=llm,
    use_tool_retriever=True,
)


@cl.on_chat_start
async def start_chat():
    import os
    from datetime import datetime
    import pytz

    current_abs_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_abs_dir)

    # Create directory with current Korean time
    korea_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(korea_tz)
    dir_name = current_time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"logs/{dir_name}", exist_ok=True)
    os.chdir(f"logs/{dir_name}")

    cl.user_session.set("message_history", [])


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(user_message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """
    user_prompt = user_message.content
    # Processing images exclusively
    for file in user_message.elements:
        print(file.path, file.name)
        os.system(f"cp {file.path} {file.name}")
        user_prompt += f"\n - user uploaded file: {file.name}\n"

    # await cl.Message(content="jaechang3").send()
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": user_prompt})

    agent_input = []
    for message in message_history:
        if message["role"] == "user":
            agent_input.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            agent_input.append(AIMessage(content=message["content"]))
    full_message = ""
    msg = cl.Message(content="")
    await msg.send()

    full_message = ""
    for chunk in agent.go_stream(agent_input):
        print(chunk, end="")
        full_message += chunk
        await msg.stream_token(chunk)
        # You can update the message content during streaming

        for tag1, tag2 in [
            ("<execute>", "```"),
            ("</execute>", "```"),
            ("<solution>", ""),
            ("</solution>", ""),
        ]:
            if tag1 in full_message:
                full_message = full_message.replace(tag1, tag2)
                msg.content = full_message
                await msg.update()

    # Final update after streaming is complete
    await msg.update()
    message_history.append({"role": "assistant", "content": full_message})


a = 1
