import chainlit as cl
from biomni.agent import A1_HITS
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
import os
import re
llm = "gemini-2.5-pro"
agent = A1_HITS(
    path="/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data",
    llm=llm,
    use_tool_retriever=True,
)
current_abs_dir = "/workdir_efs/jaechang/work2/biomni_hits_test"

@cl.on_chat_start
async def start_chat():
    import os
    from datetime import datetime
    import pytz

    os.chdir(current_abs_dir)

    # Create directory with current Korean time
    korea_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(korea_tz)
    dir_name = current_time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"chainlit_logs/{dir_name}", exist_ok=True)
    os.chdir(f"chainlit_logs/{dir_name}")
    print ("current dir", os.getcwd())
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
    
    async with cl.Step(name="Plan and execute") as chainlit_step:
        message_stream = agent.go_stream(agent_input)
        # msg = cl.Message(content="")
        # await msg.send()
        full_message = ""
        step_message = ""
        step = 1
        for chunk in message_stream:
            this_step = chunk[1][1]["langgraph_step"]
            if this_step != step:
                step_message = ""
                step = this_step
                if full_message.count("```") % 2 == 1:
                    full_message += "```\n"

            if chunk[1][1]["langgraph_node"] == "generate" and type(chunk[1][0]) == AIMessageChunk:
                chunk = chunk[1][0].content
            elif chunk[1][1]["langgraph_node"] == "execute":
                chunk = chunk[1][0].content
            else:
                continue

            if type(chunk) == str:
                full_message += chunk
                step_message += chunk
            full_message = modify_chunk(full_message)
            chainlit_step.output = full_message
            await chainlit_step.update()

    if "<solution>" in step_message and "</solution>" not in step_message:
        step_message += "</solution>"

    solution_match = re.search(
        r"<solution>(.*?)</solution>", step_message, re.DOTALL
    )
    if solution_match:
        final_message = solution_match.group(1)
    else:
        final_message = step_message
    msg = cl.Message(content=final_message)
    await msg.send()
    print (step_message)
    message_history.append({"role": "assistant", "content": full_message})


def modify_chunk(chunk):
    retval = chunk
    for tag1, tag2 in [
        ("<execute>", "\n```python\n"),
        ("</execute>", "```\n"),
        ("<solution>", ""),
        ("</solution>", ""),
        ("<observation>", "```\n#Execute result\n"),
        ("</observation>", "```\n"),
    ]:
        if tag1 in retval:
            retval = retval.replace(tag1, tag2)
    return retval