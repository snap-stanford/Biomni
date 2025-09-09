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
from datetime import datetime
import pytz
import shutil
import random
import string

# Configuration
LLM_MODEL = "gemini-2.5-pro"
BIOMNI_DATA_PATH = "/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data"
CURRENT_ABS_DIR = "/workdir_efs/jaechang/work2/biomni_hits_test"

# Initialize agent
agent = A1_HITS(
    path=BIOMNI_DATA_PATH,
    llm=LLM_MODEL,
    use_tool_retriever=True,
)


@cl.on_chat_start
async def start_chat():
    """Initialize chat session and set up working directory."""
    os.chdir(CURRENT_ABS_DIR)

    # Create directory with current Korean time
    korea_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(korea_tz)
    dir_name = current_time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"chainlit_logs/{dir_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.chdir(log_dir)
    print("current dir", os.getcwd())
    cl.user_session.set("message_history", [])


@cl.on_message
async def main(user_message: cl.Message):
    """
    Handle user messages and process them through the agent.

    Args:
        user_message: The user's message from Chainlit UI.
    """
    print("current dir:", os.getcwd())
    user_prompt = _process_user_message(user_message)
    message_history = _update_message_history(user_prompt)
    agent_input = _convert_to_agent_format(message_history)

    await _process_agent_response(agent_input, message_history)


def _process_user_message(user_message: cl.Message) -> str:
    """Process user message and handle file uploads."""
    user_prompt = user_message.content

    # Process uploaded files
    for file in user_message.elements:
        os.system(f"cp {file.path} '{file.name}'")
        user_prompt += f"\n - user uploaded file: {file.name}\n"

    return user_prompt


def _update_message_history(user_prompt: str) -> list:
    """Update and return message history."""
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": user_prompt})
    return message_history


def _convert_to_agent_format(message_history: list) -> list:
    """Convert message history to agent input format."""
    agent_input = []
    for message in message_history:
        if message["role"] == "user":
            agent_input.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            agent_input.append(AIMessage(content=message["content"]))
    return agent_input


async def _process_agent_response(agent_input: list, message_history: list):
    """Process agent response and handle streaming."""
    async with cl.Step(name="Plan and execute") as chainlit_step:
        chainlit_step.output = "Initializing..."
        await chainlit_step.update()

        message_stream = agent.go_stream(agent_input)
        full_message, step_message, raw_full_message = await _handle_message_stream(
            message_stream, chainlit_step
        )

    final_message = _extract_final_message(step_message)
    await cl.Message(content=final_message).send()

    print(step_message)
    message_history.append({"role": "assistant", "content": raw_full_message})


async def _handle_message_stream(message_stream, chainlit_step):
    """Handle streaming messages from the agent."""
    full_message = ""
    step_message = ""
    raw_full_message = ""
    current_step = 1

    for chunk in message_stream:
        this_step = chunk[1][1]["langgraph_step"]

        if this_step != current_step:
            step_message = ""
            current_step = this_step
            if full_message.count("```") % 2 == 1:
                full_message += "```\n"
                raw_full_message += "```\n"

        chunk_content = _extract_chunk_content(chunk)
        if chunk_content is None:
            continue

        if isinstance(chunk_content, str):
            raw_full_message += chunk_content
            full_message += chunk_content
            step_message += chunk_content

        full_message = _modify_chunk(full_message)
        full_message = _detect_image_name_and_move_to_public(full_message)
        chainlit_step.output = full_message
        await chainlit_step.update()

    step_message = _detect_image_name_and_move_to_public(step_message)
    return full_message, step_message, raw_full_message


def _extract_chunk_content(chunk):
    """Extract content from chunk based on node type."""
    node = chunk[1][1]["langgraph_node"]
    chunk_data = chunk[1][0]

    if node == "generate" and isinstance(chunk_data, AIMessageChunk):
        return chunk_data.content
    elif node == "execute":
        return chunk_data.content
    else:
        return None


def _extract_final_message(step_message: str) -> str:
    """Extract final message from step message."""
    if "<solution>" in step_message and "</solution>" not in step_message:
        step_message += "</solution>"

    solution_match = re.search(r"<solution>(.*?)</solution>", step_message, re.DOTALL)
    return solution_match.group(1) if solution_match else step_message


def _modify_chunk(chunk: str) -> str:
    """Modify chunk content by replacing tags."""
    retval = chunk
    tag_replacements = [
        ("<execute>", "\n```python\n"),
        ("</execute>", "```\n"),
        ("<solution>", ""),
        ("</solution>", ""),
        ("<observation>", "```\n#Execute result\n"),
        ("</observation>", "```\n"),
    ]

    for tag1, tag2 in tag_replacements:
        if tag1 in retval:
            retval = retval.replace(tag1, tag2)
    return retval


def _detect_image_name_and_move_to_public(content: str) -> str:
    """
    Detect images in markdown text, move them to public folder with random prefix.

    Args:
        content: Markdown text content

    Returns:
        Modified markdown text with updated image paths
    """
    public_dir = f"{CURRENT_ABS_DIR}/public"
    os.makedirs(public_dir, exist_ok=True)

    # Pattern to find markdown images, excluding those already with download functionality
    image_pattern = r'(?<!\[)!\[([^\]]*)\]\(([^)]+?)(?:\s+"[^"]*")?\)(?!\[Download\])'

    def replace_image(match):
        alt_text = match.group(1)
        image_path = match.group(2).strip()

        # Skip URLs
        if image_path.startswith(("http://", "https://")):
            return match.group(0)

        # Add download functionality if already in public folder
        if image_path.startswith(("./public/", "public/")):
            return (
                f"[![{alt_text}]({image_path})]({image_path})[Download]({image_path})"
            )

        # Check if file exists
        if not os.path.exists(image_path):
            return match.group(0)

        # Generate random prefix and new filename
        random_prefix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )
        file_name = os.path.basename(image_path)
        new_file_name = f"{random_prefix}_{file_name}"
        new_file_path = os.path.join(public_dir, new_file_name)

        try:
            shutil.copy2(image_path, new_file_path)
            return f"[![{alt_text}](./public/{new_file_name})](./public/{new_file_name})[Download](./public/{new_file_name})"
        except Exception as e:
            print(f"Error moving image {image_path}: {e}")
            return match.group(0)

    return re.sub(image_pattern, replace_image, content)
