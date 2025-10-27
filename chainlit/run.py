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
from biomni.config import default_config
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import time
from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
import asyncio
import logging

# from chainlit.data.base import BaseStorageClient

# Configuration
LLM_MODEL = "gemini-2.5-pro"
# LLM_MODEL = "grok-4-fast"
BIOMNI_DATA_PATH = "/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data"
CURRENT_ABS_DIR = "/workdir_efs/jaechang/work2/biomni_hits_test/Biomni_HITS/chainlit"
PUBLIC_DIR = f"{CURRENT_ABS_DIR}/public"
CHAINLIT_DB_PATH = "chainlit.db"

default_config.llm = LLM_MODEL
default_config.commercial_mode = True
# Initialize agent
agent = A1_HITS(
    path=BIOMNI_DATA_PATH,
    llm=LLM_MODEL,
    use_tool_retriever=True,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chainlit_db.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class CustomSQLAlchemyDataLayer(SQLAlchemyDataLayer):
    def __init__(self, conninfo: str, **kwargs):
        super().__init__(conninfo, **kwargs)

        self.engine: AsyncEngine = create_async_engine(
            conninfo,
            pool_size=100,  # SQLite는 단일 연결이 효율적
            max_overflow=200,  # 오버플로우 방지
            pool_timeout=60,  # 60초 대기
            pool_recycle=3600,  # 1시간마다 연결 재생성
            pool_pre_ping=True,  # 연결 상태 확인
            echo=False,  # SQL 로깅 비활성화
            connect_args={
                "timeout": 30,  # 30초 타임아웃
                "check_same_thread": False,  # 멀티스레드 허용
            },
        )

        self.async_session = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,  # 성능 향상 및 lock 시간 단축
            class_=AsyncSession,
            autoflush=False,  # 자동 플러시 비활성화로 성능 향상
        )

        # 재시도 설정
        self.max_retries = 5
        self.retry_delay = 0.1  # 100ms

    async def __aenter__(self):
        # SQLite 최적화 설정
        @event.listens_for(self.engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            # WAL 모드 활성화 (동시성 개선)
            cursor.execute("PRAGMA journal_mode=WAL")
            # 바쁜 타임아웃 설정 (60초로 증가)
            cursor.execute("PRAGMA busy_timeout=60000")
            # 동기화 모드 최적화 (NORMAL보다 안전한 FULL 사용)
            cursor.execute("PRAGMA synchronous=FULL")
            # 캐시 크기 증가
            cursor.execute("PRAGMA cache_size=20000")
            # WAL 자동 체크포인트 설정 (더 자주 체크포인트)
            cursor.execute("PRAGMA wal_autocheckpoint=500")
            # 임시 데이터를 메모리에 저장
            cursor.execute("PRAGMA temp_store=MEMORY")
            # 메모리 맵핑 크기 설정 (256MB)
            cursor.execute("PRAGMA mmap_size=268435456")
            # 외래키 제약 조건 비활성화 (성능 향상)
            cursor.execute("PRAGMA foreign_keys=OFF")
            # 락 타임아웃 추가 설정
            cursor.execute("PRAGMA lock_timeout=60000")
            # WAL 모드에서 읽기 성능 향상
            cursor.execute("PRAGMA read_uncommitted=1")
            cursor.close()

        return await super().__aenter__()


@cl.data_layer
def get_data_layer():
    # 절대 경로로 변환하여 데이터베이스 경로 설정
    db_path = os.path.abspath(CHAINLIT_DB_PATH)
    conninfo = f"sqlite+aiosqlite:///{db_path}"
    print(f"Chainlit database path: {db_path}")

    return CustomSQLAlchemyDataLayer(conninfo=conninfo, show_logger=False)


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def start_chat():
    """Initialize chat session and set up working directory."""
    os.chdir(CURRENT_ABS_DIR)
    dir_name = cl.context.session.thread_id
    log_dir = f"chainlit_logs/{dir_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.chdir(log_dir)
    print("current dir", os.getcwd())
    cl.user_session.set("message_history", [])


@cl.on_chat_resume
async def resume_chat():
    os.chdir(CURRENT_ABS_DIR)
    thread_id = cl.context.session.thread_id
    dir_name = thread_id
    log_dir = f"chainlit_logs/{dir_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.chdir(log_dir)
    print("current dir", os.getcwd())


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
    user_prompt = user_message.content.strip()

    # Process uploaded files
    for file in user_message.elements:
        os.system(f"cp {file.path} '{file.name}'")
        user_prompt += f"\n - user uploaded file: {file.name}\n"

    return user_prompt


def _update_message_history(user_prompt: str) -> list:
    """Update and return message history."""
    message_history = cl.user_session.get("message_history", [])
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

    with open(f"conversion_history.txt", "a") as f:
        f.write(agent_input[-1].content + "\n")

    async with cl.Step(name="Plan and execute") as chainlit_step:
        await chainlit_step.update()
        message_stream = agent.go_stream(agent_input)
        full_message, step_message, raw_full_message = await _handle_message_stream(
            message_stream, chainlit_step
        )

    final_message = _extract_final_message(raw_full_message)
    final_message = _detect_image_name_and_move_to_public(final_message)

    await cl.Message(content=final_message).send()

    print(os.getcwd())
    print(final_message)

    with open(f"conversion_history.txt", "a") as f:
        f.write(raw_full_message + "\n")
    message_history.append({"role": "assistant", "content": raw_full_message})


async def _handle_message_stream(message_stream, chainlit_step):
    """Handle streaming messages from the agent."""
    full_message = ""
    step_message = ""
    raw_full_message = ""
    current_step = 1
    update_counter = 0
    last_update = time.time()
    update_pending = False
    stream_index = 0
    for chunk in message_stream:
        this_step = chunk[1][1]["langgraph_step"]
        if this_step != current_step:
            step_message = ""
            current_step = this_step
            if full_message.count("```") % 2 == 1:
                full_message += "```\n"
                raw_full_message += "```\n"
            await chainlit_step.stream_token(full_message[stream_index:])
            stream_index = max(len(full_message) - 1, 0)
        chunk_content = _extract_chunk_content(chunk)
        if chunk_content is None:
            continue

        if isinstance(chunk_content, str):
            raw_full_message += chunk_content
            full_message += chunk_content
            step_message += chunk_content

        tmp_full_message = full_message
        a = len(tmp_full_message)
        full_message = _modify_chunk(full_message)
        b = len(full_message)
        full_message = _detect_image_name_and_move_to_public(full_message)
        c = len(full_message)

        print(f"{a}\t{b}\t{c}\t{stream_index}")
        if tmp_full_message[:stream_index] != full_message[:stream_index]:
            print(tmp_full_message[:stream_index])
            print("--------------------------------")
            print(full_message[:stream_index])
            print("--------------------------------")

        chainlit_step.output = full_message
        stream_chunk = full_message[stream_index:]
        if len(stream_chunk) > 100:
            await chainlit_step.stream_token(stream_chunk)
            stream_index = len(full_message) - 1

    await chainlit_step.stream_token(full_message[stream_index:])

    step_message = _detect_image_name_and_move_to_public(step_message)
    step_message = _modify_chunk(step_message)

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


def _detect_code_type(code: str) -> str:
    """
    Detect code type based on markers and content.

    Args:
        code: Code content to analyze

    Returns:
        Code type string for markdown code block (python, r, bash)
    """
    code_stripped = code.strip()

    # Check for explicit markers
    if (
        code_stripped.startswith("#!R")
        or code_stripped.startswith("# R code")
        or code_stripped.startswith("# R script")
    ):
        return "r"
    elif (
        code_stripped.startswith("#!BASH")
        or code_stripped.startswith("# Bash script")
        or code_stripped.startswith("#!CLI")
    ):
        return "bash"

    # Heuristic detection based on common patterns
    # R patterns
    r_patterns = [
        r"\blibrary\(",
        r"\brequire\(",
        r"<-",
        r"\$",
        r"\.R\b",
        r"\bdata\.frame\(",
        r"\bggplot\(",
        r"\bc\(",
    ]

    # Bash patterns
    bash_patterns = [
        r"^#!",
        r"\becho\b",
        r"\bls\b",
        r"\bcd\b",
        r"\bmkdir\b",
        r"\bcp\b",
        r"\bmv\b",
        r"\brm\b",
        r"\bgrep\b",
        r"\bawk\b",
        r"\bsed\b",
        r"\|\s*\w+",  # pipe commands
    ]

    # Count pattern matches
    r_score = sum(1 for pattern in r_patterns if re.search(pattern, code_stripped))
    bash_score = sum(
        1 for pattern in bash_patterns if re.search(pattern, code_stripped)
    )

    # Determine code type based on scores
    if r_score > 0 and r_score >= bash_score:
        return "r"
    elif bash_score > 0:
        return "bash"
    else:
        return "python"  # Default to python


def _modify_chunk(chunk: str) -> str:
    """Modify chunk content by replacing tags."""
    retval = chunk
    tag_replacements = [
        ("<execute>", "\n```CODE_TYPE\n"),
        ("</execute>", "```\n"),
        ("<solution>", ""),
        ("</solution>", ""),
        ("<observation>", "```\n#Execute result\n"),
        ("</observation>", "\n```\n"),
    ]

    for tag1, tag2 in tag_replacements:
        if tag1 in retval:
            retval = retval.replace(tag1, tag2)

    # # Handle existing CODE_TYPE placeholders in already generated code blocks
    retval = _replace_code_type_placeholders(retval)

    # # Replace biomni imports with hits imports in code blocks
    retval = _replace_biomni_imports(retval)

    return retval


def _replace_code_type_placeholders(content: str) -> str:
    """Replace CODE_TYPE placeholders with detected code types."""
    # Pattern to find code blocks with CODE_TYPE placeholder
    code_type_pattern = r"```CODE_TYPE\n(.*?)```"

    def replace_code_type(match):
        code_content = match.group(1)
        code_type = _detect_code_type(code_content)
        return f"```{code_type}\n{code_content}```"

    # Handle closed code blocks with CODE_TYPE
    content = re.sub(code_type_pattern, replace_code_type, content, flags=re.DOTALL)

    # Handle open code blocks that start with ```CODE_TYPE
    open_code_pattern = r"```CODE_TYPE\n(.*?)(?=```|\Z)"

    def replace_open_code_type(match):
        code_content = match.group(1)
        code_type = _detect_code_type(code_content)
        return f"```{code_type}\n{code_content}"

    content = re.sub(
        open_code_pattern, replace_open_code_type, content, flags=re.DOTALL
    )

    return content


def _replace_biomni_imports(content: str) -> str:
    """Replace 'from biomni.' with 'from hits.' in code blocks."""
    # Pattern to find code blocks (both with specific language and generic)
    code_block_pattern = r"```(\w+)?\n(.*?)```"

    def replace_imports_in_code(match):
        language = match.group(1) if match.group(1) else ""
        code_content = match.group(2)

        # Replace biomni imports with hits imports
        modified_code = code_content.replace("from biomni.", "from hits.")

        if language:
            return f"```{language}\n{modified_code}```"
        else:
            return f"```\n{modified_code}```"

    # Apply replacement to all code blocks
    content = re.sub(
        code_block_pattern, replace_imports_in_code, content, flags=re.DOTALL
    )

    return content


def _detect_image_name_and_move_to_public(content: str) -> str:
    """
    Detect images in markdown text, move them to public folder with random prefix.

    Args:
        content: Markdown text content

    Returns:
        Modified markdown text with updated image paths
    """
    public_dir = PUBLIC_DIR
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
            print("copied image to", new_file_path)
            return f"[![{alt_text}](../../public/{new_file_name})](../../public/{new_file_name})[Download](../../public/{new_file_name})"
        except Exception as e:
            print(f"Error moving image {image_path}: {e}")
            return match.group(0)

    return re.sub(image_pattern, replace_image, content)
