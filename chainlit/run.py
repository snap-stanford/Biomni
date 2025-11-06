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
from chainlit.data.storage_clients.base import BaseStorageClient
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
BIOMNI_DATA_PATH = "/workdir_efs/jhjeon/Biomni/biomni_data"
CURRENT_ABS_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = f"{CURRENT_ABS_DIR}/public"
CHAINLIT_DB_PATH = "chainlit.db"
# 스트리밍 타임아웃 설정 (초)
STREAMING_HEARTBEAT_INTERVAL = 30  # 하트비트 간격 (초)
STREAMING_MAX_TIMEOUT = 600  # 최대 대기 시간 (초, 기본 10분)

default_config.llm = LLM_MODEL
default_config.commercial_mode = True
# Initialize agent
agent = A1_HITS(
    path=BIOMNI_DATA_PATH,
    llm=LLM_MODEL,
    use_tool_retriever=True,
    resource_filter_config_path=f"{CURRENT_ABS_DIR}/resource.yaml",
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


class LocalStorageClient(BaseStorageClient):
    """로컬 파일 시스템에 파일을 저장하는 스토리지 클라이언트"""

    def __init__(self, storage_dir: str = None):
        """
        Args:
            storage_dir: 파일을 저장할 디렉토리 경로 (기본값: PUBLIC_DIR)
        """
        if storage_dir is None:
            storage_dir = PUBLIC_DIR
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> dict:
        """파일을 로컬 파일 시스템에 업로드"""
        file_path = os.path.join(self.storage_dir, object_key)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if isinstance(data, str):
            data = data.encode("utf-8")

        with open(file_path, "wb") as f:
            f.write(data)

        return {
            "object_key": object_key,
            "path": file_path,
            "url": f"/public/{object_key}",
        }

    async def delete_file(self, object_key: str) -> bool:
        """파일 삭제"""
        file_path = os.path.join(self.storage_dir, object_key)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {object_key}: {e}")
            return False

    async def get_read_url(self, object_key: str) -> str:
        """파일 읽기 URL 반환"""
        return f"/public/{object_key}"

    async def close(self) -> None:
        """리소스 정리 (로컬 파일 시스템에서는 필요 없음)"""
        pass


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

    # 스토리지 클라이언트 초기화 (파일 저장용)
    storage_provider = LocalStorageClient()

    return CustomSQLAlchemyDataLayer(
        conninfo=conninfo, storage_provider=storage_provider, show_logger=False
    )


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    # Support both username and email format for login
    valid_logins = [
        ("admin", "admin"),
        ("admin@example.com", "admin"),  # Email format
        ("admin@biomni.com", "admin"),  # Another email format
    ]

    if (username, password) in valid_logins:
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

    files = None

    # # Wait for the user to upload a file
    # while files == None:
    #     files = await cl.AskFileMessage(
    #         content="Please upload your omics data to analyze!",
    #         accept=["*/*"],
    #         max_size_mb=800,
    #         max_files=10,
    #     ).send()

    # # Show loading indicator while copying file
    # user_uploaded_system_file = ""
    # uploading_status_message = ""
    # async with cl.Step(name=f"Processing files...") as step:
    #     for file in files:
    #         step.output = (
    #             uploading_status_message
    #             + f"Uploading file: {file.name} ({file.size} bytes)..."
    #         )
    #         os.system(f"cp {file.path} '{file.name}'")
    #         step.output = (
    #             uploading_status_message + f"✅ {file.name} uploaded successfully!\n"
    #         )
    #         uploading_status_message = step.output
    #         user_uploaded_system_file += f" - user uploaded data file: {file.name}\n"

    # cl.user_session.set("user_uploaded_system_file", user_uploaded_system_file)

    # # Let the user know that the system is ready
    # await cl.Message(
    #     content=f"{len(files)} files uploaded, total size {sum(file.size for file in files) / (1024 * 1024):.2f} MB!. It is ready to analyze the files."
    # ).send()


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
    user_prompt = await _process_user_message(user_message)
    message_history = _update_message_history(user_prompt)
    agent_input = _convert_to_agent_format(message_history)

    await _process_agent_response(agent_input, message_history)


async def _process_user_message(user_message: cl.Message) -> str:
    """Process user message and handle file uploads."""
    user_prompt = user_message.content.strip()

    # Process uploaded files
    step_message = ""
    for file in user_message.elements:
        os.system(f"cp {file.path} '{file.name}'")
        user_prompt += f"\n - user uploaded data file: {file.name}\n"

    user_prompt += "\n" + cl.user_session.get("user_uploaded_system_file", "")

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

    try:
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

    except asyncio.CancelledError:
        # Handle stop button click
        logger.info("User requested to stop the execution")
        # await cl.Message(
        #     content="⏹️ **실행 중지됨**: 사용자가 실행을 중지했습니다."
        # ).send()
        message_history.append(
            {"role": "assistant", "content": "실행이 사용자에 의해 중지되었습니다."}
        )
        raise  # Re-raise to properly propagate cancellation
    except TimeoutError as e:
        error_message = str(e)
        logger.error(f"Streaming timeout: {error_message}")
        await cl.Message(
            content=f"⚠️ **타임아웃 발생**: {error_message}\n\n"
            f"최대 대기 시간({STREAMING_MAX_TIMEOUT}초)이 초과되었습니다. "
            "더 작은 작업으로 나누어 다시 시도해주세요."
        ).send()
        message_history.append(
            {"role": "assistant", "content": f"타임아웃 발생: {error_message}"}
        )
    except Exception as e:
        error_message = f"스트리밍 처리 중 오류 발생: {str(e)}"
        logger.error(error_message, exc_info=True)
        await cl.Message(content=f"❌ **오류**: {error_message}").send()
        message_history.append({"role": "assistant", "content": error_message})


async def _handle_message_stream(message_stream, chainlit_step):
    """Handle streaming messages from the agent.

    Similar to streamlit's approach: accumulate raw content, format when displaying,
    and only stream the delta to avoid duplicates.

    Includes heartbeat mechanism to keep connection alive during long waits.
    """
    raw_full_message = ""  # Accumulate raw content without modifications
    step_message = ""  # Step-specific message
    current_step = 1
    last_formatted_text = ""  # Track last formatted text that was displayed
    last_chunk_time = time.time()  # Track last chunk received time
    heartbeat_task = None
    stream_completed = False

    async def heartbeat_loop():
        """Send heartbeat tokens to keep connection alive."""
        nonlocal last_chunk_time, stream_completed
        while not stream_completed:
            try:
                await asyncio.sleep(STREAMING_HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                break

            if stream_completed:
                break

            # Check for cancellation
            current_task = asyncio.current_task()
            if current_task and current_task.cancelled():
                break

            # Check if we've received a chunk recently
            time_since_last_chunk = time.time() - last_chunk_time
            if time_since_last_chunk >= STREAMING_HEARTBEAT_INTERVAL:
                # Send a zero-width space to keep connection alive without visible effect
                try:
                    await chainlit_step.stream_token("\u200b")  # Zero-width space
                    logger.debug(
                        f"Heartbeat sent (last chunk: {time_since_last_chunk:.1f}s ago)"
                    )
                except (asyncio.CancelledError, Exception) as e:
                    logger.warning(f"Failed to send heartbeat: {e}")
                    break

    # Start heartbeat task
    heartbeat_task = asyncio.create_task(heartbeat_loop())

    try:
        # Wrap the streaming loop with timeout
        start_time = time.time()

        for chunk in message_stream:
            # Check for cancellation (stop button)
            current_task = asyncio.current_task()
            if current_task and current_task.cancelled():
                logger.info("Task cancelled, stopping stream")
                stream_completed = True
                raise asyncio.CancelledError("Execution stopped by user")

            # Allow cancellation to be checked periodically
            await asyncio.sleep(0)  # Yield control to allow cancellation

            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > STREAMING_MAX_TIMEOUT:
                logger.warning(f"Streaming timeout after {elapsed_time:.1f}s")
                raise TimeoutError(
                    f"스트리밍이 최대 대기 시간({STREAMING_MAX_TIMEOUT}초)을 초과했습니다. "
                    "연결이 끊어졌을 수 있습니다."
                )

            last_chunk_time = time.time()  # Update last chunk time

            this_step = chunk[1][1]["langgraph_step"]
            if this_step != current_step:
                # Step changed - reset step message
                step_message = ""
                current_step = this_step

            chunk_content = _extract_chunk_content(chunk)
            if chunk_content is None:
                continue

            if isinstance(chunk_content, str):
                # Accumulate raw content (no modifications during accumulation)
                raw_full_message += chunk_content
                step_message += chunk_content

                # Format the entire accumulated message (similar to streamlit)
                # This ensures consistency even if modifications affect earlier parts
                formatted_text = _modify_chunk(raw_full_message)
                formatted_text = _detect_image_name_and_move_to_public(formatted_text)

                # Only stream if the formatted text has changed
                # Stream only the delta (new portion) to avoid duplicates
                if formatted_text != last_formatted_text:
                    # Calculate delta: what's new since last display
                    if last_formatted_text and formatted_text.startswith(
                        last_formatted_text
                    ):
                        # Incremental: only stream the new part
                        delta = formatted_text[len(last_formatted_text) :]
                    else:
                        # Content changed in a non-sequential way (e.g., image processing)
                        # Stream the entire new content (this is rare)
                        delta = formatted_text

                    if delta:
                        await chainlit_step.stream_token(delta)
                        chainlit_step.output = formatted_text
                        last_formatted_text = formatted_text

        stream_completed = True

        # Final formatting for return values
        full_formatted = _modify_chunk(raw_full_message)
        full_formatted = _detect_image_name_and_move_to_public(full_formatted)

        # Ensure we've sent everything
        if full_formatted != last_formatted_text:
            remaining_delta = (
                full_formatted[len(last_formatted_text) :]
                if last_formatted_text
                and full_formatted.startswith(last_formatted_text)
                else full_formatted
            )
            if remaining_delta:
                await chainlit_step.stream_token(remaining_delta)
                chainlit_step.output = full_formatted

        # Format step_message for return (if needed)
        step_message = _modify_chunk(step_message)
        step_message = _detect_image_name_and_move_to_public(step_message)

        return full_formatted, step_message, raw_full_message

    except asyncio.CancelledError:
        # Propagate cancellation to properly handle stop button
        stream_completed = True
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
        raise
    finally:
        # Cancel heartbeat task
        stream_completed = True
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


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
        # ("<observation>", "```\n#Execute result\n"),
        # ("</observation>", "\n```\n"),
        ("<observation>", "\n"),
        ("</observation>", "\n"),
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
        if image_path.startswith(("./public/", "public/", "/public/")):
            # Normalize to absolute path
            if not image_path.startswith("/public/"):
                image_path = "/public/" + image_path.replace("./public/", "").replace(
                    "public/", ""
                )
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
            return f"[![{alt_text}](/public/{new_file_name})](/public/{new_file_name})[Download](/public/{new_file_name})"
        except Exception as e:
            print(f"Error moving image {image_path}: {e}")
            return match.group(0)

    return re.sub(image_pattern, replace_image, content)
