import chainlit as cl
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from biomni.agent import A1_HITS
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
import os
import re
import shutil
import random
import string
import base64
from PIL import Image
from biomni.config import default_config
from biomni.tool.memory import save_conversation
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.base import BaseStorageClient
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
import asyncio
import logging
import concurrent.futures

CURRENT_ABS_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_biomni_data_path() -> str:
    """Determine the local biomni data path, preferring existing directories."""
    candidates = []
    for env_key in ("BIOMNI_DATA_PATH", "BIOMNI_PATH"):
        env_value = os.getenv(env_key)
        if env_value:
            candidates.append(env_value)

    # Known shared locations and repo-relative defaults
    candidates.extend(
        [
            "/workdir_efs/jhjeon/Biomni/biomni_data",
            os.path.join(CURRENT_ABS_DIR, "..", "biomni_data"),
            os.path.join(CURRENT_ABS_DIR, "biomni_data"),
        ]
    )

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        abs_candidate = os.path.abspath(candidate)
        if abs_candidate in seen:
            continue
        seen.add(abs_candidate)
        if os.path.isdir(abs_candidate):
            return abs_candidate

    fallback = os.path.abspath(os.path.join(CURRENT_ABS_DIR, "..", "biomni_data"))
    os.makedirs(fallback, exist_ok=True)
    return fallback


# Configuration
LLM_MODEL = "gemini-3-pro-preview"
# LLM_MODEL = "grok-4-fast"
BIOMNI_DATA_PATH = _resolve_biomni_data_path()
PUBLIC_DIR = os.path.join(os.getcwd(), "public")
CHAINLIT_DB_PATH = os.path.join(CURRENT_ABS_DIR, "chainlit.db")
STREAMING_MAX_TIMEOUT = 3600  # Maximum streaming timeout in seconds (1 hour)

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
LOG_FILE_PATH = f"{CURRENT_ABS_DIR}/chainlit_stream.log"

# Create logger and set up handlers directly
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicates
logger.handlers.clear()

# Create and configure handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler for global log
file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Prevent propagation to root logger to avoid duplicate logs
logger.propagate = False

logger.info(f"Logger initialized. Log file: {LOG_FILE_PATH}")

# Thread-specific log handler management
_thread_log_handler = None


def add_thread_log_handler(thread_id: str):
    """Add a thread-specific log handler to capture logs for individual chat sessions.

    Args:
        thread_id: The unique thread/session identifier
    """
    global _thread_log_handler

    # Remove previous thread-specific handler if exists
    if _thread_log_handler:
        logger.removeHandler(_thread_log_handler)
        _thread_log_handler.close()

    # Create new thread-specific log file
    thread_log_path = f"{CURRENT_ABS_DIR}/chainlit_logs/{thread_id}/chainlit_stream.log"
    os.makedirs(os.path.dirname(thread_log_path), exist_ok=True)

    # Add new handler for this thread
    _thread_log_handler = logging.FileHandler(thread_log_path, mode="a")
    _thread_log_handler.setLevel(logging.INFO)
    _thread_log_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_thread_log_handler)

    logger.info(f"Thread-specific log handler added for thread: {thread_id}")
    logger.info(f"Thread log file: {thread_log_path}")


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
    valid_logins = {
        ("admin", "admin"): "admin",
        ("admin@example.com", "admin"): "admin@example.com",
        ("admin@biomni.com", "admin"): "admin@biomni.com",
        ("jslink", "5fdf7e4a-6632-48b1-a9c8-b79d9a9be2e0"): "jslink",
    }

    identifier = valid_logins.get((username, password))

    if identifier:
        return cl.User(
            identifier=identifier,  # 각 사용자마다 고유한 identifier 사용
            metadata={"role": "admin", "provider": "credentials"},
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

    # Add thread-specific log handler
    add_thread_log_handler(dir_name)

    os.chdir(log_dir)
    print("current dir", os.getcwd())
    logger.info(f"Chat session started for thread: {dir_name}")
    cl.user_session.set("message_history", [])
    cl.user_session.set("uploaded_files", [])

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

    # Add thread-specific log handler
    add_thread_log_handler(dir_name)

    os.chdir(log_dir)
    print("current dir", os.getcwd())
    logger.info(f"Chat session resumed for thread: {dir_name}")
    
    # Restore uploaded files from directory
    uploaded_files = []
    if os.path.exists(log_dir):
        # Scan directory for user-uploaded files (exclude system files)
        exclude_files = {'chainlit_stream.log', 'conversation_history.txt'}
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            if os.path.isfile(file_path) and filename not in exclude_files:
                uploaded_files.append(filename)
    
    cl.user_session.set("uploaded_files", uploaded_files)
    if uploaded_files:
        logger.info(f"Restored {len(uploaded_files)} uploaded files: {', '.join(uploaded_files)}")


@cl.on_message
async def main(user_message: cl.Message):
    """
    Handle user messages and process them through the agent.

    Args:
        user_message: The user's message from Chainlit UI.
    """
    print("current dir:", os.getcwd())
    user_data = await _process_user_message(user_message)
    message_history = _update_message_history(user_data)
    agent_input = _convert_to_agent_format(message_history)

    await _process_agent_response(agent_input, message_history)


async def _process_user_message(user_message: cl.Message) -> dict:
    """Process user message and handle file uploads.

    Returns:
        dict with 'text' and optional 'images' keys
    """
    user_prompt = user_message.content.strip()
    images = []

    # Image file extensions
    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".tiff",
        ".tif",
    }

    # Get current uploaded files list from session
    uploaded_files = cl.user_session.get("uploaded_files", [])

    # Process uploaded files
    for file in user_message.elements:
        os.system(f"cp {file.path} '{file.name}'")
        
        # Add to uploaded files list if not already present
        if file.name not in uploaded_files:
            uploaded_files.append(file.name)

        # Check if it's an image file
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext in image_extensions:
            try:
                # Extract image resolution
                img_width, img_height = None, None
                try:
                    with Image.open(file.path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    print(f"Could not extract image dimensions for {file.name}: {e}")

                # Read image and encode to base64
                with open(file.path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                # Determine MIME type
                mime_type_map = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".bmp": "image/bmp",
                    ".webp": "image/webp",
                    ".tiff": "image/tiff",
                    ".tif": "image/tiff",
                }
                mime_type = mime_type_map.get(file_ext, "image/jpeg")

                images.append(
                    {"name": file.name, "data": f"data:{mime_type};base64,{image_data}"}
                )

                # Add image info with resolution to prompt
                if img_width and img_height:
                    user_prompt += f"\n - user uploaded image file: {file.name} (resolution: {img_width}x{img_height} pixels)\n"
                else:
                    user_prompt += f"\n - user uploaded image file: {file.name}\n"
            except Exception as e:
                print(f"Error processing image {file.name}: {e}")
                user_prompt += f"\n - user uploaded data file: {file.name}\n"
        else:
            user_prompt += f"\n - user uploaded data file: {file.name}\n"

    # Update uploaded files in session
    cl.user_session.set("uploaded_files", uploaded_files)

    # Always append all uploaded files information to the prompt
    # This ensures the AI remembers all files throughout the conversation
    if uploaded_files:
        files_info = "\n\n[System: Available uploaded files in working directory: " + ", ".join(uploaded_files) + "]"
        user_prompt += files_info

    return {"text": user_prompt, "images": images}


def _update_message_history(user_data: dict) -> list:
    """Update and return message history.

    Args:
        user_data: dict with 'text' and optional 'images' keys
    """
    message_history = cl.user_session.get("message_history", [])
    message_history.append(
        {
            "role": "user",
            "content": user_data["text"],
            "images": user_data.get("images", []),
        }
    )
    return message_history


def _convert_to_agent_format(message_history: list) -> list:
    """Convert message history to agent input format."""
    agent_input = []
    for message in message_history:
        if message["role"] == "user":
            # Check if there are images to include
            images = message.get("images", [])
            if images:
                # Create content as a list with text and images
                content = [{"type": "text", "text": message["content"]}]
                for img in images:
                    content.append(
                        {"type": "image_url", "image_url": {"url": img["data"]}}
                    )
                agent_input.append(HumanMessage(content=content))
            else:
                # Text only
                agent_input.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            agent_input.append(AIMessage(content=message["content"]))
    return agent_input


async def _sync_generator_to_async(sync_gen):
    """Convert a sync generator to async generator.

    This allows us to use async for loop which automatically checks
    for CancelledError at each iteration.
    """
    loop = asyncio.get_running_loop()

    def get_next_item():
        try:
            return next(sync_gen)
        except StopIteration:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            # Run next() in executor to avoid blocking
            item = await loop.run_in_executor(executor, get_next_item)
            if item is None:
                break
            yield item


def _extract_text_from_content(content):
    """Extract text from message content (handles both string and list formats).

    Args:
        content: Either a string or a list with text and image_url dicts

    Returns:
        str: The text content
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract text from list format
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    else:
        return str(content)


async def _process_agent_response(agent_input: list, message_history: list):
    """Process agent response and handle streaming."""

    with open(f"conversation_history.txt", "a") as f:
        user_content_text = _extract_text_from_content(agent_input[-1].content)
        f.write(user_content_text + "\n")

    logger.info("[RESPONSE] Starting agent response processing")

    try:
        async with cl.Step(name="Plan and execute") as chainlit_step:
            await chainlit_step.update()
            sync_message_stream = agent.go_stream(agent_input)
            # Convert sync generator to async
            message_stream = _sync_generator_to_async(sync_message_stream)
            full_message, step_message, raw_full_message = await _handle_message_stream(
                message_stream, chainlit_step, sync_message_stream
            )

        final_message = _extract_final_message(raw_full_message)
        final_message = _detect_image_name_and_move_to_public(final_message)

        await cl.Message(content=final_message).send()

        print(os.getcwd())
        print(final_message)

        with open(f"conversation_history.txt", "a") as f:
            f.write(raw_full_message + "\n")
        message_history.append({"role": "assistant", "content": raw_full_message})

        # Save conversation to memory
        try:
            user_message_content = message_history[-2]["content"] # The message before the one we just appended
            save_conversation(user_message_content, final_message)
        except Exception as e:
            logger.error(f"Failed to save conversation to memory: {e}")

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
        error_message = _replace_biomni(error_message)
        await cl.Message(content=f"❌ **오류**: {error_message}").send()
        message_history.append({"role": "assistant", "content": error_message})


async def _handle_message_stream(message_stream, chainlit_step, sync_generator):
    """Handle streaming messages from the agent using async for loop.

    This approach is much simpler and automatically handles CancelledError
    at each await point in the async for loop.

    Args:
        message_stream: Async generator (converted from sync)
        chainlit_step: Chainlit Step object
        sync_generator: Original sync generator for cleanup
    """
    raw_full_message = ""
    step_message = ""
    current_step = 1
    last_formatted_text = ""
    image_cache = {}
    last_chunk_time = [
        asyncio.get_event_loop().time()
    ]  # Mutable to share with heartbeat
    stream_done = [False]  # Mutable flag to stop heartbeat
    last_heartbeat_msg = [""]  # Mutable to share with main loop

    logger.info("[STREAM] Starting message stream processing")

    async def heartbeat():
        """Send periodic 'working...' messages if no chunks received for a while."""
        heartbeat_count = 0

        while not stream_done[0]:
            try:
                await asyncio.sleep(2)  # Check every 2 seconds
            except asyncio.CancelledError:
                break

            if stream_done[0]:
                break

            elapsed = asyncio.get_event_loop().time() - last_chunk_time[0]

            # Only show message if waiting > 5 seconds
            if elapsed >= 5:
                heartbeat_count += 1
                elapsed_int = int(elapsed)

                logger.info(
                    f"[HEARTBEAT #{heartbeat_count}] Showing working message (elapsed: {elapsed_int}s)"
                )

                # Create new heartbeat message
                new_msg = f"\n\n_⏳ Still working... ({elapsed_int} seconds elapsed)_"

                try:
                    # Get current output and append heartbeat
                    current_output = chainlit_step.output or last_formatted_text

                    # Remove old heartbeat if present
                    if last_heartbeat_msg[0] and current_output.endswith(
                        last_heartbeat_msg[0]
                    ):
                        current_output = current_output[: -len(last_heartbeat_msg[0])]

                    # Set new output with heartbeat
                    chainlit_step.output = current_output + new_msg
                    await chainlit_step.update()  # UI 업데이트 필수!
                    last_heartbeat_msg[0] = new_msg
                    logger.debug(f"[HEARTBEAT] Updated output with message")
                except Exception as e:
                    logger.warning(f"[HEARTBEAT] Failed to update: {e}")
            else:
                # Silent heartbeat for debugging
                logger.debug(
                    f"[HEARTBEAT] Check (elapsed: {elapsed:.1f}s, waiting for 5s)"
                )

    # Start heartbeat task
    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        # async for automatically checks for CancelledError at each iteration!
        async for chunk in message_stream:
            # Update last chunk time
            last_chunk_time[0] = asyncio.get_event_loop().time()
            # At this await point, CancelledError is automatically raised
            # if the task is cancelled (page refresh or stop button)

            this_step = chunk[1][1]["langgraph_step"]
            if this_step != current_step:
                step_message = ""
                current_step = this_step

            chunk_content = _extract_chunk_content(chunk)
            if chunk_content is None:
                continue

            if not isinstance(chunk_content, str):
                chunk_content = chunk_content[0]["text"] + "\n"

            raw_full_message += chunk_content
            step_message += chunk_content

            # Remove heartbeat message if present (new chunk arrived!)
            if last_heartbeat_msg[0]:
                logger.debug("[STREAM] Removing heartbeat message (new chunk arrived)")
                current_output = chainlit_step.output or ""
                if current_output.endswith(last_heartbeat_msg[0]):
                    chainlit_step.output = current_output[: -len(last_heartbeat_msg[0])]
                last_heartbeat_msg[0] = ""  # Clear heartbeat message

            # Format and detect images
            formatted_text = _modify_chunk(raw_full_message)
            formatted_text = _detect_image_name_and_move_to_public(
                formatted_text, image_cache
            )

            # Only stream if changed
            if formatted_text != last_formatted_text:
                prev_output = last_formatted_text

                if prev_output and formatted_text.startswith(prev_output):
                    delta = formatted_text[len(prev_output) :]
                else:
                    delta = formatted_text

                if delta:
                    # This await also checks for CancelledError
                    await chainlit_step.stream_token(delta)
                    chainlit_step.output = formatted_text
                    last_formatted_text = formatted_text

        logger.info("[STREAM] Stream completed successfully")

        # Stop heartbeat
        stream_done[0] = True

        # Remove any remaining heartbeat message
        if last_heartbeat_msg[0]:
            logger.debug("[STREAM] Removing heartbeat message at completion")
            current_output = chainlit_step.output or ""
            if current_output.endswith(last_heartbeat_msg[0]):
                chainlit_step.output = current_output[: -len(last_heartbeat_msg[0])]
            last_heartbeat_msg[0] = ""

        # Final formatting
        full_formatted = _modify_chunk(raw_full_message)
        full_formatted = _detect_image_name_and_move_to_public(
            full_formatted, image_cache
        )

        # Final update if needed
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

        step_message = _modify_chunk(step_message)
        step_message = _detect_image_name_and_move_to_public(step_message, image_cache)

        return full_formatted, step_message, raw_full_message

    except asyncio.CancelledError:
        logger.info("[STREAM] CancelledError - closing generator and stopping")
        stream_done[0] = True

        # Close the sync generator explicitly to clean up resources
        if sync_generator and hasattr(sync_generator, "close"):
            try:
                sync_generator.close()
                logger.info("[STREAM] Sync generator closed successfully")
            except Exception as e:
                logger.warning(f"[STREAM] Failed to close generator: {e}")
        raise  # Re-raise to propagate cancellation

    except Exception as e:
        logger.error(f"[STREAM] Error during streaming: {e}", exc_info=True)
        stream_done[0] = True

        # Also close generator on error
        if sync_generator and hasattr(sync_generator, "close"):
            try:
                sync_generator.close()
            except:
                pass
        raise

    finally:
        # Always stop and cleanup heartbeat task
        stream_done[0] = True
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
    """Extract final message from the last <solution> tag pair."""
    # Close unclosed solution tag if needed
    if "<solution>" in step_message and "</solution>" not in step_message:
        step_message += "</solution>"

    # Find all solution tag matches
    solution_matches = list(
        re.finditer(r"<solution>(.*?)</solution>", step_message, re.DOTALL)
    )

    # Return content from the last match, or original message if no match found
    if solution_matches:
        return solution_matches[-1].group(1).strip()
    else:
        return step_message


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

    # First, handle observation tags with proper formatting
    # This prevents backticks and other special characters from being misinterpreted as markdown
    observation_pattern = r"<observation>(.*?)</observation>"

    def format_observation(match):
        observation_content = match.group(1).strip()

        # Check if it's an error message
        is_error = any(
            keyword in observation_content
            for keyword in [
                "Error",
                "error",
                "Exception",
                "Traceback",
                "Failed",
                "Execution halted",
            ]
        )

        if is_error:
            return f"\n\n**❌ Execution Error:**\n```\n{observation_content}\n```\n"

        # Check if observation contains file listings (CSV or other files)
        has_file_listings = (
            "**Newly created files:**" in observation_content
            or "**Other files:**" in observation_content
        )

        if has_file_listings:
            # Split into text before files and file section
            lines = observation_content.split("\n")
            text_before_files = []
            file_section_start = -1

            # Find where file section starts
            for i, line in enumerate(lines):
                if "**Newly created files:**" in line or "**Other files:**" in line:
                    file_section_start = i
                    break
                text_before_files.append(line)

            # Format text before files (regular text, not in code block)
            formatted_text = "\n".join(text_before_files).strip()

            # Format file section
            if file_section_start >= 0:
                file_section = "\n".join(lines[file_section_start:])

                # Detect CSV content in file section (lines with many commas)
                file_lines = file_section.split("\n")
                formatted_file_lines = []
                csv_block_lines = []
                in_csv_block = False

                for line in file_lines:
                    # Check if line looks like CSV data (starts with comma or has many commas)
                    is_csv_line = line.strip().startswith(",") or (
                        "," in line and line.count(",") >= 3
                    )

                    if is_csv_line:
                        if not in_csv_block:
                            # Start new CSV block
                            in_csv_block = True
                        csv_block_lines.append(line)
                    else:
                        if in_csv_block:
                            # End CSV block - format and add it
                            if csv_block_lines:
                                csv_lines = csv_block_lines[:10]
                                if len(csv_block_lines) > 10:
                                    csv_lines.append("... (truncated: more lines)")
                                csv_content = "\n".join(csv_lines)
                                formatted_file_lines.append(
                                    f"```csv\n{csv_content}\n```"
                                )
                                csv_block_lines = []
                            in_csv_block = False
                        formatted_file_lines.append(line)

                # Handle remaining CSV block
                if csv_block_lines:
                    csv_lines = csv_block_lines[:10]
                    if len(csv_block_lines) > 10:
                        csv_lines.append("... (truncated: more lines)")
                    csv_content = "\n".join(csv_lines)
                    formatted_file_lines.append(f"```csv\n{csv_content}\n```")

                formatted_file_section = "\n".join(formatted_file_lines)

                # Combine formatted text and file section
                if formatted_text:
                    return f"\n\n**✅ Execution Result:**\n\n{formatted_text}\n\n{formatted_file_section}\n"
                else:
                    return f"\n\n**✅ Execution Result:**\n\n{formatted_file_section}\n"
            else:
                # No file section found, format as regular text
                return f"\n\n**✅ Execution Result:**\n\n{formatted_text}\n"
        else:
            # No file listings, format as regular code block
            return f"\n\n**✅ Execution Result:**\n```\n{observation_content}\n```\n"

    retval = re.sub(observation_pattern, format_observation, retval, flags=re.DOTALL)

    # Then handle other tags
    tag_replacements = [
        ("<execute>", "\n```CODE_TYPE\n"),
        ("</execute>", "```\n"),
        ("<solution>", ""),
        ("</solution>", ""),
    ]

    for tag1, tag2 in tag_replacements:
        if tag1 in retval:
            retval = retval.replace(tag1, tag2)

    # Handle existing CODE_TYPE placeholders in already generated code blocks
    retval = _replace_code_type_placeholders(retval)

    # Replace biomni imports with hits imports in code blocks
    retval = _replace_biomni(retval)

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


def _replace_biomni(content: str) -> str:
    """Replace all occurrences of 'biomni' and 'Biomni' with 'hits' in code blocks and text."""
    # Pattern to find code blocks (both with specific language and generic)
    code_block_pattern = r"```(\w+)?\n(.*?)```"

    def replace_imports_in_code(match):
        language = match.group(1) if match.group(1) else ""
        code_content = match.group(2)

        # Replace ALL occurrences, case-insensitive for biomni
        modified_code = re.sub(r"\bbiomni\b", "hits", code_content, flags=re.IGNORECASE)
        # Also handle lowercase/uppercase word-boundary safe (edge: Biomni in class names etc.)
        # But above regex with IGNORECASE handles both "biomni" and "Biomni"

        if language:
            return f"```{language}\n{modified_code}```"
        else:
            return f"```\n{modified_code}```"

    # Apply replacement to all code blocks
    content = re.sub(
        code_block_pattern, replace_imports_in_code, content, flags=re.DOTALL
    )

    # Additional replacements for text outside code blocks:
    # 1. Replace 'biomni.' pattern (e.g., biomni.tool.omics -> hits.tool.omics)
    content = re.sub(r"biomni\.", "hits.", content, flags=re.IGNORECASE)

    # 2. Replace biomni/Biomni in paths and general text with 'hits'
    # This handles cases like /home/ec2-user/Biomni_HITS/... -> /home/ec2-user/hits_HITS/...
    # Also handles biomni_hits_test -> hits_hits_test
    content = re.sub(r"biomni", "hits", content, flags=re.IGNORECASE)

    return content


def _detect_image_name_and_move_to_public(
    content: str, image_cache: dict = None
) -> str:
    """
    Detect images in markdown text, move them to public folder with random prefix.

    Args:
        content: Markdown text content
        image_cache: Dictionary to cache already moved images {original_path: public_path}

    Returns:
        Modified markdown text with updated image paths
    """
    if image_cache is None:
        image_cache = {}

    public_dir = PUBLIC_DIR
    os.makedirs(public_dir, exist_ok=True)

    # Pattern to find markdown images, excluding those already with download functionality
    image_pattern = (
        r'(?<!\[)!\[([^\]]*)\]\(([^)]+?)(?:\s+"[^"]*")?\)(?!\]\([^\)]*\)<br>)'
    )

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
            file_name = os.path.basename(image_path)
            return (
                f"[![{alt_text}]({image_path})]({image_path})<br>"
                f'<a href="{image_path}" download="{file_name}">Download</a>'
            )

        # Check if file exists
        if not os.path.exists(image_path):
             # Try to find it relative to the current working directory (chainlit_logs/thread_id)
             # or relative to the project root (CURRENT_ABS_DIR)
            
            # 1. Check relative to CWD (already done by exists check if path is relative, but explicit check for absolute path construction might be needed)
            cwd_path = os.path.abspath(image_path)
            if os.path.exists(cwd_path):
                image_path = cwd_path
            else:
                # 2. Check relative to CURRENT_ABS_DIR (project root where run.py is, or parent of it)
                # Note: CURRENT_ABS_DIR in this file is chainlit/ directory. 
                # But the agent execution happens in chainlit_logs/{thread_id}. 
                # Sometimes paths are relative to project root.
                
                # Try relative to project root (parent of chainlit dir)
                project_root = os.path.dirname(CURRENT_ABS_DIR)
                root_path = os.path.join(project_root, image_path)
                if os.path.exists(root_path):
                    image_path = root_path
                
        if not os.path.exists(image_path):
            return match.group(0)

        # Check cache first to avoid duplicate copies
        if image_path in image_cache:
            public_path = image_cache[image_path]
            file_name = os.path.basename(public_path)
            return (
                f"[![{alt_text}]({public_path})]({public_path})<br>"
                f'<a href="{public_path}" download="{file_name}">Download</a>'
            )

        # Generate random prefix and new filename
        random_prefix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )
        file_name = os.path.basename(image_path)
        new_file_name = f"{random_prefix}_{file_name}"
        new_file_path = os.path.join(public_dir, new_file_name)

        try:
            shutil.copy2(image_path, new_file_path)
            public_path = f"/public/{new_file_name}"
            image_cache[image_path] = public_path  # Cache the result
            print("copied image to", new_file_path)
            return (
                f"[![{alt_text}]({public_path})]({public_path})<br>"
                f'<a href="{public_path}" download="{new_file_name}">Download</a>'
            )
        except Exception as e:
            print(f"Error moving image {image_path}: {e}")
            return match.group(0)

    return re.sub(image_pattern, replace_image, content)