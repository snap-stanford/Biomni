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
    # msg = cl.Message(content='![예시 이미지](./public/example.png)')
    # await msg.send()
    # return
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
        os.system(f"cp {file.path} '{file.name}'")
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
        chainlit_step.output = "Initilizing..."
        await chainlit_step.update()
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
            full_message = detect_image_name_and_move_to_public(full_message)
            chainlit_step.output = full_message
            await chainlit_step.update()

    step_message = detect_image_name_and_move_to_public(step_message)

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


def detect_image_name_and_move_to_public(content):
    public_dir = f"{current_abs_dir}/public"  # 이 경로를 원하는 값으로 수정하세요
    
    """
    마크다운 텍스트에서 이미지를 찾아서 ./public 폴더로 옮기고 랜덤 prefix를 추가합니다.
    
    Args:
        content (str): 마크다운 텍스트
        
    Returns:
        str: 수정된 마크다운 텍스트
    """
    import re
    import os
    import shutil
    import random
    import string
    
    # public 디렉토리가 없으면 생성
    # public_dir = "./public"  # 이 줄은 제거됨
    os.makedirs(public_dir, exist_ok=True)
    
    # 마크다운 이미지 패턴 찾기: ![alt text](image_path) 또는 ![alt text](image_path "title")
    image_pattern = r'!\[([^\]]*)\]\(([^)]+?)(?:\s+"[^"]*")?\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        image_path = match.group(2).strip()
        
        # URL이거나 이미 public 폴더에 있는 경우는 건드리지 않음
        if image_path.startswith(('http://', 'https://', './public/', 'public/')):
            return match.group(0)
        
        # 파일이 존재하는지 확인
        if not os.path.exists(image_path):
            return match.group(0)  # 파일이 없으면 원본 그대로 반환
        
        # 랜덤 prefix 생성 (6자리)
        random_prefix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        
        # 파일 확장자 추출
        file_name = os.path.basename(image_path)
        name, ext = os.path.splitext(file_name)
        
        # 새 파일명 생성
        new_file_name = f"{random_prefix}_{file_name}"
        new_file_path = os.path.join(public_dir, new_file_name)
        
        try:
            # 파일 복사
            shutil.copy2(image_path, new_file_path)
            
            # 새로운 마크다운 이미지 링크 반환
            return f"![{alt_text}](./public/{new_file_name})"
        except Exception as e:
            print(f"Error moving image {image_path}: {e}")
            return match.group(0)  # 에러가 발생하면 원본 그대로 반환
    
    # 모든 이미지 패턴을 찾아서 교체
    modified_content = re.sub(image_pattern, replace_image, content)
    
    return modified_content