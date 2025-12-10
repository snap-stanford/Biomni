from biomni.agent import A1_HITS
import os
import shutil
from datetime import datetime
import pytz
import time
from langchain_core.messages import SystemMessage, HumanMessage
from biomni.llm import get_llm
import markdown
import logging

# Logger 설정 (시스템 프롬프트를 초록색으로 출력)
class GreenFormatter(logging.Formatter):
    """초록색으로 로그를 출력하는 포맷터"""
    GREEN = '\033[32m'
    RESET = '\033[0m'
    
    def format(self, record):
        # 초록색으로 메시지 포맷팅
        record.msg = f"{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(GreenFormatter('%(message)s'))
logger.addHandler(handler)

# LangSmith tracing (only enable if API key is set)
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "hklee"
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    print("⚠️  Warning: LANGSMITH_API_KEY not set. LangSmith tracing disabled.")

# Create directory with current Korean time
korea_tz = pytz.timezone("Asia/Seoul")
current_time = datetime.now(korea_tz)
dir_name = "logs/" + current_time.strftime("%Y%m%d_%H%M%S")

# os.system("rm -r logs/2025*")
os.makedirs("logs", exist_ok=True)
os.makedirs(dir_name, exist_ok=True)

# Store original directory before changing
original_dir = os.getcwd()
os.chdir(dir_name)

# Copy image file if it exists in parent directory
# image_files = ["rep1.png", "rep2.png", "rep3.png"]
image_files = ["wb_full.png"]
for image_file in image_files:
    parent_image_path = os.path.join(original_dir, image_file)
    if os.path.exists(parent_image_path):
        shutil.copy2(parent_image_path, image_file)
        print(f"✅ Copied {image_file} to current directory")
    elif not os.path.exists(image_file):
        print(f"⚠️ Warning: {image_file} not found in current or parent directory")

t1 = time.time()
# llm = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
# llm = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
llm = "gemini-3-pro-preview"
# llm = "solar-pro2"
# llm = "mistral-small-2506"
# Use chainlit/biomni_data path to avoid downloading data
chainlit_biomni_data_path = os.path.join(original_dir, "chainlit", "biomni_data")
if os.path.exists(chainlit_biomni_data_path):
    # If chainlit/biomni_data exists, use its parent directory as path
    # so that biomni_data/data_lake will be found at chainlit/biomni_data/data_lake
    agent_path = os.path.join(original_dir, "chainlit")
    # Pass empty list to skip downloading - files already exist in chainlit/biomni_data/data_lake
    expected_data_lake_files = []
else:
    # Fallback to current directory if chainlit path doesn't exist
    agent_path = "./"
    expected_data_lake_files = None  # Will download if needed
agent = A1_HITS(
    path=agent_path,
    llm=llm,
    # allow_resources=["proteomics", "support_tools", "bio_image_processing"],
    use_tool_retriever=True,
    resource_filter_config_path=os.path.join(original_dir, "chainlit", "resource.yaml"),
    expected_data_lake_files=expected_data_lake_files,
)
image_file = 'rep1_full.png'
# user_command = input("Enter your command: ")
user_command = 'Please analyze the provided Western blot image, which consists of 3 experimental repetitions. Each repetition includes four conditions: control, P144, TGF-β1, and Tβ1Ab. The targets are PSMAD2, SMAD2, and GAPDH'
# user_command = f"""
# 이 웨스턴블롯 이미지에서 맨 윗줄의 밴드의 세기를 정량화 해줘. control을 기준으로 정규화해서 상대적인 발현량을 계산해야해 
#  - user uploaded data file: {image_file}
# """
# user_command = f"""
# 이미지의 녹색세포와 빨간색 세포 개수를 세서 그래프로 비교해줘
# 컨트롤은 컨트롤끼리 비교하고, 실험군은 실험군 끼리 비교해서 각각 바그래프로 보여줘
#  - user uploaded data file: {image_file}
# """

# user_command = f"""
# Run densitometry on Rep1, Rep2, and Rep3, identifying the bands from top to bottom as PSMAD2, SMAD2, and GAPDH.
# Each of the image has 12 bands.

# 1. Use find_roi_from_image function.
# 2. Quantify the intensity of them.
# 3. Caclulate PSMAD2 / (SMAD2 / GAPDH).
# 4. Normalize values relative to the control.
# 4. Generate a bar graph with error bars. Result graph should have 4 bars.
#  - user uploaded data file: {image_files}
# """

with open("logs.txt", "w") as f1, open("system_prompt.txt", "w") as f2:
    for idx, output in enumerate(agent.go(user_command)):
        print("====================", idx, "====================")
        if idx == 0:
            # 시스템 프롬프트를 logger로 초록색 출력
            logger.info(output)
            f2.write(output)
            f2.flush()  # 즉시 파일에 쓰기
        else:
            f1.write(output + "\n")
            f1.flush()  # 즉시 파일에 쓰기

t2 = time.time()
print(f"Elapsed time: {t2 - t1:.2f} seconds")
