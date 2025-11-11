from biomni.agent import A1_HITS
import os
from datetime import datetime
import pytz
import time
from langchain_core.messages import SystemMessage, HumanMessage
from biomni.llm import get_llm
import markdown

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "hklee"

# Create directory with current Korean time
korea_tz = pytz.timezone("Asia/Seoul")
current_time = datetime.now(korea_tz)
dir_name = "logs/" + current_time.strftime("%Y%m%d_%H%M%S")

os.system("rm -r logs/2025*")
os.makedirs("logs", exist_ok=True)
os.makedirs(dir_name, exist_ok=True)
os.chdir(dir_name)

t1 = time.time()
llm = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
# llm = "gemini-2.5-pro"
# llm = "solar-pro2"
# llm = "mistral-small-2506"
agent = A1_HITS(
    path="./",
    llm=llm,
    allow_resources=["proteomics", "support_tools", "bio"],
    use_tool_retriever=True,
)
user_command = """/workdir_efs/jaechang/work2/thermofisher/data/data_FC_WT_AD_simple.xlsx 파일은 정상 및 알츠하이머(5xFAD) Mouse Frontal Cortex의 Proteomics 분석 결과야. 
파일을 생성하게 되면 모두 현재 폴더에 저장해줘. 
이 데이터에 대해서 다음의 전처리 및 분석을 수행해줘.

1. 데이터 전처리 (결측치 처리, 필터링 등):
- 각 group 별로 결측치의 비율이 50% 이상인 경우 해당 단백질 제거
- 남은 단백질에 대해서 결측치는 0.0으로 채워줘

2. 아래 분석 수행해줘.
- Volcano plot 그리기
- 통계적 유의성이 나타난 상위 50개 단백질들에 대한 Dendrogram을 포함한 Heatmap
- UMAP 기반 샘플 차원 축소
- 통계적 유의성이 나타난 상위 10개 단백질들에 대한 Box plot
- 통계적으로 유의하게 upregulated 및 downregulated 된 단백질들에 대한 Enriched Pathway 분석
- 위 결과를 바탕으로 upregulated된 단백질 중 신약후보 유전자 5개 추천.

3. 위 결과를 바탕으로 추천된 단백질 중 하나에 대해서 바인딩 포켓 분석 수행.
"""

with open("logs.txt", "w") as f1, open("system_prompt.txt", "w") as f2:
    for idx, output in enumerate(agent.go(user_command)):
        print("====================", idx, "====================")
        if idx == 0:
            f2.write(output)
            f2.flush()  # 즉시 파일에 쓰기
        else:
            f1.write(output + "\n")
            f1.flush()  # 즉시 파일에 쓰기

t2 = time.time()
print(f"Elapsed time: {t2 - t1:.2f} seconds")
