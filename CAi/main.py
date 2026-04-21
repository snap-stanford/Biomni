import os 
from CAi.CAi_agent.agent import A1pro
from CAi.config import LLM_BASE_URL,LLM_API_KEY

agent = A1pro(
    # llm="claude-sonnet-4-5-20250929",
    # llm = "Qwen/Qwen3-8B", # free
    # llm = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", # free
    # llm = "gemini-3-flash-preview-thinking",
    llm = "qwen3.6-plus",
    # llm = "glm-4.7-flash", # free
    # llm = "glm-5",
    # llm = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    # llm = "x-ai/grok-4-fast",
    source="Custom",
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    use_tool_retriever=False,
    expected_data_lake_files=[],
    auto_load_tools=True,  # 自动加载工具
    auto_load_skills=False,  # 不自动加载技能
)

agent.launch_new_gradio_demo(share=False)

