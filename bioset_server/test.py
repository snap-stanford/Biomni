from biomni.config import default_config
from biomni.agent import A1
from prompt import PROMPT
import time
import json

default_config.llm="claude-haiku-4-5-20251001" #database querying
default_config.timeout_seconds = 1200

# config, thinking enabled, disabled, adaptive
# effort high
# output config

DOWNLOAD_DATA=False
agent = None
if DOWNLOAD_DATA:
    agent = A1(llm="claude-sonnet-4-6", minimal_mode=True)#agent reasoning
else:
    agent = A1(llm="claude-sonnet-4-6",expected_data_lake_files = [],minimal_mode=True) 

agent.set_system_message_prefix(PROMPT)
prompt = agent.get_full_system_prompt()
with open("full_system_prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

inputs = [
    '{"label": ["CD8A", "PDL1"]}',
    '{"label": ["CD3E", "CD8A", "MART1", "PDL1"]}',
    '{"label": ["Hoechst", "MART1", "lamin-ABC", "BAF1"]}',
    '{"label": ["PD1", "CD8A", "MART1", "PDL1"]}',
    '{"label": ["CD8A", "MART1", "CD31", "CD103"]}',
    '{"label": ["CD8A", "CD11c", "MX1", "PD1", "LAG3", "Hoechst"]}',
    '{"label": ["PRAME", "B-catenin", "LysozymeC", "Ki67", "CD3E", "FOXP3"]}',
]

start_time = time.time()
for input_str in inputs:
    output = agent.go(input_str)
end_time = time.time()

print(json.loads(output[1][7:-3]))

print(f"Total execution time: {end_time - start_time:.2f} seconds")
print(f"Average execution time per input: {(end_time - start_time) / len(inputs):.2f} seconds")