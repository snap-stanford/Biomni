import json
import os
from flask import Flask, request, jsonify
from biomni.config import default_config

from biomni.agent import A1

from prompt import PROMPT

default_config.llm = os.getenv("BIOMNI_DB_LLM", "claude-haiku-4-5-20251001")
default_config.timeout_seconds = int(os.getenv("BIOMNI_TIMEOUT_SECONDS", "1200"))


DOWNLOAD_DATA = os.getenv("DOWNLOAD_DATA", "0") == "1"

if DOWNLOAD_DATA:
    agent = A1(llm=os.getenv("BIOMNI_AGENT_LLM", "claude-sonnet-4-6"), minimal_mode=True)
else:
    agent = A1(
        llm=os.getenv("BIOMNI_AGENT_LLM", "claude-sonnet-4-6"),
        expected_data_lake_files=[],
        minimal_mode=True,
    )

agent.set_system_message_prefix(PROMPT)

prompt_text = agent.get_full_system_prompt()
with open("full_system_prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt_text)

app = Flask(__name__)

def extract_json(output) -> dict:
    raw = output[1]
    payload = raw[7:-3]
    return json.loads(payload)


@app.route("/label", methods=["POST"])
def infer():
    body = request.get_json(force=True, silent=True)
    if not body or "label" not in body or not isinstance(body["label"], list) or not body["label"]:
        return jsonify({"error": "Expected JSON body like {'label': ['CD8A','PDL1']}"}), 400

    input_str = json.dumps({"label": body["label"]})
    try:
        output = agent.go(input_str)
        parsed = extract_json(output)
        return jsonify({"result": parsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("10) starting flask on :8000")
    app.run(host="0.0.0.0", port=8000, debug=False)