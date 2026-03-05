import json
import os
import threading
from flask import Flask, request, jsonify

from biomni.config import default_config
from biomni.agent import A1
from prompt import make_prompt

app = Flask(__name__)

_agent = None
_lock = threading.Lock()


def extract_json(output) -> dict:
    raw = output[1] if isinstance(output, (list, tuple)) and len(output) > 1 else str(output)

    i = raw.find("{")
    if i == -1:
        raise ValueError(f"No JSON object found in agent output. raw[:200]={raw[:200]!r}")
    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(raw[i:])
    
    return obj


@app.post("/init")
def init():
    """
    Example body:
    {
      "dataset": "melanoma CyCIF",
      "db_llm": "claude-haiku-4-5-20251001",
      "agent_llm": "claude-sonnet-4-6",
      "timeout_seconds": 1200,
      "download_data": false,
      "expected_data_lake_files": [],
      "minimal_mode": true,
      "force": false
    }
    """
    cfg = request.get_json(force=True, silent=True) or {}
    dataset = cfg.get("dataset", "melanoma CyCIF")
    force = bool(cfg.get("force", False))

    global _agent
    with _lock:
        if _agent is not None and not force:
            return jsonify({"ok": True, "message": "Already initialized"}), 200

        try:
            default_config.llm = cfg.get("db_llm", os.getenv("BIOMNI_DB_LLM", "claude-haiku-4-5-20251001"))
            default_config.timeout_seconds = int(cfg.get("timeout_seconds", os.getenv("BIOMNI_TIMEOUT_SECONDS", "1200")))

            download_data = bool(cfg.get("download_data", False))
            agent_llm = cfg.get("agent_llm", os.getenv("BIOMNI_AGENT_LLM", "claude-sonnet-4-6"))
            minimal_mode = bool(cfg.get("minimal_mode", True))
            expected_files = cfg.get("expected_data_lake_files", [])

            if download_data:
                _agent = A1(llm=agent_llm, minimal_mode=minimal_mode)
            else:
                _agent = A1(llm=agent_llm, expected_data_lake_files=expected_files, minimal_mode=minimal_mode)

            prompt_text = make_prompt(dataset)
            _agent.set_system_message_prefix(prompt_text)

            if bool(cfg.get("write_full_system_prompt", True)):
                full_prompt = _agent.get_full_system_prompt()
                with open(cfg.get("full_system_prompt_path", "full_system_prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(full_prompt)

            return jsonify({"ok": True, "message": "Agent initialized", "dataset": dataset}), 200

        except Exception as e:
            _agent = None
            return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "agent_initialized": _agent is not None})


@app.post("/label")
def label():
    body = request.get_json(force=True, silent=True)
    if not body or "label" not in body or not isinstance(body["label"], list) or not body["label"]:
        return jsonify({"error": "Expected {'label': ['CD8A','PDL1']}"}), 400

    with _lock:
        agent = _agent
    if agent is None:
        return jsonify({"error": "Agent not initialized. Call POST /init first."}), 400

    input_str = json.dumps({"label": body["label"]})
    try:
        output = agent.go(input_str)
        parsed = extract_json(output)
        return jsonify({"result": parsed}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)