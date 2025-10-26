import json
from datetime import datetime
from app.config.settings import Config

def log_feedback(event: dict):
    e = dict(event)
    e["ts"] = datetime.utcnow().isoformat()
    with open(Config.RL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
