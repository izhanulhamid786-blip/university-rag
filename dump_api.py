import json
log_path = r"C:\Users\hp\.gemini\antigravity\brain\fc97f2d5-eb25-49ec-bb7a-a3bbd6ad2d3c\.system_generated\logs\transcript.jsonl"
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            step = json.loads(line)
            if step.get("type") == "TOOL_RESPONSE":
                calls = step.get("tool_calls", [])
                for call in calls:
                    if call.get("name") == "default_api:view_file":
                        out = call.get("response", {}).get("output", "")
                        if "api.py" in out:
                            print(out)
        except:
            pass
