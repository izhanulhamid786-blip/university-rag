import json
import os
import re

FILES_TO_RESTORE = [
    r"c:\Users\hp\university_rag\api.py",
    r"c:\Users\hp\university_rag\rag\llm.py",
    r"c:\Users\hp\university_rag\rag\pipeline.py",
    r"c:\Users\hp\university_rag\rag\text_cleanup.py",
    r"c:\Users\hp\university_rag\frontend\src\api.js",
    r"c:\Users\hp\university_rag\frontend\src\App.jsx",
]

LOG_PATHS = [
    r"C:\Users\hp\.gemini\antigravity\brain\0f0468b0-9bcc-446e-80e3-0bd74f81c98f\.system_generated\logs\transcript.jsonl",
    r"C:\Users\hp\.gemini\antigravity\brain\fc97f2d5-eb25-49ec-bb7a-a3bbd6ad2d3c\.system_generated\logs\transcript.jsonl",
]

restored = set()

for log_path in LOG_PATHS:
    if not os.path.exists(log_path):
        continue
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
            except:
                continue
                
            if step.get("type") == "TOOL_RESPONSE":
                calls = step.get("tool_calls", [])
                for call in calls:
                    if call.get("name") == "default_api:view_file":
                        output = call.get("response", {}).get("output", "")
                        
                        # The output format is:
                        # Created At: ...
                        # Completed At: ...
                        # File Path: `file:///c:/Users/hp/university_rag/api.py`
                        # ...
                        # The following code has been modified to include a line number...
                        # <lines>
                        # The above content...
                        
                        path_match = re.search(r"File Path:\s*`file:///([^`]+)`", output, re.IGNORECASE)
                        if not path_match:
                            continue
                        
                        # Convert forward slashes to backslashes
                        path = path_match.group(1).replace("/", "\\").lower()
                        
                        target_file = None
                        for target in FILES_TO_RESTORE:
                            if target.lower() == path:
                                target_file = target
                                break
                        
                        if target_file and target_file not in restored:
                            # Extract code blocks
                            # Look for lines like "1: import json" and strip the "1: "
                            code_lines = []
                            in_code = False
                            for out_line in output.splitlines():
                                if "The following code has been modified to include a line number" in out_line:
                                    in_code = True
                                    continue
                                if "The above content shows the entire, complete file contents" in out_line or "The above content does NOT show the entire file contents" in out_line:
                                    in_code = False
                                    continue
                                
                                if in_code:
                                    # Match line number and colon
                                    m = re.match(r"^\d+:\s?(.*)$", out_line)
                                    if m:
                                        code_lines.append(m.group(1))
                            
                            if code_lines:
                                with open(target_file, "w", encoding="utf-8") as out_f:
                                    out_f.write("\n".join(code_lines) + "\n")
                                print(f"Restored: {target_file}")
                                restored.add(target_file)
                                
for f in FILES_TO_RESTORE:
    if f not in restored:
        print(f"Failed to find: {f}")
