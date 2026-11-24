import json
import pandas as pd

json_file = "ACHI_analysis/synthetic_data/ACHI_2026_902_session-1_sessiondata.json"

with open(json_file, "r", encoding="utf-8") as f:
    session_data = json.load(f)

events = pd.DataFrame(session_data)
print(events["eventName"].value_counts())

seen = set()
for record in session_data:
    name = record["eventName"]
    if name not in seen:
        seen.add(name)
        print("\n---", name, "(", record["runId"], ") ---")
        print(record["data"])
        