import json
rows = {r["id"]: r for r in json.load(open("eval_filtered.json"))}
for i in ["A2", "A3", "A5", "B2", "B5"]:
    r = rows[i]
    print(f"[{i}] correct={r['correct']} abstained={r['abstained']}\n  {r['answer'][:250]}\n")