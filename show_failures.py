import json

for label, path in [("FILTERED", "eval_filtered.json"), ("UNFILTERED", "eval_unfiltered.json")]:
    rows = json.load(open(path))
    print(f"\n===== {label}: answerable but wrong or unanswered =====")
    for r in rows:
        if r["type"] == "answerable" and not r["correct"]:
            print(f"\n[{r['id']}] {r['question']}")
            print(f"  abstained={r['abstained']} sim={r['similarity']} sources={r['sources']}")
            print(f"  answer: {r['answer'][:300]}")

rows = json.load(open("eval_filtered.json"))
print("\n===== FILTERED: answers the judge called unfaithful =====")
for r in rows:
    if r["type"] == "answerable" and not r["abstained"] and not r["faithful"]:
        print(f"\n[{r['id']}] sources={r['sources']}\n  answer: {r['answer'][:250]}")

print("\n===== SAFETY misses =====")
for r in json.load(open("eval_safety.json")):
    if not r["correct"]:
        print(f"[{r['id']}] {r['question']}\n  raw verdict: {r['raw_verdict'][:200]}")