import json, os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def judge_answer(question, context, answer):
    prompt = f"""You are grading an AI assistant's answer for a visa information system.

Question: {question}

Retrieved context (what the assistant was allowed to use):
{context}

Assistant's answer: {answer}

Grade the answer on three criteria. Respond with ONLY a JSON object, no other text:
{{
  "faithful": true or false,
  "correct": true or false,
  "relevant": true or false,
  "reason": "one short sentence explaining your grades"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",     # cheap, fast, good enough for grading
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"faithful": None, "correct": None, "relevant": None, "reason": "parse failed"}


def evaluate_file(path, label):
    rows = json.load(open(path))
    answered = [r for r in rows if r["type"] == "answerable" and not r["abstained"]]

    results = []
    print(f"\n===== {label}: GPT-4o-mini judge evaluation =====")
    for r in answered:
        grade = judge_answer(r["question"], r.get("context", ""), r["answer"])
        results.append({**grade, "id": r["id"], "question": r["question"], "answer": r["answer"]})
        print(f"[{r['id']}] faithful={grade['faithful']}  correct={grade['correct']}  "
              f"relevant={grade['relevant']}")
        print(f"      reason: {grade['reason']}")

    n = len(results)
    if n:
        faithful_n = sum(1 for r in results if r["faithful"] is True)
        correct_n = sum(1 for r in results if r["correct"] is True)
        relevant_n = sum(1 for r in results if r["relevant"] is True)
        print(f"\nFaithful: {faithful_n}/{n} ({100*faithful_n/n:.0f}%)")
        print(f"Correct:  {correct_n}/{n} ({100*correct_n/n:.0f}%)")
        print(f"Relevant: {relevant_n}/{n} ({100*relevant_n/n:.0f}%)")

    with open(f"llm_judge_{label.lower()}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    evaluate_file("eval_filtered_gen_metrics_llama3.2_3b.json", "FILTERED")