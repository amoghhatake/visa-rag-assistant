from eval_runner import GOLD
from rag_pipeline import build_index_cached, retrieve

print("Building index...")
index = build_index_cached()

# Only questions where we know which document should answer them
answerable = [g for g in GOLD if g["type"] == "answerable"]


def rank_of_correct_source(query, persona, expected_source, use_filter, top_k=10):
    """Returns the 1-based rank of the first chunk whose file name contains
    expected_source. Returns None if it never appears in the top_k results."""
    retrieved = retrieve(query, index, persona, top_k=top_k, use_filter=use_filter)
    for i, (score, item) in enumerate(retrieved):
        if expected_source.lower() in item["source"].lower():
            return i + 1, float(score)
    return None, None


def run(use_filter, label):
    hit1 = hit3 = 0
    reciprocal_ranks = []
    rows = []

    for g in answerable:
        rank, score = rank_of_correct_source(g["q"], g["persona"], g["source"], use_filter)
        if rank is not None:
            reciprocal_ranks.append(1 / rank)
            if rank == 1:
                hit1 += 1
            if rank <= 3:
                hit3 += 1
        else:
            reciprocal_ranks.append(0)
        rows.append({"id": g["id"], "question": g["q"], "rank": rank, "similarity": score})

    n = len(answerable)
    mrr = sum(reciprocal_ranks) / n

    print(f"\n===== {label} =====")
    print(f"Hit@1 (correct doc ranked #1):     {hit1}/{n} ({100*hit1/n:.0f}%)")
    print(f"Hit@3 (correct doc in top 3):      {hit3}/{n} ({100*hit3/n:.0f}%)")
    print(f"MRR (mean reciprocal rank):        {mrr:.2f}")
    print()
    for r in rows:
        rank_str = f"rank {r['rank']}" if r["rank"] else "NOT FOUND in top 10"
        print(f"  [{r['id']}] {rank_str:<20} {r['question'][:60]}")

    return {"hit1": hit1, "hit3": hit3, "mrr": mrr, "n": n, "rows": rows}


if __name__ == "__main__":
    filtered = run(use_filter=True, label="FILTERED RETRIEVAL")
    unfiltered = run(use_filter=False, label="UNFILTERED RETRIEVAL")