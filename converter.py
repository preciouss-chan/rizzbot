import json
from collections import defaultdict

pairs = []
by_prompt = defaultdict(lambda: {"up": [], "down": []})

with open("rizz_logs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        e = json.loads(line)
        if not e.get("user_input") or not e.get("response"):
            continue
        if e.get("feedback") == "👍":
            by_prompt[e["user_input"]]["up"].append(e["response"])
        elif e.get("feedback") == "👎":
            by_prompt[e["user_input"]]["down"].append(e["response"])

def as_messages(user, assistant):
    return [
        {"role": "user", "content": f"Her: {user}\nMe:"},
        {"role": "assistant", "content": assistant}
    ]

for prompt, buckets in by_prompt.items():
    ups, downs = buckets["up"], buckets["down"]
    if not ups or not downs:
        continue
    # make a few random pairings
    for chosen in ups:
        for rejected in downs[:2]:   # cap fanout
            pairs.append({
                "input": { "messages": [ {"role":"user", "content": f"Her: {prompt}\nMe:"} ] },
                "preferred_output":  as_messages(prompt, chosen)[1:],       # assistant-only
                "non_preferred_output": as_messages(prompt, rejected)[1:]
            })

with open("train_dpo.jsonl", "w", encoding="utf-8") as out:
    for p in pairs:
        out.write(json.dumps(p, ensure_ascii=False) + "\n")

print("pairs:", len(pairs))

