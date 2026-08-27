# prepare_data.py
# 功能：将原始 SFT 数据切分为 train/valid/test 三个 jsonl
# 环境：Python 3.10+，无需额外依赖（仅用 json/random）
# 用法：在项目根目录执行 uv run python scripts/prepare_data.py
# 输入：data/sft/sft_train_clean.jsonl, data/sft/sft_eval_20.jsonl
# 输出：data/train.jsonl, data/valid.jsonl, data/test.jsonl

import json
import random

random.seed(42)


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def to_train_messages(item):
    user_content = item["instruction"]
    if item.get("input"):
        user_content += "\n" + item["input"]
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item["output"]},
        ]
    }


def to_eval_record(item):
    user_content = item["instruction"]
    if item.get("input"):
        user_content += "\n" + item["input"]
    return {
        "instruction": user_content,
        "expected": item["expected"],
        "category": item["category"],
        "note": item.get("note", ""),
    }


train = load_jsonl("data/sft/sft_train_clean.jsonl")
eval_items = load_jsonl("data/sft/sft_eval_20.jsonl")

random.shuffle(train)

train_split = train[:64]
valid_split = train[64:]

for filename, items in [
    ("data/train.jsonl", train_split),
    ("data/valid.jsonl", valid_split),
]:
    with open(filename, "w", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(to_train_messages(item), ensure_ascii=False) + "\n"
            )

with open("data/test.jsonl", "w", encoding="utf-8") as f:
    for item in eval_items:
        f.write(
            json.dumps(to_eval_record(item), ensure_ascii=False) + "\n"
        )

print(f"train: {len(train_split)}")
print(f"valid: {len(valid_split)}")
print(f"test:  {len(eval_items)}")
