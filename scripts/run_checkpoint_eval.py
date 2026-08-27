# run_checkpoint_eval.py
# 功能：评测某个 SFT checkpoint（MLX LoRA adapter）
# 环境：需要 mlx_lm（仅 macOS Apple Silicon）
# 用法：uv run --with mlx-lm python scripts/run_checkpoint_eval.py <checkpoint_name>
#   例：uv run --with mlx-lm python scripts/run_checkpoint_eval.py eval_100
# 模型：Qwen2.5-1.5B-Instruct-4bit + LoRA adapter
# 输入：data/test_meta.jsonl
# 输出：data/eval/<checkpoint_name>_results.jsonl

import json
import sys
from mlx_lm import load, generate

if len(sys.argv) != 2:
    print("用法：uv run --with mlx-lm python scripts/run_checkpoint_eval.py <checkpoint_name>")
    raise SystemExit(1)

checkpoint = sys.argv[1]

# 路径（请按实际存放位置修改）
BASE = "models/Qwen2.5-1.5B-Instruct-4bit"
ADAPTER = f"adapters/{checkpoint}"
OUTPUT = f"data/eval/{checkpoint}_results.jsonl"

model, tokenizer = load(BASE, adapter_path=ADAPTER)


def make_prompt(question):
    messages = [
        {
            "role": "system",
            "content": (
                "你是风控与 AI 专业知识助手。"
                "请准确回答问题；如果超出风控与 AI 知识范围，"
                "请明确说明无法回答，不要编造。"
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


with open("data/test_meta.jsonl", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

with open(OUTPUT, "w", encoding="utf-8") as out:
    for i, item in enumerate(items, 1):
        answer = generate(
            model,
            tokenizer,
            prompt=make_prompt(item["instruction"]),
            max_tokens=300,
            verbose=False,
        )
        result = {
            "id": i,
            "instruction": item["instruction"],
            "expected": item["expected"],
            "category": item["category"],
            "answer": answer,
        }
        out.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(items)}] {item['instruction']}")
        print(answer)
        print("-" * 60)

print(f"完成：{OUTPUT}")
