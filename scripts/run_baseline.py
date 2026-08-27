# run_baseline.py
# 功能：使用未微调的基座模型跑 20 条评测，生成基线结果
# 环境：需要 mlx_lm（仅 macOS Apple Silicon），pip install mlx-lm
# 用法：在项目根目录执行 uv run --with mlx-lm python scripts/run_baseline.py
# 模型：Qwen2.5-1.5B-Instruct-4bit（需提前下载到本地 models/ 目录）
# 输入：data/test.jsonl
# 输出：data/eval/baseline_results.jsonl

import json
from mlx_lm import load, generate

# 基座模型路径（请按实际存放位置修改）
MODEL = "models/Qwen2.5-1.5B-Instruct-4bit"

model, tokenizer = load(MODEL)


def make_prompt(question):
    messages = [
        {
            "role": "system",
            "content": (
                "你是风控与 AI 专业知识助手。请准确回答问题；"
                "如果超出风控与 AI 知识范围，请明确说明无法回答，不要编造。"
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


with open("data/test.jsonl", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

with open("data/eval/baseline_results.jsonl", "w", encoding="utf-8") as out:
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

print("完成：data/eval/baseline_results.jsonl")
