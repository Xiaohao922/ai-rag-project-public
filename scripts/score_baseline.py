# score_baseline.py
# 功能：打印基座模型评测结果，辅助人工判定通过/部分通过/失败
# 环境：Python 3.10+，无需额外依赖
# 用法：uv run python scripts/score_baseline.py
# 输入：data/eval/baseline_results.jsonl

import json

with open("data/eval/baseline_results.jsonl", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

print("基座模型评测结果")
print("=" * 80)

for item in items:
    print(f"[{item['id']:02d}] {item['category']} | 期望：{item['expected']}")
    print(f"问题：{item['instruction']}")
    print(f"回答：{item['answer'].replace(chr(10), ' ')[:300]}")
    print("请人工判定：通过 / 部分通过 / 失败")
    print("-" * 80)
