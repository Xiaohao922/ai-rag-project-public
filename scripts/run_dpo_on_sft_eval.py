# run_dpo_on_sft_eval.py
# 功能：DPO-on-SFT 最终评测（加载 SFT 合并模型 + DPO LoRA adapter，跑 20 条评测）
# 环境：需要 PyTorch + Transformers + PEFT
#   pip install torch transformers peft
# 用法：在项目根目录执行
#   uv run --with torch,transformers,peft python scripts/run_dpo_on_sft_eval.py
# 模型：hf_sft_v4_merged + dpo_on_sft_30steps_output/checkpoint-30
# 输入：data/sft/sft_eval_20.jsonl
# 输出：data/eval/dpo_on_sft_30_results.jsonl

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 路径（请按实际存放位置修改）
BASE = "hf_sft_v4_merged"
ADAPTER = "dpo_on_sft_30steps_output/checkpoint-30"
TEST_FILE = "data/sft/sft_eval_20.jsonl"
OUTPUT = "data/eval/dpo_on_sft_30_results.jsonl"

for path in [
    Path(BASE) / "config.json",
    Path(BASE) / "tokenizer_config.json",
    Path(ADAPTER) / "adapter_config.json",
    Path(ADAPTER) / "adapter_model.safetensors",
    Path(TEST_FILE),
]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"文件不存在或为空：{path}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)
print("base:", BASE)
print("adapter:", ADAPTER)

tokenizer = AutoTokenizer.from_pretrained(str(BASE))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, str(ADAPTER))
model.to(device)
model.eval()

items = [
    json.loads(line)
    for line in Path(TEST_FILE).read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if len(items) != 20:
    raise ValueError(f"测试集应为20条，实际为{len(items)}条")


def generate_answer(question):
    messages = [
        {
            "role": "system",
            "content": (
                "你是风控与 AI 专业知识助手。"
                "请准确回答问题；如果缺乏可靠依据或超出风控与 AI 范围，"
                "请明确说明无法确定，不要编造业务规则。"
            ),
        },
        {"role": "user", "content": question},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


with open(OUTPUT, "w", encoding="utf-8") as f:
    for index, item in enumerate(items, start=1):
        answer = generate_answer(item["instruction"])
        result = {
            "id": index,
            "instruction": item["instruction"],
            "expected": item["expected"],
            "category": item["category"],
            "answer": answer,
        }
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"[{index}/20] {item['instruction']}")
        print(answer)
        print("-" * 60)

print("评测完成：", OUTPUT)
