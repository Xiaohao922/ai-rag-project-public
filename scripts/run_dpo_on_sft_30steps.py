# run_dpo_on_sft_30steps.py
# 功能：DPO-on-SFT 正式训练（30 steps，最终版）
# 环境：需要 PyTorch + Transformers + PEFT + TRL + Datasets
#   pip install torch transformers peft trl datasets
# 用法：在项目根目录执行
#   uv run --with torch,transformers,peft,trl,datasets python scripts/run_dpo_on_sft_30steps.py
# 模型：SFT v4 合并后的模型（hf_sft_v4_merged/）
# 输入：data/dpo/train.jsonl, data/dpo/valid.jsonl
# 输出：dpo_on_sft_30steps_output/
#
# 硬件：macOS Apple Silicon (MPS) 或 CPU
# LoRA 参数：r=8, alpha=16, dropout=0.05, target=all linear projections
# 训练参数：max_steps=30, lr=5e-6, beta=0.1, constant scheduler, max_length=512
# 数据规格：train 35 条, validation 4 条, columns=[prompt, chosen, rejected]

import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# 路径（请按实际存放位置修改）
MODEL_DIR = "hf_sft_v4_merged"
TRAIN_FILE = "data/dpo/train.jsonl"
VALID_FILE = "data/dpo/valid.jsonl"
OUTPUT_DIR = "dpo_on_sft_30steps_output"

# 训练前硬检查
required_files = [
    f"{MODEL_DIR}/config.json",
    f"{MODEL_DIR}/tokenizer_config.json",
    TRAIN_FILE,
    VALID_FILE,
]

for path in required_files:
    assert os.path.isfile(path), f"文件不存在: {path}"
    assert os.path.getsize(path) > 0, f"文件为空: {path}"
    print("已确认:", path)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": VALID_FILE},
)

assert len(dataset["train"]) == 35, len(dataset["train"])
assert len(dataset["validation"]) == 4, len(dataset["validation"])
assert dataset["train"].column_names == ["prompt", "chosen", "rejected"]

print("train:", len(dataset["train"]))
print("validation:", len(dataset["validation"]))
print("columns:", dataset["train"].column_names)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16)
model.config.use_cache = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=30,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    lr_scheduler_type="constant",
    beta=0.1,
    max_length=512,
    logging_steps=5,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=1,
    report_to="none",
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    seed=20260826,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("开始 DPO-on-SFT 训练...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("DPO-on-SFT 训练完成")
print("输出目录:", OUTPUT_DIR)
