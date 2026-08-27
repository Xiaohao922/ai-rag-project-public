# run_hf_sft_v4.py
# 功能：SFT v4 正式训练（最终版，含边界样本，200 steps）
# 环境：需要 PyTorch + Transformers + PEFT + TRL + Datasets
#   pip install torch transformers peft trl datasets
# 用法：在项目根目录执行
#   uv run --with torch,transformers,peft,trl,datasets python scripts/run_hf_sft_v4.py
# 模型：Qwen2.5-1.5B-Instruct（HF 格式，需提前下载到 models/ 目录）
# 输入：data/sft_text_v4/train.jsonl, data/sft_text_v4/valid.jsonl
# 输出：hf_sft_v4_boundary_output/
#
# 硬件：macOS Apple Silicon (MPS) 或 CPU；显存不够时已开启 gradient_checkpointing
# LoRA 参数：r=8, alpha=16, dropout=0.05, target=all linear projections
# 训练参数：max_steps=200, lr=1e-5, constant scheduler, warmup=10, max_length=512

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from pathlib import Path

# 路径（请按实际存放位置修改）
MODEL_DIR = "models/Qwen2.5-1.5B-Instruct-hf"
TRAIN_FILE = "data/sft_text_v4/train.jsonl"
VALID_FILE = "data/sft_text_v4/valid.jsonl"
OUTPUT_DIR = "hf_sft_v4_boundary_output"

for p in [TRAIN_FILE, VALID_FILE]:
    assert Path(p).is_file(), f"数据文件不存在: {p}"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": VALID_FILE},
)
print("train:", len(dataset["train"]), "| validation:", len(dataset["validation"]))

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

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=200,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    lr_scheduler_type="constant",
    warmup_steps=10,
    max_length=512,
    dataset_text_field="text",
    packing=False,
    logging_strategy="steps",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=4,
    report_to="none",
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    seed=20260826,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("开始 HF-SFT v4 训练...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("训练完成:", OUTPUT_DIR)
