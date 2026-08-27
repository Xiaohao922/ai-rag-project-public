# 风控 & AI 专业问答模型实践项目

> 从 RAG 知识库到模型对齐训练到安全评测的全链路 AI 产品实践

一个 AI 产品经理的完整实战项目，覆盖 **RAG 问答 → 查询改写 → SFT 微调 → DPO 对齐 → 安全评测** 全流程。目标不是做一个"能用"的模型，而是完整验证 AI 产品的**能力边界与知识边界**。

## 🏗️ 项目架构

```
应用层（Dify）              模型层（LoRA 微调）          安全验证
RAG 问答助手               SFT（行为塑造）              功能评测（20 条）
    ↓                        ↓                            ↓
查询改写 Chatflow          DPO（偏好对齐）             安全评测（18 条）
```

**一句话：从应用层做到模型层，最后对完整系统做安全评测，每一步都有实测数据和踩坑记录。**

## ✨ 核心成果

| 指标 | 数值 |
|---|---|
| RAG 基线召回准确率 | 66.7% → 混合检索改善 |
| 功能准确率（最终模型） | **75%**（20 条） |
| 安全防御率（最终模型） | **94.4%**（18 条攻击） |
| 边界拒答 | **100%** |
| 综合评分 | **87.6 分** |

## 🧠 核心洞察（本项目最有价值的产出）

1. **知识靠数据供给，行为靠训练塑造**：SFT/DPO 改变不了训练数据没覆盖的知识（RAP 三代都答不对）
2. **DPO 必须在 SFT 基础上做**：直接在基座做 DPO 角色全面退化，必须 SFT → DPO
3. **RAG 的"知道不知道"分层解决**：检索层不识别超纲，需 Prompt 层拒答兜底
4. **对齐训练不必然付对齐税**：加了正常对照组验证，无过度拒答

## 📁 目录结构

```
├── README.md                # 本文件（仓库门面）
├── docs/                    # 项目文档
│   ├── PRD-风控AI专业问答模型.md
│   ├── 项目总结-风控AI专业问答模型.md
│   ├── 复现指南.md
│   ├── 全局路线图.md
│   ├── 01-rag-qa/          # 项目一：RAG 问答助手
│   ├── 02-query-rewrite/    # 项目二：查询改写 RAG
│   ├── 03-sft-finetuning/   # 项目三：SFT 微调
│   ├── 04-dpo-alignment/    # 项目四：DPO 对齐
│   ├── 05-security-eval/    # 项目五：综合评测
├── data/                    # 训练/测试数据（JSONL）
│   ├── sft/                 # SFT 训练/验证集
│   ├── dpo/                 # DPO 偏好/边界集
│   └── eval/                # 功能/安全测试集 + 评测结果
├── scripts/                 # 训练/评测脚本（当前在 Mac 本地 sft/，待同步）
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── eval_function.py
│   └── eval_security.py
├── configs/                 # 配置文件（参数、Prompt 模板）
├── results/                 # 评测结果
└── .gitignore
```

> 说明：训练/评测脚本当前位于 Mac 本地 `sft/` 目录（`run_dpo_on_sft_smoke.py`、`run_dpo_on_sft_30steps.py`、`run_dpo_on_sft_eval.py`、`run_security_test.py` 等）。上传前需将这些脚本复制到 `scripts/` 并按复现指南重命名。

## 🚀 快速开始

### 环境要求

- Python 3.10+
- MacBook Pro M4 Pro 24GB（或等效算力）
- PyTorch 2.13 + Transformers 5.15 + TRL 1.10 + PEFT 0.20

### 安装依赖

```bash
# 创建隔离环境（项目四用，DPO 训练）
python -m venv .venv-dpo
source .venv-dpo/bin/activate
pip install torch transformers trl peft datasets
```

### 数据准备

```bash
# 数据在 data/ 目录，已按 SFT/DPO/评测 分类
ls data/
```

### 训练

```bash
# SFT 训练（项目三，MLX 环境）
uv run python scripts/train_sft.py

# DPO 训练（项目四，需先合并 SFT adapter，PyTorch 环境）
uv run python scripts/merge_sft_adapter.py
uv run python scripts/train_dpo.py
```

### 评测

```bash
# 功能评测（20 条）
uv run python scripts/eval_function.py

# 安全评测（18 条）
uv run python scripts/eval_security.py
```

## 🛠️ 技术栈

| 层 | 选型 |
|---|---|
| 基座模型 | Qwen2.5-1.5B-Instruct |
| RAG 平台 | Dify（对话应用 + Chatflow） |
| 向量化 | text-embedding-v4（阿里云通义） |
| 重排序 | qwen3-rerank |
| 训练框架 | MLX（SFT）+ Transformers/TRL/PEFT（DPO） |
| 训练方式 | LoRA |
| 运行环境 | MacBook Pro M4 Pro 24GB |

## 📚 文档导航

- [PRD（产品需求文档）](docs/PRD-风控AI专业问答模型.md)
- [项目总结](docs/项目总结-风控AI专业问答模型.md)
- [复现指南](docs/复现指南.md)
- [全局路线图](docs/全局路线图.md)
- [项目五-综合评测报告](docs/05-security-eval/项目五-AI模型综合评测报告.md)

## 📄 许可证

仅供学习与项目展示使用，模型权重遵循各基座模型开源协议。

## 👤 关于作者

安全方向 AI 产品经理，通过本项目完整实践 AI 产品从应用层到模型层到安全验证的全链路能力。
