# Tianchi 评论观点挖掘 - Qwen3.5-9B LoRA 项目整理版

本项目是对“【天池经典打榜赛】赛道六-评论观点挖掘赛”的 LoRA 路线整理，目标是可复现、可持续迭代

## 1. 我们在做什么

- 任务：从评论中抽取四元组  
  `AspectTerms, OpinionTerms, Categories, Polarities`
- 方案：用 `Qwen/Qwen3.5-9B` 做指令式序列生成，输出标准 JSON 数组，再转提交格式。

## 2. 微调架构（核心）

- 基座模型：`Qwen3.5-9B`（Causal LM）
- 训练范式：`QLoRA`（4bit NF4 量化 + LoRA 适配器）
- 关键技术栈：
  - `bitsandbytes` 4bit 量化加载
  - `peft` LoRA 注入
  - `transformers` + `Trainer` 训练
- LoRA 注入模块：
  - `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

## 3. 已验证可跑成功的模型配置与分数

| 配置 | 说明 | Fold | Precision | Recall | F1 |
| --- | --- | --- | ---: | ---: | ---: |
| `configs/qwen35_9b_lora_quick.yaml` | 冒烟子集训练 | 0 | 0.5350 | 0.3661 | 0.4347 |
| `configs/qwen35_9b_lora.yaml` | 标准训练（2 epoch） | 0 | 0.7719 | 0.7778 | 0.7748 |
| `configs/qwen35_9b_lora_aggressive.yaml` | 激进训练（3 epoch, r=64） | 0 | 0.7854 | 0.7968 | 0.7911 |

当前已验证最佳离线 fold0：**F1 = 0.7911**

完整台账见 [results/experiment_results.md](./results/experiment_results.md) 与 [results/experiment_results.csv](./results/experiment_results.csv)。

> 说明：以上分数均来自真实云端日志，均为“训练+验证完整跑通”结果。

## 4. 目录结构

```text
qwen35_9b_lora_github/
  configs/            # 各训练/推理配置
  scripts/            # 训练、续训、推理、检查脚本
  src/                # 公共工具与评分
  data/               # 数据占位（默认不入库）
  results/            # 已跑通实验记录
  requirements.txt
  .gitignore
```

## 5. 环境依赖

推荐（云端）：

- Python 3.10+
- CUDA 12.x
- PyTorch 2.7+ / 2.9+（均可）
- GPU：A800 80G 或同级显存

安装：

```bash
cd qwen35_9b_lora_github
bash scripts/setup_env.sh
```

## 6. 数据与模型准备

### 6.1 数据

把比赛数据放到 `data/`（文件名必须一致）：

- `Train_reviews.csv`
- `Train_labels.csv`
- `Test_reviews.csv`
- `Result(example).csv`

### 6.2 本地模型（推荐离线）

将 `Qwen3.5-9B` 放到云端路径，例如：

- `/root/data/models/Qwen3.5-9B`

并设置：

```bash
export MODEL_NAME_OR_PATH=/root/data/models/Qwen3.5-9B
export LOCAL_FILES_ONLY=1
```

## 7. 训练与推理命令

### 7.1 快速冒烟

```bash
bash scripts/run_train_fold.sh configs/qwen35_9b_lora_quick.yaml 0
```

### 7.2 标准训练（fold0）

```bash
bash scripts/run_train_fold.sh configs/qwen35_9b_lora.yaml 0
```

### 7.3 激进训练（fold0）

```bash
bash scripts/run_train_fold.sh configs/qwen35_9b_lora_aggressive.yaml 0
```

### 7.4 断点续训

```bash
bash scripts/run_resume_fold.sh configs/qwen35_9b_lora.yaml 0 outputs/qwen35_lora_fold0/checkpoint-xxx
```

### 7.5 生成测试提交

```bash
bash scripts/run_predict_test.sh \
  configs/qwen35_9b_lora.yaml \
  outputs/qwen35_lora_fold0/adapter \
  outputs/submit_qwen35_lora_fold0.csv
```

## 8. 主要配置说明

- `qwen35_9b_lora_quick.yaml`：快速验证流程，样本子集，调试优先
- `qwen35_9b_lora.yaml`：标准训练配置（已验证）
- `qwen35_9b_lora_aggressive.yaml`：更激进训练（已验证）
- `qwen35_9b_lora_infer_boost.yaml`：推理增强（多路生成投票，待继续验证）
- `qwen35_9b_lora_aggressive_v2.yaml`：激进训练+推理增强（待继续验证）

