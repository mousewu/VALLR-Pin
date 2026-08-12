# Stage-II 解耦训练方案

## 目标

Stage-II 独立学习 `带噪无声调拼音 -> 中文句子`。主训练集只需要中文文本，不需要视频、
Stage-I 检查点或字符候选，因此数据规模可以独立扩展。

## 1. 构建干净文本语料

复制配置并填写字幕、访谈转写、口语文本和通用文本路径：

```bash
cp configs/stage2_text.example.yaml configs/stage2_text.local.yaml
python -m vallr_pin.cli build-stage2-text --config configs/stage2_text.local.yaml
```

支持逐行文本、JSONL 和 TSV/CSV。构建器会：

- 按句切分，只保留中文并执行整句多音字 G2P；
- 文本去重，限制句长；
- 按 `document_id` 稳定切分，防止同一文档跨 train/val/test；
- 用 `exclude_paths` 排除 Stage-I dev/test 中完全相同的句子；
- 输出 `text + pinyin`，不预先复制噪声版本。

## 2. 在线噪声

训练时按 `configs/llm_sft.yaml` 动态执行音节替换、删除、插入、交换和 `<mask>`。默认：

- 25% 干净输入，学习“无需纠正时保持不动”；
- 50% 轻度噪声；
- 25% 重度噪声；
- 每条文本每 epoch 生成 `variants_per_text` 个确定但不同的版本。

验证集的噪声由固定 seed 生成，不随 epoch 改变，保证可比较。

## 3. 单机多卡 LoRA

```bash
torchrun --standalone --nproc_per_node=8 -m vallr_pin.cli sft \
  --config configs/llm_sft.yaml
```

`batch_size` 是每卡 batch；有效 batch 为
`batch_size × GPU 数 × accum_steps`。每个进程各加载一份 4B 基座模型，LoRA 梯度由 DDP 同步。
训练只对 assistant 答案计算损失，提示词和拼音输入全部 mask。

使用 ms-swift 前先固化在线噪声：

```bash
python -m vallr_pin.cli materialize-stage2 \
  --config configs/llm_sft.yaml \
  --input data/stage2_text/train.jsonl \
  --output data/stage2_text/train.messages.jsonl --variants 2
python -m vallr_pin.cli sft --config configs/llm_sft.yaml \
  --train-jsonl data/stage2_text/train.messages.jsonl --print-swift
```

## 4. SwanLab

在 Stage-I 或 Stage-II YAML 中设置：

```yaml
swanlab:
  enabled: true
  project: VALLR-Pin
  experiment_name: stage2-pinyin-llm
  mode: online       # 也可用 offline/local
  logdir: swanlog
```

先在训练节点执行 `pip install -r requirements-swanlab.txt` 和 `swanlab login`。DDP 模式只有
rank 0 初始化实验并记录跨卡聚合的 epoch 指标，
不会产生多个重复实验。

## 5. 可选真实错误校准

`decode-ckpts` 与 `build-llm-data` 仍可从 Stage-I 输出构造小规模校准集，但应在大规模纯文本
LoRA 之后以较小学习率短暂训练。它不是 Stage-II 主数据，也不应与最终 VSR test 混用。

## 6. 评估

分别报告：

- 固定合成噪声验证集的字符错误率；
- Stage-I dev 的端到端 `视频 -> 拼音 -> LLM -> 文字` CER；
- Stage-I 拼音 SER；
- 干净拼音样本被改错的 over-correction rate；
- 删除、插入、替换和同音歧义子集指标。
