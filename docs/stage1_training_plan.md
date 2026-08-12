# VALLR-Pin Stage-I 从零训练方案

## 1. 训练目标与对照实验

第一轮不要使用 CNC-AV 伪标签。固定同一份 ROI、同一 speaker-independent 划分、同一视觉
编码器和相同 optimizer steps，依次训练。Pinyin-CTC 是解耦主方案，其余两组是对照：

| 实验 | 文字权重 | 拼音权重 | 最佳模型指标 | 目的 |
|---|---:|---:|---|---|
| Text-CTC | 1.0 | 0.0 | dev CER | 直接文字基线 |
| Pinyin-CTC | 0.0 | 1.0 | dev SER | 检验音节中介本身 |
| Joint CTC | 0.1 | 0.9 | dev SER，同时保存最优 CER | 检查文字辅助是否改善共享表示 |

模型约 53M 参数（字符表按 7k 估算，虽然 Pinyin-only 不反传字符头）。输入是连续 25fps、96×96 灰度嘴部 ROI，训练时随机裁
88×88；前端和 SANM 都保持逐帧时间长度。Pinyin-only 必须报告无声调拼音 SER 及
S/D/I 分解，并用 CTC prefix beam 报告拼音 N-best oracle；只有 Char-CTC 和字符辅助实验
才报告 Stage-I 字符 CER。所有方案都应报告 Stage-II 最终 CER。

## 2. 数据角色

- **CN-CVS**：主训练源，覆盖说话人和真实环境最多；建议采样概率 0.70。
- **CMLR**：高质量有文本监督数据；建议采样概率 0.30，并保留独立测试集做跨库评估。
- **CNC-AV / CN-Celeb-AV**：若发布包没有句级转写，不属于监督 VSR 数据。第一轮完全排除；
  后续只能加入置信度至少 0.95、且 ASR 与人工抽检通过的 pseudo label，采样概率从 0.05 开始。
- **其他中文 VSR 数据**：manifest 构建器不硬编码数据集名称；JSONL、CSV/TSV、Kaldi 和 sidecar
  标注均可作为独立 `sources` 项加入。默认 Pinyin-only 配置的 `source_weights: {}` 会自动使用
  manifest 中全部来源。只有需要重平衡时才填写显式来源权重。

数据许可可能限制商业用途，原始文件和生成 ROI 不应提交 Git。

## 3. 统一 manifest

复制并修改配置：

```bash
cp configs/corpora.example.yaml configs/corpora.local.yaml
python scripts/build_stage1_manifests.py configs/corpora.local.yaml
```

支持四种标注入口：

- `jsonl`：字段名由 `*_field` 指定；
- `delimited`：TSV/CSV 的列号由 `*_column` 指定；
- `kaldi`：目录内必须有 `text`、`utt2spk`、`video.scp`；
- `sidecar`：每个视频对应一个文本文件，可用 `text_prefix: "Text:"`。

输出记录至少包含：

```json
{"id":"cn_cvs:utt1","video":"/abs/utt1.npy","text":"今天天气很好","speaker_id":"cn_cvs:spk1","source":"cn_cvs","split":"train","n_frames":42}
```

如果同一公众人物可能同时出现在 CN-CVS 与 CNC-AV，需先用人脸 embedding 聚类得到跨库
`global_speaker_id`，写回 JSONL。仅给 speaker 加数据集前缀无法阻止跨库身份泄漏。

## 4. ROI 预处理

如果拿到的是原始视频，先让 manifest 指向视频，再批量产生轨迹和 ROI：

```bash
python scripts/preprocess_stage1_roi.py data/stage1/train.jsonl \
  --out-manifest data/stage1/train.roi.jsonl \
  --out-root /DATA/VALLR_PIN_DERIVED \
  --face-model models/face_landmarker.task --workers 8 --min-coverage 0.95
```

同样处理 dev/test。轨迹默认保留，后续修改 ROI 几何无需重新做人脸检测。不要用
`prepare_manifest.py` 的“下半脸中心粗裁”跑正式实验；正式数据必须使用关键点稳定的嘴部裁剪。

关键质量门：

- 固定 25fps，不能对整句稀疏抽 16 帧；
- ROI 检出覆盖率至少 95%；
- 字幕必须对应完整 clip，不允许只截视频而不截标签；
- `frames >= labels + adjacent_repeat(labels)`；
- train/dev/test 无同一说话人，跨库名人需额外去重；
- 重复视频、错误字幕、画外音、配音和强遮挡样本应剔除。

## 5. 数据审计与归一化

```bash
python scripts/audit_stage1_data.py \
  data/stage1/train.roi.jsonl data/stage1/dev.roi.jsonl data/stage1/test.roi.jsonl \
  --pixel-samples 5000 --out data/stage1/audit.json
```

脚本发现 speaker leakage 或非法 CTC 长度时以非零状态退出。把报告中的
`roi_normalization.mean/std` 写入三个训练配置，确保统计量来自当前 ROI，而不是沿用 AV-HuBERT
经验值。

## 6. 训练步骤

### 6.1 单卡小样本验证

先复制主配置，设置 `epoch_samples: 10000`、`epochs: 2`、独立 `out_dir`。确认 loss 下降、
显存和吞吐稳定、`metrics.jsonl` 正常，再启动全量训练。

```bash
python -m vallr_pin.cli train --config configs/stage1_multicorpus.yaml \
  --set epoch_samples=10000 epochs=2 out_dir=exp/pilot
```

### 6.2 多卡全量训练

8 卡时默认有效 batch 为 `8 samples/GPU × 8 GPU × accum 4 = 256`：

```bash
torchrun --standalone --nproc_per_node=8 -m vallr_pin.cli train \
  --config configs/stage1_pinyin_only.yaml
```

多节点使用标准 `torchrun --nnodes --node_rank --master_addr --master_port` 参数。每个进程只取自己
的长度桶 batch；所有 rank 的 optimizer collective 次数完全一致。

断点恢复：

```bash
torchrun --standalone --nproc_per_node=8 -m vallr_pin.cli train \
  --config configs/stage1_pinyin_only.yaml \
  --set resume=exp/stage1_pinyin_only/ckpts/last.pt
```

`last.pt` 包含模型、optimizer、GradScaler、epoch、global step 和 RNG；`best.pt` 按
`selection_metric` 保存。启用文字头时另存 `best_cer.pt`，启用拼音头时另存
`best_ser.pt`。

### 6.3 公平消融

```bash
torchrun --standalone --nproc_per_node=8 -m vallr_pin.cli train --config configs/stage1_char_only.yaml
torchrun --standalone --nproc_per_node=8 -m vallr_pin.cli train --config configs/stage1_pinyin_only.yaml
torchrun --standalone --nproc_per_node=8 -m vallr_pin.cli train --config configs/stage1_multicorpus.yaml
```

也可以用统一脚本串行启动三组实验，避免同一组 GPU 显存超订：

```bash
GPUS=8 bash scripts/run_stage1_comparison.sh
python scripts/summarize_stage1_comparison.py
```

至少使用 3 个 seed。若预算有限，先用相同 `epoch_samples` 做固定 step 的三组对比，而不是比较
不同训练时长。

## 7. 验收标准

工程验收：

- 训练 2 epoch 可恢复且恢复后的 global step 连续；
- 无 NaN/Inf、无被静默吞掉的 CTC 样本；
- 长度分桶后的 padding 比随机 batching 明显降低；
- 各来源实际抽样比例接近配置；
- `last.pt`、`best.pt` 以及已启用头对应的 `best_cer.pt`/`best_ser.pt` 均可解码；
- text-only 解码结果中 `pinyin_ser=null`，pinyin-only 结果中 `cer_top1=null`，禁止用未训练头产生随机分数；
- CER 始终与原始文字 token 比较，不允许用编码后的 `<unk>` 替代参考字；同时报告
  `text_oov_rate`，词表 OOV 过高时不应解读文字头 CER；

研究验收：

- Pinyin-only 的 SER 显著低于直接字符模型映射出的拼音错误率；
- 字符辅助消融在至少两个随机种子上优于 Pinyin-only，才能宣称该辅助头有效；
- Stage-II 必须相对“拼音 top-1 直接转写”和“无微调 LLM”基线有稳定提升，且
  over-correction rate 单独报告；
- CMLR/CN-CVS 交叉库测试不能只报告混合 dev 集均值。

在真实训练结果出现前，任何架构优劣都只能称为待验证假设。
