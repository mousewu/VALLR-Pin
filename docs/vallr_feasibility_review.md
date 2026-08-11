# VALLR 方法可行性审查与 VALLR-Pin 修订决策

## 结论

“视觉输入 → 音系单位 → 语言模型恢复文本”的任务分解具有技术可行性，尤其适合普通话：无声调拼音比汉字更接近可见发音信息。但 VALLR 英文开源仓库不足以证明其论文报告的 18.7% WER 可复现，也不应直接移植其 Stage-I 时序结构。

VALLR-Pin 因此采用以下修订：

1. Stage-I 以无声调拼音 CTC 为主目标（90%），字符 CTC 仅作辅助正则和候选来源（10%）。
2. 每个视频帧保留一个时序特征；禁止把 VideoMAE 的空间 patch token 展平后称为“时间”。
3. 使用 CTC prefix beam 同时生成拼音 N-best 和字符 N-best；不再使用两个自回归视觉解码器。
4. Stage-II 接收拼音主假设/后续可扩展的拼音 lattice，以及低权重字符候选；LLM 只承担同音消歧和局部纠错。
5. CTC 目标长于有效视频帧时立即报错，不再用 `zero_infinity=True` 静默吞掉无效样本。

## 对英文项目的代码与权重审查

审查对象：VALLR 论文 `2503.21408v2.pdf`、官方仓库代码及公开 `VALLR.path` Stage-I 权重。

### 可确认的事实

- 权重是 427 个张量、182,434,260 个参数，严格匹配仓库的 Hugging Face VideoMAE V1；并不匹配后来增加的纯 PyTorch V2。
- 输入 16 帧、224×224 RGB 后，VideoMAE 产生 1,568 个 patch token，卷积/池化后只有 8 个 CTC step。
- 开源推理仅包含 Stage-I；论文中决定最终 WER 的 Llama LoRA 权重和完整端到端评测链路未发布。
- 论文表格/正文存在内部不一致，例如 GPT-2 WER 在正文和表中分别出现 23.9% 与 33.9%，LRS3 表注也出现与 18.7% 行值不同的数字。

### 实测

以公开权重严格加载，在本地真实视频上做结构烟测（该视频为普通话，不能用于评估英文准确率）：

| 输入变体 | 输出步数 | 平均最大类概率 | 相对原序列 logits 余弦相似度 |
|---|---:|---:|---:|
| 原 16 帧 | 8 | 0.858 | 1.000 |
| 时间反转 | 8 | 0.882 | 0.990 |
| 单帧复制成静态序列 | 8 | 0.816 | 0.992 |
| 不同时间窗口 | 8 | 0.875–0.944 | 0.986–0.988 |

这不是跨语言准确率结论，但它揭示了结构风险：模型输出对真实时序变化不够敏感。可用 `scripts/evaluate_vallr_reference.py` 在英文 LRS2/LRS3 样本上复现同一审查。

### 开源实现的可复现性缺口

- `load_videos()` 的 `frame_size` 参数未使用，输入没有稳定 resize/crop/normalize。
- 从整句均匀采 16 帧会破坏自然运动速度，并把 CTC 最大输出长度限制到 8 个非重复标签附近。
- `run_inference()` 做 argmax 后没有标准 CTC collapse，blank 和连续重复处理不正确。
- 论文称 adapter 沿时间下采样，但代码实际沿展平的时空 patch-token 轴卷积。
- 训练代码会跳过目标长于 8 的样本，长句因此可能系统性退出训练。
- 缺少 Stage-II 权重、LRS manifest、确定的数据预处理统计量和一键评测脚本。

## 修订后架构

```text
mouth ROI (T,1,88,88)
  -> temporal-preserving 3D stem + 2D ResNet
  -> SANM encoder (T,D)
  -> pinyin CTC (primary, 0.9) -> pinyin prefix beam / lattice
  -> char CTC   (auxiliary, 0.1) -> low-weight text N-best
  -> constrained Mandarin LM -> final characters
```

字符辅助头并非让视觉网络重新承担完整语言建模；它只提供与视觉后验相关的候选。核心验收指标必须分层报告：Stage-I 拼音 SER、字符辅助 CER、Stage-II 最终 CER，以及拼音 oracle/lattice recall。

## 可行性门槛

在宣称方案有效前，应完成：

1. CNVSRC/CMLR speaker-independent 划分，禁止同人跨 train/dev/test。
2. 25 fps 连续 ROI，做 SyncNet/ASD/字幕对齐质量门；禁止跨整句稀疏抽帧。
3. 对比字符 CTC、拼音 CTC、拼音主任务+字符辅助三组 Stage-I。
4. 对比 top-1 拼音、拼音 N-best/lattice，以及有无字符候选的 Stage-II。
5. 分别报告同音替换、删除、插入和中英混说子集；至少三次随机种子。

当前代码完成的是结构修正和可执行性验证，不等价于已经复现论文指标。
