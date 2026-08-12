# VALLR-Pin：基于拼音的普通话视觉语音识别（唇语识别）技术方案与实现

本仓库最初参考 [VALLR-Pin 论文的两个版本](https://arxiv.org/abs/2512.20032)；现已结合英文 VALLR（ICCV 2025）论文、
官方代码和公开 Stage-I 权重完成独立审查与架构修订。完整证据和复现实验见
[`docs/vallr_feasibility_review.md`](docs/vallr_feasibility_review.md)。

> 公开仓库只包含代码、配置、测试与文档，不包含训练数据、公开视频、字幕、派生张量、
> 模型权重、运行数据库或论文 PDF。对应资源需要由使用者按各自许可证自行准备。

> Chang Sun, Dongliang Xie, Wanpeng Xie, Bo Qin, Hong Yang.
> *VALLR-Pin: Uncertainty-Factorized Visual Speech Recognition for Mandarin with Pinyin Guidance*（v2；
> v1 题为 *Dual-Decoding VSR for Mandarin with Pinyin-Guided LLM Refinement*）

两篇原草稿的方法主体一致（双解码 + 拼音中介 + LLM 纠错），差异在**叙事框架**与**实验**：
v1 把卖点放在"多任务学习 + LLM 后处理"，v2 改用**不确定性分解**（uncertainty
factorization）来论证——普通话 VSR 直接预测字符是病态问题，因为视觉证据只能约束到
音节层；v2 另外补了 CMLR 数据集实验、把 baseline 换成"Paraformer 化的 CNVSRC 2025 baseline"，
并明确 LLM 是**约束式的补充**而非独立识别器。审查后保留了“拼音中介”，但废弃了
双自回归视觉解码器。当前默认主链路完全解耦：Stage-I 是**拼音-only CTC**，Stage-II
用独立中文纯文本训练“带噪拼音→原文”；字符 CTC 只保留为可选辅助头和消融实验。

> 兼容性提示：旧双 AR 检查点属于 `architecture_version=1`，不会被静默部分加载；
> 需要用当前配置重新训练。这样可以避免看似成功、实际参数错配的实验。

---

## 1. 问题与设计出发点

普通话唇语识别的核心困难不是"看不清"，而是**信息瓶颈的位置错了**：

- 视觉上，唇形能相对可靠地区分**声母/韵母**构成的音节（唇形是发音器官的直接投影）；
- 但汉字与音节是**多对一**的：一个 `shi` 对应上百个字，`shi jian` 既是"时间"也是"事件"；
- 端到端 `P(Y|X)` 让视觉编码器同时承担"辨音"和"辨义"两件事，后者本质上不是视觉任务。

于是把推理显式分解成两层（论文 Eq.5）：

```
P(Y|X) = Σ_P P(Y | P) · P(P | X)
         └── 语言层 (LLM) ──┘  └ 视觉层 (VSR) ┘

X --F_VSR--> P̂ --F_LLM--> Y
```

- **视觉层**只需要解决"这段唇动是什么音"——这是它擅长且可学的；
- **语言层**在拼音约束下从同音候选中选字——这是 LLM 擅长且几乎零成本的。

关键取舍：**拼音去掉声调**。声调来自基频，唇部几乎不可见，强行建模等于往标签里注入
不可约噪声；去掉声调后建模单元从 ~1300 降到 ~400（论文报告 397，本实现的全量表为 410，
差异只是语料覆盖），既压缩了搜索空间，又保留了唇形真正敏感的"声母+韵母"结构。

---

## 2. 总体架构

```
唇部 ROI 视频 X ─► 3D stem + ResNet + SANM ─► 拼音 CTC ─► Pinyin N-best
                                                       │
独立中文纯文本 ─► 无声调拼音 ─► 在线替换/删除/插入/mask ─► LLM LoRA
                                                       │
推理：Pinyin top-1/N-best ────────────────────────────► 最终文字 Y

可选消融：共享视觉编码器 ─► 低权重字符 CTC（不作为 Stage-II 必需输入）
```

四个步骤与论文 Fig.1 一一对应：

| 步骤 | 内容 | 本仓库入口 |
|---|---|---|
| Step 1 | 视频→无声调拼音 CTC | `cli train` |
| Step 2 | 独立纯文本→干净文字/拼音语料 | `cli build-stage2-text` |
| Step 3 | 在线拼音加噪→LLM LoRA | `cli sft` |
| Step 4 | 推理：Stage-I 解码 → Stage-II 精化 | `cli decode` + `cli refine` |

---

## 3. Stage-I：拼音/文字 CTC 可对比训练

### 3.1 视觉前端与编码器

- **前端**：`Conv3d(1→64, k=(5,7,7), s=(1,2,2))` + `MaxPool3d` + 2D ResNet-18/50 主干，
  逐帧池化到 `d_model`。**时间维不下采样**：普通话语速约 4–6 字/秒，25fps 下每字仅 4–6 帧，
  再降采样会破坏 CTC 的 `T' ≥ L` 约束（`vallr_pin/models/frontend.py`）。
- **编码器**：12 层 SANM（`vallr_pin/models/sanm.py`）。SANM = 标准自注意力 **并联**
  一个 FSMN memory block（对 V 做 kernel=11 的深度可分离一维卷积 + 残差）。
  唇动的判别信息高度局部（协同发音只影响相邻几帧），这个局部记忆分支正是对症的归纳偏置。

### 3.2 主任务、辅助任务与损失

| 分支 | 结构 | 建模单元 | 作用 |
|---|---|---|---|
| 拼音主头 | 线性层 + CTC prefix beam | 无声调音节（约 410） | 学习视觉可辨的音节并产出 N-best |
| 文字头 | 线性层 + CTC prefix beam | 汉字/整体英文 token | 可独立训练，或作为拼音头的辅助监督 |

默认主方案损失为：

```
L = L_ctc^pinyin
```

三种模式由显式权重控制，总损失会按权重和归一化：

```text
Pinyin-only: text_ctc_weight=0.0, pinyin_ctc_weight=1.0
Text-only:   text_ctc_weight=1.0, pinyin_ctc_weight=0.0
Joint:       text_ctc_weight=0.1, pinyin_ctc_weight=0.9
```

任何目标长度超过有效视频帧数的样本都会立即报错，避免 CTC 用零损失掩盖错误切段。
`alpha` 仅为旧配置兼容字段，新实验应使用上述两个权重。

### 3.3 N-best 与拼音假设

- 两个头都使用标准 **CTC prefix beam**。解码器根据检查点中的权重自动启用已训练的头：
  text-only 不会运行随机拼音头，pinyin-only 也不会运行随机文字头。

  ```
  score(Y) = log P_ctc(Y | X) / |Y|^lp
  ```

  `--pinyin-mode ctc` 会输出拼音 beam；旧的 `ar` 参数只作为兼容别名，模型内部不再创建
  自回归视觉解码器。

---

## 4. Stage-II：解耦的带噪拼音→文字 LLM

### 4.1 独立纯文本数据

Stage-II 不解码 Stage-I 训练集，而是从字幕、访谈、口语转写和通用中文文本中构建：

```
原文：我的手机放在桌子上
干净拼音：wo de shou ji fang zai zhuo zi shang
在线噪声：wo de shou qi fang zai zhuo <mask> shang
监督答案：我的手机放在桌子上
```

`build-stage2-text` 负责清洗、整句多音字 G2P、去重、按文档切分以及排除 VSR dev/test
中的完全相同句子。生成的 JSONL 只保存干净 `text+pinyin`，不依赖视频或 Stage-I 检查点。

### 4.2 在线拼音噪声

LoRA 数据集在每个 epoch 动态产生替换、删除、插入、相邻交换和 `<mask>`。默认 25% 保持
干净、50% 轻噪声、25% 重噪声，既模拟 Pinyin CTC 错误，也让模型学会不要过度纠正。
`variants_per_text` 提供虚拟扩增，不需要把多个副本写到磁盘。

### 4.3 LoRA 与可选真实错误校准

`W = W₀ + AB`，默认 `r=8, α=32, lr=1e-4`，只对 assistant 答案计算损失。原生训练器支持
`torchrun` 单机多卡 DDP；每张卡独立加载一个基座模型副本。若使用 ms-swift，先用
`materialize-stage2` 把在线噪声固化为 messages JSONL。

原来的 `decode-ckpts + build-llm-data` 仍保留，但只定位为可选的小规模真实错误校准，
不再决定 Stage-II 主训练数据量。

### 4.4 无 LLM 基线：拼音受限的 n-gram 重打分

`vallr_pin/llm/refine.py::NgramPinyinRefiner`。用训练文本上的字符 bigram，在
"预测拼音允许的同音字集合"上做 Viterbi（本质就是经典拼音输入法解码）。有字符 N-best
时可附加候选提示，没有时也能独立运行：

```
score(Y) = LM(Y)/|Y| − β · SER(pinyin(Y), P̂)          β = 6
```

重排。它有三个用处：① 离线/无 GPU 时整条链路仍可复现；② 作为"LLM 到底带来了多少增益"的
诚实下界；③ 当 LLM 输出被护栏拒绝时的兜底。

---

## 5. 代码结构

```
vallr_pin/
├── text/
│   ├── pinyin.py          无声调音节表(410)、整句多音字消歧、同音字表
│   └── tokenizer.py       字符/拼音双词表 (<blank>/<sos|eos>/<unk> 布局)
├── data/
│   ├── dataset.py         manifest 数据集、ROI 变换、padding collate
│   └── synthetic.py       受控合成数据（视觉只编码音节 → 复刻同音歧义）
├── models/
│   ├── frontend.py        3D stem + ResNet-18/34/50
│   ├── sanm.py            FSMN memory / SANM 自注意力 / 编解码层
│   ├── decoders.py        SANM 编码器（旧 AR 类仅供历史代码参考）
│   └── vallr_pin.py       拼音 CTC 主头、可选字符 CTC、prefix beam、长度校验
├── engine/
│   ├── trainer.py         Stage-I DDP、Noam 调度、精确恢复
│   ├── tracking.py        可选 SwanLab rank-0 指标记录
│   ├── decode.py          拼音 N-best；字符头按配置可选
│   └── metrics.py         CER / 音节错误率（S+D+I）/N
├── llm/
│   ├── text_data.py       独立中文纯文本清洗、G2P、按文档切分
│   ├── noise.py           在线拼音替换/删除/插入/mask/交换
│   ├── prompt.py          无候选主提示词 + 可选校准提示词
│   ├── build_data.py      可选真实 VSR 错误校准数据
│   ├── lora_sft.py        LoRA SFT（DDP + SwanLab + ms-swift 命令）
│   └── refine.py          LLM 精化器、n-gram 受限重打分器、批量评测入口
└── cli.py                 train / decode / build-stage2-text / sft / refine / pipeline
configs/  cnvsrc_base.yaml（论文规格）· smoke.yaml（小规模）· llm_sft.yaml
scripts/  prepare_manifest.py（CNVSRC/CMLR→manifest+ROI）· run_full_pipeline.sh
tests/    test_pipeline_units.py
```

---

## 6. 快速开始

```bash
pip install -r requirements.txt
python tests/test_pipeline_units.py          # 单元测试，不需要 GPU/数据
```

### 6.0 从 CN-CVS / CMLR 零开始训练 Stage-I

生产训练入口、数据角色、ROI 质量门、DDP/恢复训练和三组公平消融见
[`docs/stage1_training_plan.md`](docs/stage1_training_plan.md)。最短路径：

```bash
cp configs/corpora.example.yaml configs/corpora.local.yaml
python scripts/build_stage1_manifests.py configs/corpora.local.yaml
python scripts/audit_stage1_data.py data/stage1/{train,dev,test}.jsonl --out data/stage1/audit.json
torchrun --standalone --nproc_per_node=8 -m vallr_pin.cli train \
  --config configs/stage1_pinyin_only.yaml
```

一次顺序训练文字-only、拼音-only 和联合双头，并汇总最优 dev CER/SER：

```bash
GPUS=8 bash scripts/run_stage1_comparison.sh
```

每个实验的 `ckpts/` 会保存 `best.pt`（按 `selection_metric`），联合模式还会分别保存
`best_cer.pt` 和 `best_ser.pt`，避免两个头在不同 epoch 达到最优时无法公平对比。

如果 `CNC-AV` 指的是 `CN-Celeb-AV`，它在没有句级转写时不能作为监督 Stage-I 数据；
默认 manifest 配置会关闭该 pseudo source。

### 6.1 构建并多卡训练 Stage-II

```bash
cp configs/stage2_text.example.yaml configs/stage2_text.local.yaml
python -m vallr_pin.cli build-stage2-text --config configs/stage2_text.local.yaml

torchrun --standalone --nproc_per_node=8 -m vallr_pin.cli sft \
  --config configs/llm_sft.yaml
```

启用 SwanLab 时，在对应 YAML 中设置 `swanlab.enabled: true` 和 `mode: online`。DDP 默认只由
rank 0 记录已经跨卡聚合的 epoch 指标，避免同一实验重复八份日志。
首次启用前执行 `pip install -r requirements-swanlab.txt` 和 `swanlab login`。

### 6.2 合成数据端到端演示（无需真实数据）

```bash
python -m vallr_pin.cli pipeline --work-dir exp/demo --epochs 60 --device cpu
```

合成数据不是随手造的噪声：**每个无声调音节固定一张 viseme 模板，视频帧只由音节决定**，
因此"视觉只能确定到音节、字符层一对多"这一病态性被精确复刻；语料里塞满同音异形词
（手机/收集、事件/时间、权利/权力、报到/报道……），Stage-I 必然在这些位置犯同音错误，
而拼音分支不会——正好用来检验 Stage-II 是否真的在起作用。

### 6.3 从网络视频自建数据集

**最省事的路径**：找有**人工字幕的单人口播**素材，这类视频不需要 ASR、不需要
对齐、不需要 ASD，字幕直接就是句级标签。批量筛选 + 构建一条命令：

```bash
python scripts/batch_harvest.py urls.txt --model face_landmarker.task \
    --out-dir harvest --build
```

筛选按**成本递增**分四关，不过就立刻停手：

| 关卡 | 成本 | 判据 |
|---|---|---|
| 元数据 | 0 字节 | 时长/帧率/画质/**是否有中文人工字幕** |
| 字幕 | ~100 KB | 语速分布（正常普通话 4–6 字/秒）、条数、重叠率 |
| 探针片段 | ~4 MB | 抽 60s 看轨迹数/主轨占比/人脸覆盖/唇部像素/正脸程度 |
| 完整构建 | 全片 | 前三关全过才下整片，切 ROI 出 manifest |

判断"是不是单人独白"要看**主轨占比**而不是轨迹条数——双人访谈的反打镜头次轨
可能只占 2–6%，单看条数会漏判（实测教训，见 §8.3）。

如果素材没有人工字幕、或是多人对话，再走下面这条完整路径：

```bash
# ① 素材体检：这段视频值不值得采
python scripts/probe_video.py clip.mp4 --model face_landmarker.task --out-dir probe

# ② Active Speaker Detection：逐帧判定"画面上的人是不是在说话"
python scripts/asd_pipeline.py clip.mp4 --model face_landmarker.task \
    --out-dir asd --speakers 2 --names 主持人 嘉宾 --dump-faces

# ③ 文字稿 -> 带时间戳的句段（锚点对齐，自动拒掉编辑改写的部分）
python scripts/align_transcript.py transcript.txt --asr funasr --audio clip.wav \
    --out aligned.jsonl --min-match 0.8

# ④ 语料可用率体检（含中英混说的救回率）
python scripts/transcript_stats.py transcript.txt --loanwords
```

三个设计要点：

**ASD 走身份对齐而非 SyncNet。** 人脸 embedding 聚类 + 声纹/基频 diarization，
两侧的簇用**共现矩阵自动配对**——正常剪辑下"画面上的人就是说话人"占多数时间，
所以共现矩阵的最优匹配就是身份对应，人工只需给簇起个名字。反打镜头（听者出镜、
说话人画外）会被判 `is_speaker=False`，这些帧是"闭着嘴的画面 + 别人的文本"，
必须丢掉。自带的 F0 diarizer 零依赖但**只在说话人音高可分时有效**（典型是一男一女），
同性别对话请换 `--diarizer rttm` 接 pyannote。

**文字稿要锚点对齐，不能直接强制对齐。** 出版的访谈稿是编辑稿：口水词被删、
句子被重组、夹着编者注。做法是先 ASR 拿带时间戳的字序列，再与文字稿找唯一
k-gram 锚点、取单调子序列、块内 DP，最后按句统计匹配率，低于阈值直接丢。
**宁可少要，不可要错**——一条时间戳错位的样本比缺这条有害得多。

**中英混说必须处理。** 中文科技口语的英文混入率极高（实测某 AI 访谈稿 34% 的
句子含英文），而拼音中介方案对英文词无能为力。`vallr_pin/text/loanword.py` 按
中国人的实际读法把英文转成普通话音节（`token`→`tou ken`，缩写按字母音
`RL`→`a er ai er`）。一个英文词在**字符侧算 1 个 token、音节侧展开成多个音节**——
本仓库两个 CTC 头的目标序列彼此独立，不要求等长，所以天然支持。

### 6.3 公开数据集（CNVSRC / CMLR）

```bash
# 1) 在 corpora.local.yaml 中按实际发行包填写 CN-CVS、CMLR，并可启用
#    CNVSRC.Dev、CN-CVS2-P1、CN-CVS3 等其他有句级文本的来源。
cp configs/corpora.example.yaml configs/corpora.local.yaml
python scripts/build_stage1_manifests.py configs/corpora.local.yaml
python scripts/audit_stage1_data.py data/stage1/{train,dev,test}.jsonl \
  --out data/stage1/audit.json

# 2) 视频和纯文本两阶段解耦训练
GPUS=8 bash scripts/run_full_pipeline.sh exp/vallr_pin_cnvsrc \
  configs/stage1_pinyin_only.yaml configs/stage2_text.local.yaml
```

manifest 每行除 `id/video/text` 外还应带 `speaker_id/source/split/n_frames`；
训练器不硬编码数据集名称，`source_weights: {}` 会按 manifest 中的自然比例使用全部来源。

---

### 6.4 规模化控制面、多节点 Worker 与 Web 控制台

采集、通用分析和模型适配是三个完全独立的任务队列。提交视频链接时不选择模型：

```text
Web/API 控制节点（唯一持有 SQLite）
        │ HTTP worker API + token
        ├──────── download：metadata / source / subtitles
        ├──────── analyze：landmarks / source references 等通用标注
        └──────── render：按模型规格生成 ROI / WebDataset shards

共享 data/ 存储：raw/ → tracks/ → derived/
```

**不要让多台机器通过 NFS 直接打开同一个 SQLite WAL。** SQLite 只放控制节点；远程
worker 使用 `--api`。不同节点将同一共享存储挂载为各自的 `--data-root`，任务只传
`video_id`，因此各节点的挂载绝对路径可以不同。

```bash
# 控制节点
export PIPELINE_TOKEN='replace-with-a-long-random-token'
python scripts/pipeline_web.py --host 0.0.0.0 --port 8080 \
  --db data/pipeline.sqlite --worker-token "$PIPELINE_TOKEN"

# 下载节点（需要 yt-dlp / ffmpeg）
python scripts/pipeline_worker.py --role download \
  --api http://CONTROL_NODE:8080 --token "$PIPELINE_TOKEN" \
  --data-root /shared/vallr-data --node-id downloader-01

# 通用分析节点（需要 opencv / mediapipe）
python scripts/pipeline_worker.py --role analyze \
  --api http://CONTROL_NODE:8080 --token "$PIPELINE_TOKEN" \
  --data-root /shared/vallr-data --node-id analyzer-01

# 模型适配节点（需要 opencv / pypinyin）
python scripts/pipeline_worker.py --role render \
  --api http://CONTROL_NODE:8080 --token "$PIPELINE_TOKEN" \
  --data-root /shared/vallr-data --node-id renderer-01
```

浏览器打开 `http://CONTROL_NODE:8080`，可按行批量提交 URL、查看三类任务与节点状态，并执行
暂停、继续、取消、失败重试。状态机使用租约和心跳；节点掉线后任务自动重新排队，超过
最大尝试次数才永久失败。worker token 通过 `X-Pipeline-Token` 校验。

命令行也支持批量采集；`urls.txt` 每行一个链接。下载成功后只自动创建模型无关的
`analyze`，不会提前生成某个模型的 Tensor：

```bash
python scripts/pipeline_submit.py --api http://CONTROL_NODE:8080 \
  --token "$PIPELINE_TOKEN" --file urls.txt

# analyze 完成后，同一批 video_id 可一次创建多种模型格式
python scripts/pipeline_render.py Fy6tKSHGEXQ VIDEO_ID_2 \
  --api http://CONTROL_NODE:8080 --token "$PIPELINE_TOKEN" \
  --spec vallr_pin --spec avhubert
```

目录对应三层边界：`raw/<video_id>/` 是不可变采集层，`tracks/<video_id>.npz/.json`
是可复用标准分析层，`derived/<spec>/<video_id>/` 才是模型相关适配层。新增模型只需增加
render spec，不必重新下载或分析原视频。后续的 diarization、文字稿对齐和 ASD 也应作为
`analyze` 的模型无关产物写入标准层，而不是放进某个 render 实现。

流式处理不再缓存整段视频 ROI，只保留当前字幕句；输出为 WebDataset tar shard：

```bash
python scripts/stream_build.py source.mp4 subtitles.json3 tracks.npz \
  --out-dir data/derived/vallr_pin/VIDEO_ID --source-id VIDEO_ID \
  --speaker-id SPEAKER_ID --spec vallr_pin --shard-samples 1000
```

每条样本为 `<key>.npy + <key>.json`。manifest 使用
`wds://shards/shard-000000.tar::<key>.npy`，`LipReadingDataset` 可直接读取。tar 与
manifest 先写 `.partial`，成功后原子改名，中断不会发布半成品。

统一去重、说话人聚类与 speaker-independent 划分：

```bash
python scripts/build_dataset_catalog.py data/derived/*/*/manifest.jsonl \
  --out-dir data/manifests --dev-percent 5 --test-percent 5
```

去重键是“规范化文本 + 视觉哈希”，不会误删不同说话人说相同文本；显式 `speaker_id`
优先，缺失时使用 ROI 外观描述子聚类；最终按说话人簇稳定哈希分 train/dev/test，保证同一
说话人不跨集合。轻量描述子适合固定机位口播初筛；复杂跨域数据应替换为 ArcFace embedding。

## 7. 实验设置（对齐论文）

| 项 | 设置 |
|---|---|
| 训练数据 | CNVSRC 2025 固定赛道 S3 = CN-CVS + CNVSRC.Dev + CN-CVS2-P1 + CN-CVS3 |
| 评测集 | CNVSRC-Multi.Dev（43 说话人：23 录音棚 + 20 网络视频）、CMLR、自采集 |
| 前端/编码器 | 3D Conv + ResNet-50，12 层 SANM，d=512 |
| 主损失 | 拼音 CTC；字符-only/辅助字符 CTC 作为消融 |
| 解码 | 拼音 CTC prefix beam=10，nbest=5 |
| LLM 数据 | 与视频解耦的中文纯文本，在线合成 Pinyin CTC 风格噪声 |
| LLM | Qwen3-4B-Instruct-2507 + LoRA(r=8, α=32, lr=1e-4)，原生 DDP 或 ms-swift |
| 指标 | CER = (S+D+I)/N |

论文报告的参照值（CER%，越低越好）：

| 方法 | CNVSRC-Multi.Dev | 自采集 |
|---|---|---|
| CNVSRC 2025 baseline | 31.91 | 40.21 |
| Char-only baseline（Paraformer 化） | 30.87 | 38.60 |
| + Pinyin decoder（消融） | — | 37.23 |
| + 零样本 LLM（消融） | — | 37.86 ↑ |
| **VALLR-Pin（全量）** | **28.39** | **35.49** |
| VALLR-Pin（open 赛道） | 24.10 | 32.22 |
| CMLR：Ma's model → +本方案 | 9.10 → **7.89** | — |

> 注：本仓库不包含 CNVSRC/CMLR 数据（需向主办方申请），因此**未复现上表数字**；
> 上表是论文声称的结果，作为对齐目标列出。

---

## 8. 合成数据上的历史验证结果（v1，已退役）

> 本节数字来自旧双自回归架构，只保留为工程演进记录，不能作为当前
> `architecture_version=2` 的性能结论。当前架构已通过一轮端到端烟测；正式数字必须在
> speaker-independent 的 CNVSRC/CMLR 划分上重跑。

在受控合成集上跑 `cli pipeline`（50 句 × 4 说话人训练，1 个未见说话人测试，
d_model=128 / 4 层 SANM / ResNet-18(width=16)，MacBook CPU 训练 60 epoch，约 10 分钟）。
**这是实现正确性的验证，不是性能指标**——训练与测试共用同一套 50 个句子（只有说话人参数、
噪声、语速抖动不同），模型可以记住语料，所以下面的绝对数值没有泛化意义。

**Stage-I 收敛过程**（dev，CTC greedy）：

| epoch | 1 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| 字符 CER % | 100.0 | 94.1 | 77.1 | 30.0 | 12.2 | 8.7 | **5.7** |
| 拼音 SER % | 96.6 | 94.9 | 71.4 | 32.5 | 13.8 | 9.7 | **6.1** |

旧版换成 **beam=6 + CTC 联合打分**后：dev 字符 CER **0.00%**，拼音 SER **1.01%**
（联合打分把 CTC greedy 的 5.7% 压到 0，主要修掉的是漏字/复读）。

**Stage-II 诊断**（`scripts/homophone_stress_test.py`，把参考文本按 30% 比例随机换成同音字，
隔离评估"音对字错"这一主导错误的修复能力）：

| 语言模型 | 注入后 CER | 精化后 CER | 修复率 |
|---|---|---|---|
| bigram LM（与测试同域，50 句） | 16.63% | **1.62%** | **90.2%** |
| bigram LM（40 句训练 / 10 句留出） | 15.79% | 20.00% | −26.7%（变差） |

第一行说明**拼音约束确实能把同音错误改回来**；第二行同样重要：一个只见过 40 个句子的
n-gram 语言模型**没有世界知识**，在没见过的句子上会越改越错——这说明 Stage-II
需要有语言知识的 LLM，并需通过大规模带噪拼音→原文微调学会遵守拼音约束；真实 VSR
error-aware 样本只是可选校准，不是数据主体。

**过度纠正的真实代价**：当 Stage-I 已经全对（CER 0.00%）时，n-gram 精化器仍会改坏
50 句中的 1 句（CER 0.00% → 0.20%）。改坏的原因不是语言模型，而是**拼音预测本身的错误
被传播**（`P̂` 把 "…事实" 解成 "shi shi shi"，精化器据此补出 "事实施"）。围绕这个现象，
本实现补了三道防线，效果按顺序累积：

| 措施 | 该句 dev CER 退化 |
|---|---|
| 无防护 | 2.43% |
| + Viterbi 长度护栏 + top-1 先验 | 1.42% |
| + 拼音项改为相对 SER（以 N-best 最小 SER 为基线） | 0.81% |
| + 拼音贪心解码的连续重复阻断（`no_repeat_run=3`） | **0.20%** |

同时拼音 SER 从 2.03% 降到 1.01%——自回归解码在 "shi shi shi" 这类重复音节上的塌缩
是普通话拼音解码的一个具体坑，值得单独处理。

**Error-aware 数据构造**：3 个检查点 × 60 句解码后，过滤器丢掉 120 条 CER>0.8 的样本
（早期检查点的胡言乱语）与 52 条已经全对的样本，保留 8 条——机制符合预期。
真实数据上这个比例会完全不同（Stage-I CER 30% 左右时绝大多数样本都会被保留）。

复现：

```bash
python -m vallr_pin.cli pipeline --work-dir exp/demo --epochs 60 --device cpu
python scripts/homophone_stress_test.py --texts exp/demo/data/train.jsonl --rate 0.3
python scripts/analyze_refine.py exp/demo/dev_refined.jsonl
```

---

### 8.2 真实素材上的数据管线验证

在一段 60 秒的中文播客片段（720p / 25fps / 一男一女访谈 / 无字幕轨）上跑完整管线：

| 环节 | 结果 |
|---|---|
| 人脸检出率 | 100%（1500/1500 帧），脸宽 179px，唇部 ROI 约 62px |
| 人脸聚类 | 2 簇自动分开，切分区间与镜头切换点完全吻合 |
| 簇配对 | 共现矩阵 `[[31,22],[459,19]]`，最优匹配自动得到脸↔声对应，无需人工 |
| ASD 可用率 | 声道平滑前 64.5% / 平滑后 **83.9%**（碎片数 56 → 21 段） |
| 抓到的反打镜头 | `[55.04s–58.24s]` 主持人出镜但嘉宾在说话，正确判为丢弃 |
| 中英混说救回 | 语料可用率 54.3% → **80.9%** |
| 处理速度 | 约 5× 实时（M4 CPU） |

两个必须说明的负面结论：

* **音画同步的手工粗测方法失效**。用"张嘴幅度 × 音频包络"在音节带做互相关，
  峰值相关系数只有 0.11、次峰 0.09，换个特征峰值就从 −1 帧漂到 −4 帧。
  播客音频普遍过压缩，包络的音节结构被压平。`probe_video.py` 因此在
  置信度不足时直接报 `reliable: false`，**不要拿这个数去做偏移校正**。
  这个问题后来用 SyncNet 解决了，见 §8.4。
* **自带的 F0 diarizer 只适用于音高可分的说话人**。本例是一男一女所以干净，
  同性别对话必然失败，请接 pyannote 的 RTTM。

### 8.3 单人口播素材上的完整验证

同样的管线在一段**有人工字幕的单人口播**片段（60s / 720p / 30fps）上跑通，
产出了第一批真实训练样本：

| 指标 | 播客访谈 | 单人口播 |
|---|---|---|
| 稳定人脸轨迹 | 5 条（正反打） | **1 条** |
| 主轨占比 | 0.83 | **1.00** |
| 平均偏航 \|yaw\| | 0.162 | **0.015** |
| 唇部 ROI 宽度 | 62px | **75px** |
| 标签来源 | 需 ASR + 锚点对齐 | **人工字幕直接可用** |
| 最终 yield | 估算 10–15% | **实测 87.4%** |

字幕时间戳的可用性是**实测确认**的，不是假设：语速中位 5.36 字/秒
（p10–p90 = 4.48–6.29，98.7% 落在 3–7 区间），相邻间隙中位 66ms、零重叠；
产出样本的每字帧数中位 5.86，与 30fps ÷ 5.36 字/秒 = 5.6 的理论值吻合。
另用画面里的烧录字幕做独立校验——字幕消失与出现的两个画面变化点相隔 6 帧 = 0.2s，
精确等于字幕文件里两条之间的间隙，说明渲染时刻与时间戳逐帧一致。

**结论：有人工字幕的单人口播是自采数据的正确目标，多人播客不是。**
筛选可以完全自动化（`batch_harvest.py`），实测能正确放行口播、
在第 1 关（0 字节成本）就拒掉无字幕的播客。

### 8.4 SyncNet 音画偏移标定

手工特征测不出偏移，换成 SyncNet（Chung & Zisserman, ACCV 2016）后问题解决。
脚本 [scripts/syncnet_offset.py](scripts/syncnet_offset.py)，需官方权重
`syncnet_v2.model`（54 MB）与仓库里的 `SyncNetModel.py`。

**输入规格必须严格对齐训练分布**，否则结果无意义：25fps 视频（脚本自动重采样，
30fps 素材直接喂会因时间尺度错配而失效）、16kHz 音频、人脸按 `2.8·bs` 方形裁剪
且垂直中心下移 `0.4·bs`（对准嘴部）、224×224、BGR 不归一化、每 5 帧配 20 帧 MFCC。

实测结果：

| 素材 | 偏移 | 置信度 | 最小距离 / 基线 |
|---|---|---|---|
| 单人口播 | **0 帧** | 5.83 | 8.80 / 14.63 |
| 单人口播 + 注入 4 帧延迟 | **−4 帧** | 5.77 | 8.80 / 14.57 |
| 双人播客 | **−3 帧 (−120ms)** | 4.24 | 9.54 / 13.78 |

第二行是**自检**：注入已知的 4 帧音频延迟，测出恰好 −4 帧且置信度不变，
证明整条测量链路与符号约定都正确（负值 = 音频滞后视频）。没有这一步，
"口播偏移为 0"这个结果无法与"链路根本没生效"区分开。

第三行是真正的收获：播客素材有 3 帧偏移，而 25fps 下普通话每字仅 4–6 帧，
**这接近半个字，会系统性污染每一条标签**——手工方法完全测不出来。
测得的值用 `build_from_subtitles.py --av-offset-frames` 直接补偿。

## 9. 本实现相对论文的补充

论文只有 5 页，很多工程细节未写明；以下是本实现补齐或增强的部分，均已在代码中注明：

1. **两阶段训练解耦**：Stage-II 主数据来自独立中文纯文本，数据量不受视频规模限制。
2. **在线 Pinyin CTC 风格噪声**：每个 epoch 动态生成替换、删除、插入、mask 和交换；
   25% 干净样本用于抑制 over-correction。
3. **可选真实错误校准**：多检查点解码和 CER 分桶保留为后续校准工具，不再是主训练入口。
4. **长度/格式护栏**：输出异常时，有字符 top-1 则回退；纯拼音模式返回空并计为错误。
5. **无 LLM 的受限重打分基线**：给出"拼音约束到底值多少"的下界，也让整条链路可离线复现。
6. **多音字整句消歧**：拼音标签用整句 pypinyin 转换（词组消歧），逐字转换会让"银行/行走"
   这类标签错误，直接污染拼音分支的监督信号。
7. **单机多卡与 SwanLab**：Stage-I、Stage-II 都支持 `torchrun`，rank 0 记录聚合指标。
8. **严格 CTC 长度校验**：标签长于有效帧时立即拒绝样本，不允许 `zero_infinity`
   把错误切段伪装成零损失。
9. **按文档切分和污染排除**：同一文档不跨 Stage-II train/val/test，并可排除 VSR dev/test 原句。

## 10. 已知局限

- 未在 CNVSRC/CMLR 上训练，**没有复现任何论文数字**；合成数据只验证链路正确性。
- 英文 VALLR 的公开 Stage-I 权重可加载，但 Stage-II 权重及完整评测资产未公开；其
  18.7% WER 在本仓库中视为待独立复现的论文声明，而非已验证基线。
- ROI 提取脚本默认是粗裁；要对齐 CNVSRC baseline 需接入 RetinaFace + 68 点关键点。
- LLM 精化是逐句独立的，跨句上下文（对话/篇章）未利用。
- 拼音是**无声调**的，"实施/时事/事实"这类**完全同音**词仍只能靠语言模型判别——
  这是方案的固有上界；引入声调需要有声调标注且要面对视觉不可辨问题（论文亦未涉及）。
