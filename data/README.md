# data/ 目录说明

本目录是数据管线的本地产出。公开仓库只保留本说明文件，下面列出的媒体、字幕、
manifest、张量、报告和运行数据库全部由 `.gitignore` 排除，不随代码仓库分发。

```
data/
├── source_catalog.jsonl            不含媒体的来源索引，可进版本控制
├── raw/<video_id>/                 不可变原始媒体（长期保存，不进版本控制）
│   ├── source.mp4                  原片；片段会在 source.json 明确标 partial
│   ├── subtitles*.json3            原始字幕，不改写
│   ├── source.json                 URL、SHA-256、ffprobe、完整/片段标记
│   └── tracks.npz                  可选关键点缓存
├── tracks/                         跨模型共享的轻量关键点缓存
├── derived/<spec>/<video_id>/      按模型规格渲染的可再生输入
├── mono_demo/                     单人口播素材产出的示例数据集
│   ├── manifest.jsonl             25 条样本的文本 + 拼音标签（唯一需要进版本控制的）
│   ├── build_report.json          构建统计：yield、各环节丢弃计数、字幕自查结果
│   └── roi/*.npy                  每条样本的唇部序列，(T, 96, 96) uint8 灰度
├── clips/                         旧版兼容目录；新数据统一进 raw/
└── reports/                       各环节的体检与标定报告
```

## manifest 格式

每行一条样本，可直接喂给 `python -m vallr_pin.cli train`：

```json
{"id": "utt_00129", "video": "roi/utt_00129.npy",
 "text": "舍弃一些东西可以换来更多其他的东西",
 "pinyin": "she qi yi xie dong xi ke yi huan lai geng duo qi ta de dong xi",
 "start": 299.77, "end": 302.37, "n_frames": 74, "frames_per_unit": 4.35}
```

`video` 是相对 `data/mono_demo/` 的路径；训练时用 `data_root` 指向该目录。
`pinyin` 字段仅供人工检查，训练时由 `vallr_pin.text` 从 `text` 现场生成，
两者应当一致。

## reports/ 里的关键结论

| 文件 | 结论 |
|---|---|
| `syncnet_mono.json` | 口播素材音画偏移 **0 帧**，置信度 5.83 |
| `syncnet_mono_inject4.json` | 注入 4 帧延迟测出 **−4 帧** —— 测量链路的自检 |
| `syncnet_podcast.json` | 播客素材偏移 **−3 帧 (−120ms)**，需补偿后才能用 |
| `asd_podcast.json` | 播客的逐段说话人判定，含被丢弃的反打镜头 |
| `probe_mono.json` | 单轨、100% 人脸覆盖、\|yaw\|=0.015、唇宽 75px |
| `probe_podcast.json` | 5 条轨迹、主轨仅占 0.83 —— 多人素材的典型特征 |
| `sample_roi_strip.png` | 一条真实样本的 ROI 序列，用于肉眼抽检 |

## 重建方式

`clips/` 与 `roi/` 被 gitignore，需要时按下面重建：

```bash
# 0) 登记已有原片；默认复制且拒绝覆盖哈希不同的文件
python scripts/register_source.py video.mp4 --video-id VIDEO_ID \
    --url "<video-url>" --subtitles subs.json3 --out-root data/raw

# 1) 若本地没有原片，重新下载后先登记
yt-dlp -f "136+bestaudio[ext=m4a]" --download-sections "*300-360" \
    --force-keyframes-at-cuts --merge-output-format mp4 \
    -o data/clips/mono.mp4 "<video-url>"

# 2) 重建数据集
python scripts/build_from_subtitles.py \
    data/clips/mono.mp4 data/clips/mono.zh-Hans.json3 \
    --model models/face_landmarker.task --out-dir data/mono_demo \
    --clip-start 300.0 --av-offset-frames 0 --pad-frames 2 --check
```

## 换模型而不重跑检测

```bash
# 贵步骤只做一次：源视频 -> MediaPipe 关键点轨迹
python scripts/extract_tracks.py data/raw/VIDEO_ID/source.mp4 \
    --model models/face_landmarker.task --out data/tracks/VIDEO_ID.npz --keep-subset

# 廉价步骤可重复：同一轨迹渲染成不同模型格式
python scripts/render_variant.py data/raw/VIDEO_ID/source.mp4 data/tracks/VIDEO_ID.npz \
    --spec vallr_pin --out data/derived/vallr_pin/VIDEO_ID/visual.npy
python scripts/render_variant.py data/raw/VIDEO_ID/source.mp4 data/tracks/VIDEO_ID.npz \
    --spec auto_avsr --out data/derived/auto_avsr/VIDEO_ID/visual.npy
```

`auto_avsr`/`avhubert` 采用双眼中心、鼻尖、两嘴角的五点 similarity transform，
直接映射到固定平均脸模板；`vallr_pin`、`cnvsrc_baseline` 使用嘴部框裁剪；
`syncnet` 使用 224px BGR 整脸框。所有派生物都能由 `source.mp4 + tracks.npz + spec`
重新生成，原片永远不覆盖、不转码。

## 关于源素材

`clips/` 里是从公开视频下载的片段，仅用于本项目的技术验证。视频版权归原作者，
不随仓库分发（已 gitignore）。若要对外发布数据集，业界惯例是**只发布标注与
视频 ID/时间戳，不发布视频本身**（CN-CVS、LRS3、VoxCeleb 均如此）。
