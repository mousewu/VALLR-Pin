#!/usr/bin/env python3
"""批量筛选素材：给一批视频/频道链接，挑出真正能用于 VSR 训练的，并直接产出 manifest。

筛选按**成本递增**分四关，每关不过就立刻停手，避免为垃圾素材付下载带宽：

    第 1 关  元数据      (0 字节)   时长/帧率/分辨率/**是否有人工字幕**
    第 2 关  字幕        (~100 KB)  语速分布是否跟得住语音；条数与长度是否合理
    第 3 关  探针片段    (~4 MB)    抽 60 秒看人脸覆盖率/轨迹数/唇部像素/正脸程度
    第 4 关  完整构建    (全片)     只有前三关全过才下整片，切 ROI 出 manifest

判定标准全部可配置，且**每一条拒绝都会记录具体原因**，方便你回看是标准太严还是
素材确实不行。产物 ``report.jsonl`` 每行一个视频，含各关卡的实测指标。

用法::

    # 只筛选，不下整片
    python scripts/batch_harvest.py urls.txt --model face_landmarker.task --out-dir harvest
    # 筛完直接构建通过的那些
    python scripts/batch_harvest.py urls.txt --model ... --out-dir harvest --build

urls.txt 每行一个视频/播放列表/频道链接（``#`` 开头为注释）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_from_subtitles import check_timing, load_cues  # noqa: E402
from scripts.register_source import register  # noqa: E402


@dataclass
class Criteria:
    # 第 1 关：元数据
    min_duration: float = 120.0
    max_duration: float = 7200.0
    min_height: int = 720
    min_fps: float = 24.0
    manual_subs_langs: Tuple[str, ...] = ("zh-Hans", "zh-CN", "zh", "zh-Hant")
    allow_auto_subs: bool = False        # 自动字幕有错字+时间戳粗，默认不收
    # 第 2 关：字幕
    min_cues: int = 50
    min_rate_in_range: float = 0.85      # 语速落在 3-7 字/秒的比例
    max_overlap_ratio: float = 0.05
    # 第 3 关：画面
    max_tracks: int = 1                  # 1 = 只收单人独白；>1 需要配合 ASD
    # 主轨占比才是判断"是不是单人独白"的可靠指标：反打镜头的次轨可能只占几个百分点，
    # 单看轨迹条数会漏判（实测某双人访谈的次轨覆盖率只有 0.028~0.057）
    min_main_track_coverage: float = 0.90
    min_face_coverage: float = 0.85
    min_lip_width_px: float = 55.0
    max_abs_yaw: float = 0.12
    max_roi_jitter: float = 4.0


@dataclass
class Verdict:
    url: str
    video_id: str = ""
    title: str = ""
    passed: bool = False
    stage: str = "metadata"
    reasons: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def yt_json(url: str) -> Optional[dict]:
    p = subprocess.run(["yt-dlp", "--no-playlist", "--dump-json", url],
                       capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return None


def expand_urls(entries: List[str], limit_per_source: int) -> List[str]:
    """把频道/播放列表展开成视频链接；单个视频链接原样返回。"""
    out = []
    for e in entries:
        if "/watch?" in e or "youtu.be/" in e:
            out.append(e)
            continue
        p = subprocess.run(["yt-dlp", "--flat-playlist", "--print", "%(url)s",
                            "--playlist-end", str(limit_per_source), e],
                           capture_output=True, text=True)
        out.extend([u for u in p.stdout.split() if u.startswith("http")])
    return out


def screen_metadata(meta: dict, crit: Criteria) -> Tuple[bool, List[str], Dict]:
    reasons: List[str] = []
    dur = meta.get("duration") or 0
    subs = meta.get("subtitles") or {}
    auto = meta.get("automatic_captions") or {}
    manual_hit = [k for k in crit.manual_subs_langs if k in subs]
    auto_hit = [k for k in crit.manual_subs_langs if k in auto]
    vfmts = [f for f in meta.get("formats", []) if f.get("vcodec") not in (None, "none")]
    height = max([f.get("height") or 0 for f in vfmts] or [0])
    fps = max([f.get("fps") or 0 for f in vfmts] or [0])

    if not (crit.min_duration <= dur <= crit.max_duration):
        reasons.append(f"时长 {dur:.0f}s 不在 [{crit.min_duration:.0f},"
                       f"{crit.max_duration:.0f}] 内")
    if height < crit.min_height:
        reasons.append(f"最高画质 {height}p < {crit.min_height}p")
    if fps < crit.min_fps:
        reasons.append(f"帧率 {fps} < {crit.min_fps}")
    if not manual_hit and not (crit.allow_auto_subs and auto_hit):
        reasons.append("无中文人工字幕" + ("（只有自动字幕）" if auto_hit else ""))

    return (not reasons), reasons, {
        "duration_s": dur, "height": height, "fps": fps,
        "manual_subs": manual_hit, "auto_subs": bool(auto_hit)}


def screen_subtitles(sub_path: str, crit: Criteria) -> Tuple[bool, List[str], Dict]:
    cues = load_cues(sub_path)
    st = check_timing(cues)
    reasons = []
    if st["n_cues"] < crit.min_cues:
        reasons.append(f"字幕仅 {st['n_cues']} 条 < {crit.min_cues}")
    if st["in_normal_range"] < crit.min_rate_in_range:
        reasons.append(f"语速正常区间占比 {st['in_normal_range']:.2f} < "
                       f"{crit.min_rate_in_range}（时间戳多半是显示时间）")
    if st["overlap_ratio"] > crit.max_overlap_ratio:
        reasons.append(f"字幕重叠比例 {st['overlap_ratio']:.2f} 过高")
    return (not reasons), reasons, st


def screen_video(clip: str, model: str, crit: Criteria, out_dir: str
                 ) -> Tuple[bool, List[str], Dict]:
    from scripts.probe_video import run as probe_run
    args = SimpleNamespace(video=clip, out_dir=out_dir, model=model, roi_size=96,
                           roi_scale=1.6, max_faces=3, max_frames=0, min_track_len=15,
                           max_lag_frames=12, sync_min_corr=0.15, track_gap=15,
                           track_dist=0.8, preview=8)
    rep = probe_run(args)
    tracks = [t for t in rep["tracks"] if t["coverage"] >= 0.02]
    reasons = []
    if not tracks:
        return False, ["探针片段里没有稳定的人脸轨迹"], rep
    main = tracks[0]
    if len(tracks) > crit.max_tracks:
        reasons.append(f"人脸轨迹 {len(tracks)} 条 > {crit.max_tracks}"
                       "（多人/多机位，需要先跑 asd_pipeline.py）")
    if main["coverage"] < crit.min_main_track_coverage:
        reasons.append(f"主轨仅占 {main['coverage']:.2f} < "
                       f"{crit.min_main_track_coverage}（存在切换到其他人/机位）")
    if rep["face_coverage"] < crit.min_face_coverage:
        reasons.append(f"人脸覆盖率 {rep['face_coverage']:.2f} < {crit.min_face_coverage}")
    if main["lip_width_px"] < crit.min_lip_width_px:
        reasons.append(f"唇部宽度 {main['lip_width_px']:.0f}px < "
                       f"{crit.min_lip_width_px:.0f}px（换更高画质或放弃）")
    if main["abs_yaw_mean"] > crit.max_abs_yaw:
        reasons.append(f"平均偏航 {main['abs_yaw_mean']:.3f} > {crit.max_abs_yaw}（侧脸太多）")
    if main["roi_jitter_px_per_frame"] > crit.max_roi_jitter:
        reasons.append(f"ROI 抖动 {main['roi_jitter_px_per_frame']:.1f}px/帧过大")
    metrics = {"n_tracks_stable": len(tracks), "face_coverage": rep["face_coverage"],
               "lip_width_px": main["lip_width_px"], "abs_yaw": main["abs_yaw_mean"],
               "roi_jitter": main["roi_jitter_px_per_frame"]}
    return (not reasons), reasons, metrics


# --------------------------------------------------------------------------- #
def fetch_subs(url: str, vid: str, cache: str, langs: Tuple[str, ...]) -> Optional[str]:
    for lang in langs:
        dst = os.path.join(cache, f"{vid}.{lang}.json3")
        if os.path.exists(dst):
            return dst
    subprocess.run(["yt-dlp", "--no-playlist", "--skip-download", "--write-sub",
                    "--sub-langs", ",".join(langs), "--sub-format", "json3/srt",
                    "-o", os.path.join(cache, vid), url],
                   capture_output=True, text=True)
    for f in sorted(os.listdir(cache)):
        if f.startswith(vid + ".") and (f.endswith(".json3") or f.endswith(".srt")):
            return os.path.join(cache, f)
    return None


def fetch_clip(url: str, vid: str, cache: str, start: float, dur: float,
               height: int) -> Optional[str]:
    dst = os.path.join(cache, f"{vid}_probe.mp4")
    if os.path.exists(dst):
        return dst
    p = subprocess.run(
        ["yt-dlp", "--no-playlist", "-f",
         f"bestvideo[height<={height}]+bestaudio/best[height<={height}]",
         "--download-sections", f"*{start:.0f}-{start + dur:.0f}",
         "--force-keyframes-at-cuts", "--merge-output-format", "mp4", "-o", dst, url],
        capture_output=True, text=True)
    return dst if os.path.exists(dst) else (None if p.returncode else None)


def harvest(args) -> List[Verdict]:
    crit = Criteria(max_tracks=args.max_tracks, min_height=args.min_height,
                    allow_auto_subs=args.allow_auto_subs)
    cache = os.path.join(args.out_dir, "cache")
    os.makedirs(cache, exist_ok=True)

    entries = [l.strip() for l in open(args.urls, encoding="utf-8")
               if l.strip() and not l.startswith("#")]
    urls = expand_urls(entries, args.per_source)
    print(f"待筛选 {len(urls)} 个视频", flush=True)

    verdicts: List[Verdict] = []
    for n, url in enumerate(urls, 1):
        v = Verdict(url=url)
        meta = yt_json(url)
        if not meta:
            v.reasons = ["元数据获取失败"]
            verdicts.append(v)
            continue
        v.video_id = meta.get("id", "")
        v.title = (meta.get("title") or "")[:60]
        ok, reasons, m = screen_metadata(meta, crit)
        v.metrics["metadata"] = m
        v.reasons += reasons
        print(f"[{n}/{len(urls)}] {v.title}", flush=True)
        if not ok:
            print(f"    ✗ 元数据: {'; '.join(reasons)}", flush=True)
            verdicts.append(v)
            time.sleep(args.sleep)
            continue

        v.stage = "subtitles"
        sub = fetch_subs(url, v.video_id, cache, crit.manual_subs_langs)
        if not sub:
            v.reasons.append("字幕下载失败")
            verdicts.append(v)
            continue
        ok, reasons, st = screen_subtitles(sub, crit)
        v.metrics["subtitles"] = st
        v.reasons += reasons
        if not ok:
            print(f"    ✗ 字幕: {'; '.join(reasons)}", flush=True)
            verdicts.append(v)
            time.sleep(args.sleep)
            continue

        v.stage = "video"
        mid = max((m["duration_s"] - args.probe_seconds) / 2, 0)
        clip = fetch_clip(url, v.video_id, cache, mid, args.probe_seconds,
                          args.probe_height)
        if not clip:
            v.reasons.append("探针片段下载失败")
            verdicts.append(v)
            continue
        ok, reasons, vm = screen_video(clip, args.model, crit,
                                       os.path.join(cache, f"{v.video_id}_probe"))
        v.metrics["video"] = vm
        v.reasons += reasons
        v.passed = ok
        v.stage = "passed" if ok else "video"
        print(("    ✓ 通过 " if ok else "    ✗ 画面: " + "; ".join(reasons)) +
              (f"(唇宽{vm.get('lip_width_px', 0):.0f}px 覆盖率"
               f"{vm.get('face_coverage', 0):.2f} yaw{vm.get('abs_yaw', 0):.3f})"
               if ok else ""), flush=True)
        verdicts.append(v)
        time.sleep(args.sleep)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "report.jsonl"), "w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(asdict(v), ensure_ascii=False) + "\n")
    passed = [v for v in verdicts if v.passed]
    with open(os.path.join(args.out_dir, "passed.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(v.url for v in passed))
    return verdicts


def build_passed(verdicts: List[Verdict], args) -> None:
    """对通过筛选的视频保存完整原片，再产出可再生的派生数据。"""
    cache = os.path.join(args.out_dir, "cache")
    for v in [x for x in verdicts if x.passed]:
        raw_dir = os.path.join(args.out_dir, "raw", v.video_id)
        os.makedirs(raw_dir, exist_ok=True)
        full = os.path.join(raw_dir, "source.mp4")
        if not os.path.exists(full):
            subprocess.run(["yt-dlp", "--no-playlist", "-f",
                            f"bestvideo[height<={args.build_height}]+bestaudio/"
                            f"best[height<={args.build_height}]",
                            "--merge-output-format", "mp4", "-o", full, v.url],
                           capture_output=True, text=True)
        sub = None
        for f in sorted(os.listdir(cache)):
            if f.startswith(v.video_id + ".") and f.endswith((".json3", ".srt")):
                sub = os.path.join(cache, f)
                break
        if not (os.path.exists(full) and sub):
            print(f"  跳过 {v.video_id}: 整片或字幕缺失", flush=True)
            continue
        source = register(full, os.path.join(args.out_dir, "raw"), v.video_id,
                          url=v.url, subtitles=sub, complete=True)
        source_video = os.path.join(source["root"], source["video"])
        source_sub = os.path.join(source["root"], source["subtitles"])
        out = os.path.join(args.out_dir, "derived", "vallr_pin", v.video_id)
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "build_from_subtitles.py"),
                        source_video, source_sub, "--model", args.model, "--out-dir", out,
                        "--source-record", os.path.join(source["root"], "source.json"),
                        "--prefix", v.video_id, "--check"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", help="每行一个视频/播放列表/频道链接")
    ap.add_argument("--model", required=True, help="mediapipe face_landmarker.task")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-source", type=int, default=20, help="每个频道最多取多少个视频")
    ap.add_argument("--probe-seconds", type=float, default=60)
    ap.add_argument("--probe-height", type=int, default=720)
    ap.add_argument("--build-height", type=int, default=1080)
    ap.add_argument("--max-tracks", type=int, default=1)
    ap.add_argument("--min-height", type=int, default=720)
    ap.add_argument("--allow-auto-subs", action="store_true")
    ap.add_argument("--sleep", type=float, default=2.0, help="请求间隔，别把源站打疼")
    ap.add_argument("--build", action="store_true", help="筛完直接构建通过的视频")
    args = ap.parse_args()

    verdicts = harvest(args)
    passed = [v for v in verdicts if v.passed]
    print(f"\n通过 {len(passed)}/{len(verdicts)}")
    by_stage: Dict[str, int] = {}
    for v in verdicts:
        if not v.passed:
            by_stage[v.stage] = by_stage.get(v.stage, 0) + 1
    if by_stage:
        print("拒绝分布:", by_stage)
    if args.build and passed:
        build_passed(verdicts, args)


if __name__ == "__main__":
    main()
