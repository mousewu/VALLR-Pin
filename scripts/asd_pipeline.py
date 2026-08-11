#!/usr/bin/env python3
"""Active Speaker Detection：逐帧判定"画面上这张脸是不是正在说话的人"。

不用 SyncNet，走的是**身份对齐**路线 —— 对固定机位的多人访谈类素材，这条路
比音视频同步模型更简单也更可靠：

    人脸 embedding 聚类  ──┐
                          ├─► 共现矩阵自动配对 ─► 逐帧判定 on-screen == speaker
    声纹/基频 diarization ─┘

关键点是**两侧的簇可以自动配对**，不需要人工标注对应关系：正常剪辑下"画面上的人
就是说话人"占绝大多数时间，所以共现矩阵的主对角线天然显著。人工只需要做一件事：
给簇起名字（cluster0 = 张三），而这件事甚至可以从文字稿的说话人标签推出来。

反打镜头 (听者出镜、说话人画外) 会被判为 ``is_speaker=False``，这些帧必须丢掉 ——
它们是"闭着嘴的画面 + 别人的文本"，是最典型的毒数据。

diarization 后端：
  * ``f0``      —— 自带，零依赖，基于基频聚类。**只在说话人音高差异明显时有效**
                   (典型是一男一女)；同性别对话会失败，别硬用。
  * ``rttm``    —— 读 pyannote.audio 等工具产出的 RTTM 文件，通用方案。

用法::

    python scripts/asd_pipeline.py clip.mp4 --model face_landmarker.task \\
        --out-dir asd_out --speakers 2 --names 杨植麟 张小珺
    python scripts/asd_pipeline.py clip.mp4 --model ... --diarizer rttm --rttm out.rttm
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

SR = 16000
NO_SPEAKER = -1          # 静音 / 无人说话
UNKNOWN = -2             # 未检出人脸


# --------------------------------------------------------------------------- #
#                                  人脸侧                                       #
# --------------------------------------------------------------------------- #
def face_features(video: str, model_path: str, sample_every: int = 2,
                  max_faces: int = 1) -> Tuple[List[int], np.ndarray, List[dict], float, int]:
    """逐帧提取人脸特征。返回 (帧号, 特征矩阵, 每帧几何信息, fps, 总帧数)。

    特征 = 灰度人脸块 (直方图均衡, 抗光照) ⊕ 关键点几何构型 (尺度归一, 抗远近)。
    两者都做 L2 归一后拼接：像素分支管肤色/发型/衣着，几何分支管脸型，
    对"固定机位 + 少数几个人"的场景已经足够，不需要专门的人脸识别模型。
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_faces=max_faces))

    frames, feats, geo, i = [], [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % sample_every == 0:
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps))
            if res.face_landmarks:
                p = np.array([[q.x * W, q.y * H] for q in res.face_landmarks[0]],
                             dtype=np.float32)
                x0, y0 = max(p[:, 0].min(), 0), max(p[:, 1].min(), 0)
                x1, y1 = min(p[:, 0].max(), W), min(p[:, 1].max(), H)
                crop = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[int(y0):int(y1), int(x0):int(x1)]
                if crop.size > 100:
                    pix = cv2.equalizeHist(cv2.resize(crop, (32, 32))).astype(np.float32).ravel()
                    pix /= np.linalg.norm(pix) + 1e-6
                    fw = max(float(p[:, 0].max() - p[:, 0].min()), 1.0)
                    g = ((p - p.mean(0)) / fw).ravel()
                    g /= np.linalg.norm(g) + 1e-6
                    frames.append(i)
                    feats.append(np.concatenate([pix, g]))
                    geo.append({"cx": float(p[:, 0].mean()), "cy": float(p[:, 1].mean()),
                                "face_w": fw})
        i += 1
    cap.release()
    return frames, (np.stack(feats) if feats else np.zeros((0, 1))), geo, fps, i


def kmeans(X: np.ndarray, k: int, iters: int = 100, seed: int = 0) -> np.ndarray:
    """k-means++ 初始化的朴素 k-means（避免引入 sklearn 依赖）。"""
    rng = np.random.RandomState(seed)
    C = [X[rng.randint(len(X))]]
    for _ in range(k - 1):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        C.append(X[rng.choice(len(X), p=d / (d.sum() + 1e-12))])
    C = np.array(C)
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


def cluster_faces(feats: np.ndarray, k: int, n_pca: int = 8, seed: int = 0) -> np.ndarray:
    if len(feats) < k:
        return np.zeros(len(feats), dtype=int)
    Xc = feats - feats.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return kmeans(Xc @ Vt[: min(n_pca, Vt.shape[0])].T, k, seed=seed)


def median_smooth(labels: np.ndarray, w: int) -> np.ndarray:
    """对簇标签做多数投票平滑，消除单帧跳变。"""
    if w <= 1 or len(labels) == 0:
        return labels
    out = labels.copy()
    half = w // 2
    for i in range(len(labels)):
        seg = labels[max(0, i - half): i + half + 1]
        vals, cnts = np.unique(seg, return_counts=True)
        out[i] = vals[cnts.argmax()]
    return out


# --------------------------------------------------------------------------- #
#                                  音频侧                                       #
# --------------------------------------------------------------------------- #
def read_audio(video: str, sr: int = SR) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", video, "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def f0_track(x: np.ndarray, sr: int = SR, win_s: float = 0.04, hop_s: float = 0.02,
             fmin: float = 70, fmax: float = 320, rms_floor: float = 0.01,
             ac_floor: float = 0.3) -> Tuple[np.ndarray, float]:
    """自相关基频轨迹。无声/清音处返回 0。"""
    win, hop = int(win_s * sr), int(hop_s * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    out = []
    for i in range(0, max(len(x) - win, 0), hop):
        seg = x[i:i + win] - x[i:i + win].mean()
        if np.sqrt((seg ** 2).mean()) < rms_floor:
            out.append(0.0)
            continue
        ac = np.correlate(seg, seg, "full")[win - 1:]
        ac /= ac[0] + 1e-9
        k = int(np.argmax(ac[lo:hi])) + lo
        out.append(sr / k if ac[k] > ac_floor else 0.0)
    return np.array(out), hop / sr


def diarize_f0(f0: np.ndarray, n_speakers: int, seed: int = 0) -> np.ndarray:
    """按基频把浊音帧聚成 n 类。返回每帧的说话人簇 (静音 = NO_SPEAKER)。

    局限很明确：只有当说话人的音高分布可分时才成立 (一男一女最典型)。
    同性别、或有人音域重叠时会失败 —— 那种情况请换 --diarizer rttm。
    """
    out = np.full(len(f0), NO_SPEAKER, dtype=int)
    voiced = f0 > 0
    if voiced.sum() < n_speakers:
        return out
    z = np.log(f0[voiced])[:, None]
    lab = kmeans(z, n_speakers, seed=seed)
    # 按音高从低到高重排簇号，保证跨文件的簇号语义稳定
    order = np.argsort([z[lab == j].mean() for j in range(n_speakers)])
    remap = {int(o): j for j, o in enumerate(order)}
    out[voiced] = [remap[int(v)] for v in lab]
    return out


def smooth_voice(voice: np.ndarray, hop_s: float, med_win_s: float = 0.30,
                 max_gap_s: float = 0.60) -> np.ndarray:
    """对说话人轨迹做中值平滑 + 短静音填补。

    不做这一步，**每个字间停顿都会把片段截断一次** —— 实测 60 秒会碎成 56 段。
    另外基频法偶发的倍频误判也靠中值窗压掉。填补只发生在"缺口两侧是同一个人"
    时，说话人真正切换处的静音会被保留，不会把两个人的话粘成一段。
    """
    out = voice.copy()
    w = max(int(med_win_s / hop_s) | 1, 3)
    half = w // 2
    v = voice.copy()
    for i in range(len(v)):                      # 只在浊音帧上做中值
        if v[i] < 0:
            continue
        seg = v[max(0, i - half): i + half + 1]
        seg = seg[seg >= 0]
        if len(seg):
            vals, cnts = np.unique(seg, return_counts=True)
            out[i] = vals[cnts.argmax()]

    max_gap = int(max_gap_s / hop_s)
    i = 0
    while i < len(out):
        if out[i] == NO_SPEAKER:
            j = i
            while j < len(out) and out[j] == NO_SPEAKER:
                j += 1
            left = out[i - 1] if i > 0 else NO_SPEAKER
            right = out[j] if j < len(out) else NO_SPEAKER
            if j - i <= max_gap and left >= 0 and left == right:
                out[i:j] = left
            i = j
        else:
            i += 1
    return out


def diarize_rttm(path: str, n_frames: int, hop_s: float) -> Tuple[np.ndarray, List[str]]:
    """读 RTTM (pyannote.audio 等的标准输出)。"""
    spans, names = [], []
    for line in open(path, encoding="utf-8"):
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            start, dur, spk = float(f[3]), float(f[4]), f[7]
            if spk not in names:
                names.append(spk)
            spans.append((start, start + dur, names.index(spk)))
    out = np.full(n_frames, NO_SPEAKER, dtype=int)
    for a, b, s in spans:
        out[int(a / hop_s): int(b / hop_s)] = s
    return out, names


# --------------------------------------------------------------------------- #
#                                  两侧配对                                     #
# --------------------------------------------------------------------------- #
def match_clusters(face_lab: np.ndarray, voice_at_frame: np.ndarray, n_face: int,
                   n_voice: int) -> Tuple[Dict[int, int], np.ndarray]:
    """共现矩阵 + 全局最优配对：脸簇 i ↔ 声簇 j。

    依据是"正常剪辑里画面上的人多数时候就是说话人"，所以共现矩阵的最优匹配
    就是身份对应。这一步**不需要人工**；人工只用来给簇起名字。

    用枚举排列取全局最优，而不是逐格贪心：说话人时长极不平衡时（本例 88% vs 12%），
    贪心可能先吃掉一个大格再被迫做出错误的次优配对。说话人数很小，枚举足够。
    """
    from itertools import permutations

    M = np.zeros((n_face, n_voice), dtype=np.int64)
    for f, v in zip(face_lab, voice_at_frame):
        if f >= 0 and v >= 0:
            M[f, v] += 1
    n = min(n_face, n_voice)
    best, best_score = None, -1
    for perm in permutations(range(n_voice), n):
        score = sum(M[i, perm[i]] for i in range(n))
        if score > best_score:
            best, best_score = perm, score
    mapping = {i: int(best[i]) for i in range(n)} if best else {}
    return mapping, M


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    face_cluster: int
    speaker: str
    is_speaker: bool

    def to_dict(self, fps: float) -> dict:
        return {"start_frame": self.start_frame, "end_frame": self.end_frame,
                "start_s": round(self.start_frame / fps, 2),
                "end_s": round(self.end_frame / fps, 2),
                "n_frames": self.end_frame - self.start_frame + 1,
                "face_cluster": self.face_cluster, "speaker": self.speaker,
                "is_speaker": self.is_speaker}


def to_segments(frames: Sequence[int], face_lab: np.ndarray, is_spk: np.ndarray,
                names: Dict[int, str], min_len: int = 8) -> List[Segment]:
    segs: List[Segment] = []
    if not len(frames):
        return segs
    cur = None
    for idx, fr in enumerate(frames):
        key = (int(face_lab[idx]), bool(is_spk[idx]))
        if cur and cur[0] == key and fr - cur[2] <= 4:
            cur = (key, cur[1], fr)
        else:
            if cur:
                segs.append(cur)
            cur = (key, fr, fr)
    segs.append(cur)
    out = []
    for (fc, spk), a, b in segs:
        if b - a + 1 >= min_len:
            out.append(Segment(a, b, fc, names.get(fc, f"cluster{fc}"), spk))
    return out


# --------------------------------------------------------------------------- #
def run(args) -> dict:
    frames, feats, geo, fps, n_total = face_features(args.video, args.model,
                                                     args.sample_every)
    if not len(frames):
        raise SystemExit("没有检出任何人脸")
    face_lab = median_smooth(cluster_faces(feats, args.speakers, seed=args.seed),
                             args.smooth)

    x = read_audio(args.video)
    f0, hop_s = f0_track(x)
    if args.diarizer == "rttm":
        voice, rttm_names = diarize_rttm(args.rttm, len(f0), hop_s)
        n_voice = max(len(rttm_names), 1)
    else:
        voice = diarize_f0(f0, args.speakers, seed=args.seed)
        n_voice = args.speakers
    voice = smooth_voice(voice, hop_s, args.voice_smooth, args.max_gap)

    # 音频帧 -> 视频帧
    v_at_frame = np.array([voice[min(int(fr / fps / hop_s), len(voice) - 1)]
                           for fr in frames])
    mapping, cooc = match_clusters(face_lab, v_at_frame, args.speakers, n_voice)
    is_spk = np.array([mapping.get(int(f), -99) == int(v)
                       for f, v in zip(face_lab, v_at_frame)])

    names = {i: (args.names[i] if i < len(args.names) else f"cluster{i}")
             for i in range(args.speakers)}
    segs = to_segments(frames, face_lab, is_spk, names, args.min_seg)

    n_face_frames = len(frames)
    n_speaking = int(is_spk.sum())
    n_silent = int((v_at_frame == NO_SPEAKER).sum())
    report = {
        "video": args.video, "fps": round(fps, 2), "frames_total": n_total,
        "frames_with_face": n_face_frames,
        "diarizer": args.diarizer,
        "face_cluster_sizes": {int(k): int((face_lab == k).sum())
                               for k in range(args.speakers)},
        "cooccurrence_face_x_voice": cooc.tolist(),
        "face_to_voice_mapping": {str(k): v for k, v in mapping.items()},
        "cluster_names": {str(k): v for k, v in names.items()},
        "frames_onscreen_is_speaker": n_speaking,
        "frames_onscreen_not_speaker": int(n_face_frames - n_speaking - n_silent),
        "frames_silence": n_silent,
        "usable_ratio": round(n_speaking / max(n_face_frames, 1), 3),
        "segments": [s.to_dict(fps) for s in segs],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "asd.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(args.out_dir, "per_frame.npy"),
            np.stack([np.array(frames), face_lab, v_at_frame, is_spk.astype(int)]))

    if args.dump_faces:
        import cv2
        for k in range(args.speakers):
            pos = np.where(face_lab == k)[0]
            if not len(pos):
                continue
            cap = cv2.VideoCapture(args.video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[pos[len(pos) // 2]])
            ok, fr = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(os.path.join(args.out_dir, f"cluster{k}_face.png"), fr)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", required=True, help="mediapipe face_landmarker.task")
    ap.add_argument("--out-dir", default="asd_out")
    ap.add_argument("--speakers", type=int, default=2)
    ap.add_argument("--names", nargs="*", default=[], help="按簇号顺序给的人名")
    ap.add_argument("--diarizer", choices=["f0", "rttm"], default="f0")
    ap.add_argument("--rttm", default="")
    ap.add_argument("--sample-every", type=int, default=2, help="每隔几帧提一次特征")
    ap.add_argument("--smooth", type=int, default=9, help="簇标签多数投票窗口")
    ap.add_argument("--min-seg", type=int, default=8)
    ap.add_argument("--voice-smooth", type=float, default=0.30, help="声道中值窗(秒)")
    ap.add_argument("--max-gap", type=float, default=0.60, help="填补的最长静音(秒)")
    ap.add_argument("--dump-faces", action="store_true", help="导出每簇的代表帧供命名")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    r = run(args)
    print(json.dumps({k: v for k, v in r.items() if k != "segments"},
                     ensure_ascii=False, indent=2))
    print(f"\n共 {len(r['segments'])} 段：")
    for s in r["segments"]:
        flag = "✓可用" if s["is_speaker"] else "✗听者/画外音，丢弃"
        print(f"  [{s['start_s']:>6.2f}s - {s['end_s']:>6.2f}s] {s['n_frames']:>5}帧  "
              f"画面={s['speaker']:<8} {flag}")


if __name__ == "__main__":
    main()
