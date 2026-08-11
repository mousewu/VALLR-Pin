"""ROI 规格与渲染：一份人脸轨迹缓存，喂给任意模型。

**为什么需要这层抽象**：不同模型对输入的要求差异很大，而且互不兼容 ——

    VALLR-Pin / AV-HuBERT   88~96 灰度，嘴部 ROI，减均值除标准差
    SyncNet                 224 BGR 三通道，人脸框，**不归一化**，垂直中心下移
    Auto-AVSR (Ma et al.)   96 灰度，按 68 点仿射对齐到平均脸

如果直接把裁好的 ROI 存盘，就等于把裁剪决策焊死了：换一个模型，存好的数据全废，
只能回源视频重跑人脸检测 —— 而检测正是整条链路里最贵的一步 (约 5× 实时)。

正确的分层是把**贵且不可逆的计算**（人脸检测 + 关键点）与**廉价的裁剪决策**分开：

    源视频 + 轨迹缓存(landmarks)  ──render(spec)──►  任意模型要的张量

轨迹缓存很小（每帧几百字节），存一次可以反复渲染出不同规格，
而且渲染是纯几何操作，不需要 GPU、不需要模型。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# MediaPipe FaceMesh 的嘴唇外轮廓点
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]


@dataclass
class ROISpec:
    """一个模型对视觉输入的完整要求。"""
    name: str
    size: int = 96                    # 输出边长
    color: str = "gray"               # "gray" | "bgr" | "rgb"
    anchor: str = "lips"              # "lips" 以嘴部为中心 | "face" 以人脸框为中心
    align: str = "crop"               # "crop" | "similarity"（平均脸仿射对齐）
    # 裁剪边长 = scale × extent，其中 extent = 参考区域的最大边长（**不是半径**）。
    # 各家论文/代码常以半径 bs = extent/2 为基准写系数，换算时务必除以 2，
    # 否则裁剪范围会整整差一倍，模型直接失效（本仓库踩过这个坑）。
    scale: float = 1.6
    y_shift: float = 0.0              # 垂直中心偏移，单位 = extent/2 (正=下移)
    fps: Optional[int] = None         # None = 保持原帧率
    normalize: Optional[Tuple[float, float]] = None   # (mean, std)，作用在 0-1 上
    # 归一化在哪一层做。默认 "load"：渲染仍存 uint8（体积小 4 倍、且随机裁剪/翻转
    # 等增强本来就得在加载时做），由 DataLoader 的 transform 应用 normalize。
    # 只有模型要求存盘即为浮点时才用 "render"。
    normalize_at: str = "load"        # "load" | "render"
    dtype: str = "uint8"              # 存盘 dtype
    note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# 各模型的规格。加新模型时在这里追加一条，不要去改渲染逻辑。
PRESETS: Dict[str, ROISpec] = {
    "vallr_pin": ROISpec(
        name="vallr_pin", size=96, color="gray", anchor="lips", scale=1.6,
        note="存 96 训 88，多出的余量给随机裁剪；归一化放在 VideoTransform 里做"),
    "syncnet": ROISpec(
        name="syncnet", size=224, color="bgr", anchor="face", scale=1.4, y_shift=0.4,
        fps=25,
        note="官方 crop_scale=0.4 是以半径 bs 为基准的 2.8·bs，换算成 extent 即 1.4；"
             "喂 0-255 原始值，不归一化；必须 25fps"),
    "avhubert": ROISpec(
        name="avhubert", size=96, color="gray", anchor="lips", align="similarity",
        scale=1.6,
        normalize=(0.421, 0.165), normalize_at="load",
        note="AV-HuBERT / Auto-AVSR：五点相似变换对齐到固定平均脸模板；"
             "归一化由 DataLoader 应用"),
    "auto_avsr": ROISpec(
        name="auto_avsr", size=96, color="gray", anchor="lips", align="similarity",
        scale=1.6, normalize=(0.421, 0.165), normalize_at="load",
        note="Auto-AVSR 风格：双眼/鼻尖/嘴角五点相似变换，输出 96px 灰度"),
    "cnvsrc_baseline": ROISpec(
        name="cnvsrc_baseline", size=96, color="gray", anchor="lips", scale=1.5,
        note="CNVSRC baseline 的口部 ROI，训练时随机裁到 88"),
}


@dataclass
class FaceTrack:
    """一段视频的人脸轨迹缓存。这是唯一需要长期保存的中间产物。"""
    frames: np.ndarray                # (N,) 帧号
    landmarks: np.ndarray             # (N, K, 2) float16，像素坐标
    fps: float
    width: int
    height: int
    meta: Dict = field(default_factory=dict)
    point_indices: Optional[np.ndarray] = None  # compact landmarks 对应的原始索引

    def save(self, path: str) -> None:
        np.savez_compressed(path, frames=self.frames,
                            landmarks=self.landmarks.astype(np.float16),
                            fps=self.fps, width=self.width, height=self.height,
                            meta=np.array([repr(self.meta)]),
                            point_indices=(self.point_indices if self.point_indices is not None
                                           else np.array([], dtype=np.int16)))

    @classmethod
    def load(cls, path: str) -> "FaceTrack":
        d = np.load(path, allow_pickle=False)
        meta = {}
        if "meta" in d:
            try:
                meta = eval(str(d["meta"][0]), {"__builtins__": {}}, {})
            except Exception:
                meta = {}
        indices = d["point_indices"] if "point_indices" in d and len(d["point_indices"]) else None
        return cls(d["frames"], d["landmarks"].astype(np.float32), float(d["fps"]),
                   int(d["width"]), int(d["height"]), meta, indices)

    def points(self, row: int) -> np.ndarray:
        """按需恢复原始 MediaPipe 索引空间；只为当前帧分配约 4KB。"""
        if self.point_indices is None:
            return self.landmarks[row]
        n = int(self.meta.get("landmark_count", 478))
        full = np.full((n, 2), np.nan, dtype=np.float32)
        full[self.point_indices] = self.landmarks[row]
        return full

    @property
    def nbytes_per_frame(self) -> float:
        return self.landmarks.shape[1] * 2 * 2  # K 点 × xy × float16


def anchor_box(pts: np.ndarray, spec: ROISpec) -> Tuple[float, float, float]:
    """由关键点算出裁剪中心与参考半径 (cx, cy, r)。

    **必须用 nan-aware 的统计量**：轨迹缓存支持只保留必要关键点（其余为 NaN），
    若先把 NaN 当成 0 再取 min/max，人脸框会从画面原点一直拉到脸部，
    半径暴增到半个画面 —— 渲染出的图整帧缩小，模型直接失效（本仓库踩过这个坑）。
    """
    if spec.anchor == "lips":
        ref = pts[LIPS_OUTER]
    else:                                            # 整脸
        ref = pts
    x0, y0 = float(np.nanmin(ref[:, 0])), float(np.nanmin(ref[:, 1]))
    x1, y1 = float(np.nanmax(ref[:, 0])), float(np.nanmax(ref[:, 1]))
    if spec.anchor == "lips":
        cx, cy = float(np.nanmean(ref[:, 0])), float(np.nanmean(ref[:, 1]))
    else:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0) / 2
    return cx, cy + spec.y_shift * r, max(r, 4.0)


# MediaPipe FaceMesh 的五个稳定区域。眼睛取内外眼角中点，比单点更抗检测抖动。
_ALIGN_GROUPS = ((33, 133), (362, 263), (1,), (61,), (291,))

# 96×96 嘴部窗口上的平均脸模板。坐标以输出边长归一化，五点依次为
# 左眼、右眼、鼻尖、左嘴角、右嘴角。眼睛有意落在输出画面上方（负 y）：
# 它等价于“先把整脸对齐到 256px 平均脸，再裁 96px 嘴部窗口”，但把两次
# 几何变换合并成一次 warp，避免二次插值。若把眼睛也放进 96px 输出，得到的
# 会是整脸缩略图而不是 VSR 所需的嘴部 ROI——形状虽对，语义却错。
_MEAN_FACE_5_NORM = np.array([
    [0.073, -0.417], [0.927, -0.417], [0.500, 0.188],
    [0.167, 0.604], [0.833, 0.604],
], dtype=np.float32)


def alignment_points(pts: np.ndarray) -> np.ndarray:
    """从 MediaPipe 关键点得到五点相似变换锚点，返回 (5,2)。"""
    out = []
    for group in _ALIGN_GROUPS:
        p = pts[list(group)]
        if not np.isfinite(p).all():
            raise ValueError(f"仿射对齐缺少关键点 {group}；轨迹缓存需重新抽取")
        out.append(p.mean(axis=0))
    return np.asarray(out, dtype=np.float32)


def mean_face_template(size: int) -> np.ndarray:
    """返回指定输出尺寸上的五点平均脸模板。"""
    return _MEAN_FACE_5_NORM * float(size - 1)


def render_aligned_frame(frame_bgr: np.ndarray, pts: np.ndarray,
                         spec: ROISpec) -> np.ndarray:
    """用五点相似变换直接渲染到平均脸坐标系。"""
    import cv2

    src = alignment_points(pts)
    dst = mean_face_template(spec.size)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.LMEDS, refineIters=10)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("无法估计人脸相似变换")
    out = cv2.warpAffine(frame_bgr, matrix, (spec.size, spec.size),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(110, 110, 110))
    return out


def render_frame(frame_bgr: np.ndarray, pts: np.ndarray, spec: ROISpec) -> np.ndarray:
    """按 spec 从整帧里裁出一张图。frame_bgr: (H, W, 3) uint8。"""
    import cv2

    if spec.align == "similarity":
        out = render_aligned_frame(frame_bgr, pts, spec)
    else:
        cx, cy, r = anchor_box(pts, spec)
        half = max(int(r * spec.scale), 4)
        pad = half + 2
        padded = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=(110, 110, 110))
        px, py = int(cx) + pad, int(cy) + pad
        patch = padded[py - half: py + half, px - half: px + half]
        if patch.size == 0:
            patch = np.full((2 * half, 2 * half, 3), 110, dtype=np.uint8)
        out = cv2.resize(patch, (spec.size, spec.size), interpolation=cv2.INTER_LINEAR)
    if spec.color == "gray":
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    elif spec.color == "rgb":
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    if spec.dtype == "float32" or spec.normalize_at == "render":
        out = out.astype(np.float32) / 255.0
        if spec.normalize:
            m, s = spec.normalize
            out = (out - m) / s
    return out


def resolve_spec(spec: "str | ROISpec") -> ROISpec:
    if isinstance(spec, ROISpec):
        return spec
    if spec not in PRESETS:
        raise KeyError(f"未知规格 '{spec}'，可选: {sorted(PRESETS)}")
    return PRESETS[spec]


def describe_presets() -> str:
    lines = []
    for k, s in PRESETS.items():
        lines.append(f"{k:<18} {s.size:>3}px {s.color:<4} anchor={s.anchor:<5} "
                     f"align={s.align:<10} scale={s.scale} y_shift={s.y_shift} "
                     f"fps={s.fps or '原始'}  {s.note}")
    return "\n".join(lines)
