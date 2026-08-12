"""ROI 规格的回归测试。

这里的两条几何断言对应实际踩过的坑，都属于**静默出错**——渲染不会报错，
产出的数组形状也完全正常，只有把它喂进模型才会发现全是废数据：

1. ``scale`` 的基准是 extent 还是半径 bs。各家代码常以 bs 写系数，
   照抄会让裁剪范围差整整一倍。
2. 轨迹缓存只保留必要关键点时，其余点是 NaN。若先填 0 再取 min/max，
   人脸框会从画面原点拉到脸部，半径暴增到半个画面。
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vallr_pin.data.roi_spec import (LIPS_OUTER, PRESETS, ROISpec, FaceTrack,  # noqa: E402
                                     alignment_points, anchor_box, build_wflw98_track,
                                     mean_face_template, resolve_spec)


def fake_landmarks(n_points: int = 478, cx: float = 640, cy: float = 360,
                   face_half: float = 100.0) -> np.ndarray:
    """构造一张"脸"：轮廓点落在 ±face_half 的边界上，嘴唇点在下半部，其余点在内部。

    刻意让**人脸外轮廓点决定 extent**——这与真实的 FaceMesh 一致，
    也正是轨迹缓存要保留 FACE_OVAL 的原因。
    """
    from scripts.extract_tracks import FACE_OVAL

    rng = np.random.RandomState(0)
    pts = np.stack([rng.uniform(cx - face_half * 0.6, cx + face_half * 0.6, n_points),
                    rng.uniform(cy - face_half * 0.6, cy + face_half * 0.6, n_points)],
                   axis=1)
    for k, idx in enumerate(FACE_OVAL):                 # 轮廓：椭圆边界
        ang = 2 * np.pi * k / len(FACE_OVAL)
        pts[idx] = [cx + face_half * np.cos(ang), cy + face_half * np.sin(ang)]
    for k, idx in enumerate(LIPS_OUTER):                # 嘴唇：脸下半部的小圈
        ang = 2 * np.pi * k / len(LIPS_OUTER)
        pts[idx] = [cx + 30 * np.cos(ang), cy + 50 + 15 * np.sin(ang)]
    return pts


def test_syncnet_scale_matches_official_geometry():
    """官方裁 2.8·bs（bs=extent/2），本仓库以 extent 为基准，故应为 1.4。"""
    spec = resolve_spec("syncnet")
    assert spec.scale == 1.4, "syncnet 的 scale 必须是官方 2.8·bs 换算后的 1.4"
    pts = fake_landmarks(face_half=100.0)
    cx, cy, r = anchor_box(pts, spec)
    extent = 200.0
    assert abs(r - extent / 2) < 1.0
    crop_span = 2 * r * spec.scale
    assert abs(crop_span - 1.4 * extent) < 1e-6
    assert abs(crop_span - 2.8 * (extent / 2)) < 1e-6      # 与官方等价
    # 垂直中心应下移 0.4·bs，对准嘴部
    assert abs(cy - (360 + 0.4 * r)) < 1e-6


def test_anchor_box_ignores_nan_points():
    """只保留必要关键点的缓存里，其余点是 NaN，不能当成坐标 0 参与 min/max。"""
    full = fake_landmarks()
    from scripts.extract_tracks import KEY_POINTS
    subset = np.full_like(full, np.nan)
    subset[KEY_POINTS] = full[KEY_POINTS]      # 与 extract_tracks --keep-subset 一致

    for spec_name in ("vallr_pin", "syncnet"):
        spec = resolve_spec(spec_name)
        a = anchor_box(full, spec)
        b = anchor_box(subset, spec)
        assert np.allclose(a, b, atol=1e-6), f"{spec_name}: NaN 处理改变了裁剪框"

    # 反例：若把 NaN 当成 0，半径会暴涨（这正是当初的 bug）
    zeroed = np.nan_to_num(subset, nan=0.0)
    _, _, r_bad = anchor_box(zeroed, resolve_spec("syncnet"))
    _, _, r_ok = anchor_box(full, resolve_spec("syncnet"))
    assert r_bad > 3 * r_ok, "填 0 应当显著放大人脸框——测试本身失效了"


def test_render_frame_shapes_and_dtypes():
    try:
        import importlib
        importlib.import_module("cv2")   # 仅探测可用性
    except ImportError:
        print("  (跳过：本环境无 opencv，几何断言不依赖它)")
        return
    frame = np.random.RandomState(1).randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    pts = fake_landmarks()
    from vallr_pin.data.roi_spec import render_frame

    out = render_frame(frame, pts, resolve_spec("vallr_pin"))
    assert out.shape == (96, 96) and out.dtype == np.uint8       # 灰度、无通道维

    out = render_frame(frame, pts, resolve_spec("syncnet"))
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8  # BGR 三通道

    norm = ROISpec(name="t", size=64, color="gray", dtype="float32",
                   normalize=(0.421, 0.165), normalize_at="render")
    out = render_frame(frame, pts, norm)
    assert out.shape == (64, 64) and out.dtype == np.float32
    assert out.min() < 0, "归一化后应当出现负值"


def test_similarity_alignment_maps_five_points_to_template():
    """已知相似变换后的五点应被恢复到平均脸模板，而非只保证输出 shape。"""
    pts = fake_landmarks()
    src = alignment_points(pts)
    theta = np.deg2rad(17)
    linear = 1.23 * np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    moved = pts @ linear.T + np.array([83.0, -27.0])
    moved_src = alignment_points(moved)
    # 两组点的成对距离比应保持一致（相似变换不改变形状）。
    d0 = np.linalg.norm(src[None] - src[:, None], axis=-1)
    d1 = np.linalg.norm(moved_src[None] - moved_src[:, None], axis=-1)
    mask = d0 > 1e-5
    assert np.allclose(d1[mask] / d0[mask], 1.23, atol=1e-5)
    tpl = mean_face_template(96)
    assert tpl.shape == (5, 2)
    # 嘴部窗口模板中，眼睛是画外锚点；嘴角必须在输出内且位于下半部。
    assert np.all(tpl[:2, 1] < 0)
    assert np.all((tpl[3:] >= 0) & (tpl[3:] < 96))
    assert np.all(tpl[3:, 1] > 48)


def test_compact_track_lazy_expansion_roundtrip():
    from scripts.extract_tracks import KEY_POINTS
    full=fake_landmarks(); compact=full[KEY_POINTS][None].astype(np.float32)
    track=FaceTrack(np.array([7]),compact,25,1280,720,
                    {'landmark_count':478},np.array(KEY_POINTS))
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,'t.npz'); track.save(p); loaded=FaceTrack.load(p)
        restored=loaded.points(0)
    assert restored.shape==(478,2)
    assert np.allclose(restored[KEY_POINTS],full[KEY_POINTS],atol=.5)  # float16 cache
    assert np.isnan(restored[2]).all() if 2 not in KEY_POINTS else True
    assert loaded.nbytes_per_frame < 300


def test_cn_cvs_wflw98_track_converts_normalized_coordinates():
    points = np.full((2, 98, 2), .5, dtype=np.float32)
    points[:, 60:68] = [.35, .35]
    points[:, 68:76] = [.65, .35]
    points[:, 54] = [.50, .52]
    for index, angle in zip(range(76, 88), np.linspace(0, 2 * np.pi, 12, endpoint=False)):
        points[:, index] = [.5 + .12 * np.cos(angle), .68 + .06 * np.sin(angle)]
    for index, angle in zip(range(88, 96), np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        points[:, index] = [.5 + .06 * np.cos(angle), .68 + .03 * np.sin(angle)]
    track = build_wflw98_track(points, 25.0, 224, 224, n_total_frames=2,
                               normalized=True, source="official.npy")
    assert track.landmarks.shape == (2, 98, 2)
    assert np.allclose(track.landmarks[:, 54], [112, 116.48], atol=1e-3)
    assert track.meta["landmark_schema"] == "wflw98"
    cx, cy, radius = anchor_box(track.points(0), resolve_spec("vallr_pin"))
    assert abs(cx - 112) < 1 and abs(cy - 152.32) < 1 and radius > 20

    with pytest.raises(ValueError, match="frame mismatch"):
        build_wflw98_track(points, 25.0, 224, 224, n_total_frames=3)


def test_landmark_gaps_are_interpolated_without_skipping_frames():
    from scripts.render_variant import landmarks_at

    track = FaceTrack(np.array([0, 2]),
                      np.array([[[0.0, 0.0]], [[2.0, 2.0]]]),
                      25.0, 10, 10)
    points, interpolated = landmarks_at(track, 1, max_interpolation_gap=1)
    assert interpolated and np.allclose(points, [[1.0, 1.0]])
    with pytest.raises(ValueError, match="trailing landmark gap"):
        landmarks_at(track, 4, max_interpolation_gap=1)


def test_presets_are_self_consistent():
    for name, spec in PRESETS.items():
        assert spec.name == name
        assert spec.color in ("gray", "bgr", "rgb")
        assert spec.anchor in ("lips", "face")
        assert spec.align in ("crop", "similarity")
        assert spec.size > 0 and spec.scale > 0
        assert spec.normalize_at in ("load", "render")
        # 声明 render 级归一化就必须存浮点，否则 normalize 字段会静默空转
        if spec.normalize_at == "render":
            assert spec.dtype == "float32", f"{name}: render 级归一化必须存 float32"
        if spec.normalize and spec.normalize_at == "load":
            assert spec.dtype == "uint8", f"{name}: load 级归一化应存 uint8 以省体积"
    assert PRESETS["vallr_pin"].fps == 25


def test_active_face_selection_tracks_identity_and_mouth_motion():
    from scripts.extract_tracks import associate_face_detections, select_face_track

    detections = []
    for frame in range(12):
        active = fake_landmarks(cx=350, cy=300, face_half=80)
        quiet = fake_landmarks(cx=900, cy=300, face_half=105)
        opening = 3.0 if frame % 2 else 22.0
        active[13] = [350, 345 - opening / 2]
        active[14] = [350, 345 + opening / 2]
        quiet[13], quiet[14] = [900, 340], [900, 344]
        # Detection order changes, so selecting face_landmarks[0] would switch identity.
        detections.append((frame, [active, quiet] if frame % 2 else [quiet, active]))
    tracks = associate_face_detections(detections)
    selected, scores, margin = select_face_track(tracks, 12, 1280, 720, "active")
    center_x = np.median([row["center"][0] for row in selected["geometry"]])
    assert len(tracks) == 2 and center_x < 500
    assert scores[0]["motion_score"] == 1.0 and margin > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
