"""面向 UI/CLI 的任务提交辅助函数。"""
from __future__ import annotations

from typing import Iterable

RENDER_SPECS = {"vallr_pin", "syncnet", "avhubert", "auto_avsr", "cnvsrc_baseline"}


def clean_values(values: Iterable[str]) -> list[str]:
    """按行/逗号拆分、去空白并稳定去重。"""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for line in str(value).replace(",", "\n").splitlines():
            item = line.strip()
            if item and item not in seen:
                seen.add(item); out.append(item)
    return out


def submit_downloads(db, urls: Iterable[str], *, height: int = 1080,
                     speaker_id: str = "", priority: int = 0) -> list[str]:
    clean = clean_values(urls)
    if not clean:
        raise ValueError("at least one URL is required")
    invalid = [url for url in clean if not (url.startswith("https://") or url.startswith("http://"))]
    if invalid:
        raise ValueError(f"invalid URL: {invalid[0]}")
    ids = []
    for url in clean:
        payload = {"url": url, "height": int(height)}
        if speaker_id:
            payload["analysis"] = {"speaker_id": speaker_id}
        ids.append(db.submit("download", payload, priority=priority))
    return ids


def submit_renders(db, video_ids: Iterable[str], specs: Iterable[str], *,
                   speaker_id: str = "", priority: int = 0) -> list[str]:
    videos, formats = clean_values(video_ids), clean_values(specs)
    if not videos:
        raise ValueError("at least one video_id is required")
    if not formats:
        raise ValueError("at least one render spec is required")
    invalid = [spec for spec in formats if spec not in RENDER_SPECS]
    if invalid:
        raise ValueError(f"unknown render spec: {invalid[0]}")
    ids = []
    for video_id in videos:
        for spec in formats:
            payload = {"video_id": video_id, "spec": spec}
            if speaker_id:
                payload["speaker_id"] = speaker_id
            ids.append(db.submit("render", payload, priority=priority))
    return ids
