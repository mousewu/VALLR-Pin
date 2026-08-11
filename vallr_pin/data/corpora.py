"""Normalize heterogeneous Mandarin AV corpora into one strict manifest."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

from ..text.pinyin import text_to_pinyin_mixed


@dataclass
class CorpusSpec:
    name: str
    root: str
    annotation: str
    format: str = "jsonl"       # jsonl | delimited | kaldi | sidecar
    media_root: str = ""
    media_glob: str = "**/*"
    media_extensions: List[str] = field(
        default_factory=lambda: [".npy", ".mp4", ".avi", ".mov", ".mkv"]
    )
    delimiter: str = "\t"
    has_header: bool = False
    id_field: str = "id"
    text_field: str = "text"
    video_field: str = "video"
    speaker_field: str = "speaker_id"
    global_speaker_field: str = "global_speaker_id"
    split_field: str = "split"
    id_column: int = 0
    text_column: int = 1
    speaker_column: int = -1
    split_column: int = -1
    video_column: int = -1
    default_split: str = ""
    speaker_regex: str = ""
    speaker_path_index: Optional[int] = None
    text_prefix: str = "Text:"
    supervision: str = "supervised"  # supervised | pseudo
    pseudo_confidence: float = 1.0


@dataclass
class BuildConfig:
    sources: List[CorpusSpec] = field(default_factory=list)
    out_dir: str = "data/stage1"
    seed: int = 0
    dev_speaker_percent: int = 5
    test_speaker_percent: int = 5
    min_chars: int = 2
    max_chars: int = 80
    min_frames_per_label: float = 1.5
    allow_pseudo: bool = False
    min_pseudo_confidence: float = 0.95
    require_speaker: bool = True
    absolute_paths: bool = True


def _path(root: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(root, value)


def _clean_label(text: str) -> str:
    # Keep Chinese chars and whole Latin/number tokens, drop punctuation/spaces.
    tokens, _, _ = text_to_pinyin_mixed(str(text).strip())
    return "".join(tokens)


def _records_jsonl(spec: CorpusSpec) -> Iterator[dict]:
    with open(_path(spec.root, spec.annotation), encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                yield {"id": row.get(spec.id_field, ""),
                       "text": row.get(spec.text_field, ""),
                       "video": row.get(spec.video_field, ""),
                       "speaker_id": row.get(spec.speaker_field, ""),
                       "global_speaker_id": row.get(spec.global_speaker_field, ""),
                       "split": row.get(spec.split_field, spec.default_split),
                       "confidence": row.get("confidence", spec.pseudo_confidence)}


def _records_delimited(spec: CorpusSpec) -> Iterator[dict]:
    with open(_path(spec.root, spec.annotation), encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=spec.delimiter)
        if spec.has_header:
            next(reader, None)
        for row in reader:
            if not row or max(spec.id_column, spec.text_column) >= len(row):
                continue
            yield {"id": row[spec.id_column], "text": row[spec.text_column],
                   "video": row[spec.video_column] if 0 <= spec.video_column < len(row) else "",
                   "speaker_id": row[spec.speaker_column] if 0 <= spec.speaker_column < len(row) else "",
                   "split": row[spec.split_column] if 0 <= spec.split_column < len(row) else spec.default_split,
                   "confidence": spec.pseudo_confidence}


def _read_kaldi_map(path: str) -> Dict[str, str]:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            key, sep, value = line.strip().partition(" ")
            if sep:
                out[key] = value.strip()
    return out


def _records_kaldi(spec: CorpusSpec) -> Iterator[dict]:
    base = _path(spec.root, spec.annotation)
    texts = _read_kaldi_map(os.path.join(base, "text"))
    speakers = _read_kaldi_map(os.path.join(base, "utt2spk"))
    videos = _read_kaldi_map(os.path.join(base, "video.scp"))
    for utt, text in texts.items():
        media = videos.get(utt, "")
        # Shell pipelines in scp files are not safe or meaningful for video loading.
        if media.endswith("|"):
            media = ""
        yield {"id": utt, "text": text, "video": media,
               "speaker_id": speakers.get(utt, ""), "split": spec.default_split,
               "confidence": spec.pseudo_confidence}


def _records_sidecar(spec: CorpusSpec) -> Iterator[dict]:
    pattern = _path(spec.root, spec.annotation)
    for name in sorted(glob.glob(pattern, recursive=True)):
        text_path = Path(name)
        text = ""
        for line in text_path.read_text(encoding="utf-8-sig").splitlines():
            if not spec.text_prefix or line.startswith(spec.text_prefix):
                text = line[len(spec.text_prefix):].strip() if spec.text_prefix else line.strip()
                if text:
                    break
        yield {"id": text_path.stem, "text": text, "video": "", "speaker_id": "",
               "split": spec.default_split, "_sidecar": str(text_path),
               "confidence": spec.pseudo_confidence}


def read_source(spec: CorpusSpec) -> Iterator[dict]:
    readers = {"jsonl": _records_jsonl, "delimited": _records_delimited,
               "kaldi": _records_kaldi, "sidecar": _records_sidecar}
    if spec.format not in readers:
        raise ValueError(f"unsupported annotation format: {spec.format}")
    yield from readers[spec.format](spec)


def _media_index(spec: CorpusSpec) -> Dict[str, str]:
    root = Path(_path(spec.root, spec.media_root or ""))
    allowed = {x.lower() for x in spec.media_extensions}
    found: Dict[str, str] = {}
    collisions = set()
    for path in root.glob(spec.media_glob):
        if path.is_file() and path.suffix.lower() in allowed:
            if path.stem in found:
                collisions.add(path.stem)
            else:
                found[path.stem] = str(path)
    for stem in collisions:
        found.pop(stem, None)
    return found


def _resolve_media(spec: CorpusSpec, row: dict, index: Dict[str, str]) -> str:
    value = str(row.get("video", "")).strip()
    if value:
        return _path(spec.root, value)
    media_root = _path(spec.root, spec.media_root or "")
    utt = str(row.get("id", ""))
    speaker = str(row.get("speaker_id", ""))
    stems = [utt]
    if speaker:
        stems.insert(0, os.path.join(speaker, utt))
    for stem in stems:
        for extension in spec.media_extensions:
            candidate = os.path.join(media_root, stem + extension)
            if os.path.exists(candidate):
                return candidate
        candidate = os.path.join(media_root, stem)
        if os.path.exists(candidate):
            return candidate
    return index.get(utt, "")


def _speaker(spec: CorpusSpec, row: dict, media: str) -> str:
    if row.get("global_speaker_id"):
        return str(row["global_speaker_id"])
    if row.get("speaker_id"):
        return str(row["speaker_id"])
    target = row.get("id", "") + " " + media
    if spec.speaker_regex:
        match = re.search(spec.speaker_regex, target)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    if spec.speaker_path_index is not None and media:
        parts = Path(media).parts
        index = spec.speaker_path_index
        if -len(parts) <= index < len(parts):
            return parts[index]
    return ""


def _split(speaker: str, seed: int, dev: int, test: int) -> str:
    digest = hashlib.sha1(f"{seed}:{speaker}".encode()).hexdigest()
    value = int(digest[:8], 16) % 100
    return "test" if value < test else ("dev" if value < test + dev else "train")


def _n_frames(path: str) -> int:
    if path.endswith(".npy"):
        try:
            return int(np.load(path, mmap_mode="r").shape[0])
        except Exception:
            return 0
    return 0


def build_manifests(cfg: BuildConfig) -> dict:
    """Build train/dev/test manifests and a machine-readable rejection report."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    accepted, rejected = [], Counter()
    seen_ids, seen_media = set(), set()
    speakers_by_split: Dict[str, set] = defaultdict(set)

    for spec in cfg.sources:
        if spec.supervision == "pseudo" and not cfg.allow_pseudo:
            rejected[f"{spec.name}:pseudo_disabled"] += 1
            continue
        index = _media_index(spec)
        for row in read_source(spec):
            utt = str(row.get("id", "")).strip()
            text = _clean_label(row.get("text", ""))
            if not utt or not text:
                rejected[f"{spec.name}:missing_id_or_text"] += 1; continue
            unique_id = f"{spec.name}:{utt}"
            if unique_id in seen_ids:
                rejected[f"{spec.name}:duplicate_id"] += 1; continue
            media = _resolve_media(spec, row, index)
            if not media or not os.path.exists(media):
                rejected[f"{spec.name}:missing_media"] += 1; continue
            media = os.path.abspath(media) if cfg.absolute_paths else os.path.relpath(media, cfg.out_dir)
            if media in seen_media:
                rejected[f"{spec.name}:duplicate_media"] += 1; continue
            confidence = float(row.get("confidence", spec.pseudo_confidence))
            if spec.supervision == "pseudo" and confidence < cfg.min_pseudo_confidence:
                rejected[f"{spec.name}:low_pseudo_confidence"] += 1; continue
            if not cfg.min_chars <= len(text) <= cfg.max_chars:
                rejected[f"{spec.name}:text_length"] += 1; continue
            speaker = _speaker(spec, row, media)
            if cfg.require_speaker and not speaker:
                rejected[f"{spec.name}:missing_speaker"] += 1; continue
            speaker_key = (speaker if row.get("global_speaker_id")
                           else f"{spec.name}:{speaker}")
            split = str(row.get("split") or "").lower()
            if split not in {"train", "dev", "test"}:
                split = _split(speaker_key, cfg.seed,
                               cfg.dev_speaker_percent, cfg.test_speaker_percent)
            frames = _n_frames(media)
            _, syllables, unknown = text_to_pinyin_mixed(text)
            if unknown or not syllables:
                rejected[f"{spec.name}:pinyin_unknown"] += 1; continue
            if frames and frames < max(len(syllables), len(text)) * cfg.min_frames_per_label:
                rejected[f"{spec.name}:insufficient_frames"] += 1; continue
            accepted.append({"id": unique_id, "video": media, "text": text,
                             "speaker_id": speaker_key, "source": spec.name,
                             "split": split, "n_frames": frames,
                             "supervision": spec.supervision,
                             "confidence": confidence})
            speakers_by_split[split].add(speaker_key)
            seen_ids.add(unique_id); seen_media.add(media)

    overlap = ((speakers_by_split["train"] & speakers_by_split["dev"]) |
               (speakers_by_split["train"] & speakers_by_split["test"]) |
               (speakers_by_split["dev"] & speakers_by_split["test"]))
    if overlap:
        raise ValueError(f"speaker leakage across splits: {sorted(overlap)[:10]}")

    counts, source_counts = Counter(), Counter()
    for split in ("train", "dev", "test"):
        path = os.path.join(cfg.out_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as stream:
            for item in accepted:
                if item["split"] == split:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    source_counts[f"{split}:{item['source']}"] += 1
    report = {"accepted": len(accepted), "counts": dict(counts),
              "source_counts": dict(source_counts), "rejected": dict(rejected),
              "speakers": {k: len(v) for k, v in speakers_by_split.items()},
              "speaker_overlap": 0, "config": {
                  "seed": cfg.seed, "allow_pseudo": cfg.allow_pseudo,
                  "min_pseudo_confidence": cfg.min_pseudo_confidence}}
    with open(os.path.join(cfg.out_dir, "build_report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report
