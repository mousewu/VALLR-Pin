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
    enabled: bool = True
    format: str = "jsonl"       # jsonl | delimited | kaldi | sidecar | cn_cvs | cmlr
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
    # Optional explicit speaker split. This is preferable to percentage hashing
    # for corpora such as CMLR that only contain a small number of speakers.
    speaker_split_map: Dict[str, str] = field(default_factory=dict)
    # Base used to derive collision-safe sidecar IDs. Empty means the static
    # (non-glob) prefix of ``annotation``.
    sidecar_id_root: str = ""
    text_prefix: str = "Text:"
    supervision: str = "supervised"  # supervised | pseudo
    pseudo_confidence: float = 1.0
    # Spatial form of the media referenced by this source.  Keeping this in the
    # manifest prevents a full news frame or a face crop from being silently
    # treated as a model-ready mouth ROI.
    input_type: str = "mouth_roi"     # raw_scene | face_crop | mouth_roi
    # Required description for a source that already ships model-ready mouth
    # arrays. Raw/face-crop inputs get these values from ROI preprocessing.
    fps: float = 25.0
    roi_spec: str = "external"


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
                output = {"id": row.get(spec.id_field, ""),
                       "text": row.get(spec.text_field, ""),
                       "video": row.get(spec.video_field, ""),
                       "speaker_id": row.get(spec.speaker_field, ""),
                       "global_speaker_id": row.get(spec.global_speaker_field, ""),
                       "split": row.get(spec.split_field, spec.default_split),
                       "confidence": row.get("confidence", spec.pseudo_confidence)}
                for key in ("landmark_path", "landmark_format", "face_box_path",
                            "audio_path", "n_frames"):
                    if row.get(key) not in (None, ""):
                        output[key] = row[key]
                yield output


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
    if spec.sidecar_id_root:
        id_root = Path(_path(spec.root, spec.sidecar_id_root)).resolve()
    else:
        wildcard = min((pattern.find(ch) for ch in "*?[" if ch in pattern),
                       default=len(pattern))
        static = pattern[:wildcard]
        id_root = Path(static if static.endswith(os.sep) else os.path.dirname(static)).resolve()
    for name in sorted(glob.glob(pattern, recursive=True)):
        text_path = Path(name)
        text = ""
        for line in text_path.read_text(encoding="utf-8-sig").splitlines():
            if not spec.text_prefix or line.startswith(spec.text_prefix):
                text = line[len(spec.text_prefix):].strip() if spec.text_prefix else line.strip()
                if text:
                    break
        try:
            uid = _relative_no_suffix(text_path.resolve(), id_root)
        except ValueError:
            uid = text_path.stem
        yield {"id": uid, "text": text, "video": "", "speaker_id": "",
               "split": spec.default_split, "_sidecar": str(text_path),
               "confidence": spec.pseudo_confidence}


def _first_nonempty_line(path: Path) -> str:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _relative_no_suffix(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


def _records_cn_cvs(spec: CorpusSpec) -> Iterator[dict]:
    """Read the official CN-CVS speaker-directory release.

    A speaker directory contains parallel ``video/``, ``txt/``,
    ``landmark/``, ``roi/`` and ``audio/`` directories.  The supplied 98-point
    landmarks are normalized to the 224x224 face crop and are carried through
    the raw manifest so ROI preprocessing can avoid a second face detector.
    """
    root = Path(spec.root).resolve()
    pattern = spec.annotation or "**/txt/*.txt"
    for text_path in sorted(root.glob(pattern)):
        if not text_path.is_file() or text_path.parent.name != "txt":
            continue
        speaker_root = text_path.parent.parent
        stem = text_path.stem
        video = speaker_root / "video" / f"{stem}.mp4"
        landmark_candidates = (
            speaker_root / "landmark" / f"{stem}_landmark.npy",
            speaker_root / "landmark" / f"{stem}.npy",
        )
        landmark = next((path for path in landmark_candidates if path.exists()), None)
        face_box = speaker_root / "roi" / f"{stem}.json"
        audio = speaker_root / "audio" / f"{stem}.wav"
        uid = (speaker_root.relative_to(root) / stem).as_posix()
        row = {"id": uid, "text": _first_nonempty_line(text_path),
               "video": str(video), "speaker_id": speaker_root.name,
               "split": spec.default_split, "confidence": spec.pseudo_confidence}
        if landmark is not None:
            row.update({"landmark_path": str(landmark),
                        "landmark_format": "wflw98_normalized",
                        "n_frames": _n_frames(str(landmark))})
        if face_box.exists():
            row["face_box_path"] = str(face_box)
        if audio.exists():
            row["audio_path"] = str(audio)
        yield row


def _records_cmlr(spec: CorpusSpec) -> Iterator[dict]:
    """Read the official CMLR mirrored text/video directory release.

    Text ``root/text/sN/date/x.txt`` maps to video
    ``root/sN/date/x.mp4``.  The complete relative path is the utterance key;
    basenames are not unique across speakers and dates in the full corpus.
    """
    root = Path(spec.root).resolve()
    text_root = root / "text"
    pattern = spec.annotation or "text/**/*.txt"
    for text_path in sorted(root.glob(pattern)):
        if not text_path.is_file():
            continue
        try:
            relative = text_path.relative_to(text_root)
        except ValueError:
            raise ValueError(
                f"CMLR annotation must be below {text_root}: {text_path}") from None
        if len(relative.parts) < 3:
            continue
        uid = relative.with_suffix("").as_posix()
        video = root / relative.with_suffix(".mp4")
        speaker = relative.parts[0]
        yield {"id": uid, "text": _first_nonempty_line(text_path),
               "video": str(video), "speaker_id": speaker,
               "split": spec.speaker_split_map.get(speaker, spec.default_split),
               "confidence": spec.pseudo_confidence}


def read_source(spec: CorpusSpec) -> Iterator[dict]:
    readers = {"jsonl": _records_jsonl, "delimited": _records_delimited,
               "kaldi": _records_kaldi, "sidecar": _records_sidecar,
               "cn_cvs": _records_cn_cvs, "cmlr": _records_cmlr}
    if spec.format not in readers:
        raise ValueError(f"unsupported annotation format: {spec.format}")
    yield from readers[spec.format](spec)


def _media_index(spec: CorpusSpec) -> Dict[str, str]:
    root = Path(_path(spec.root, spec.media_root or ""))
    allowed = {x.lower() for x in spec.media_extensions}
    found: Dict[str, str] = {}
    by_basename: Dict[str, List[str]] = defaultdict(list)
    for path in root.glob(spec.media_glob):
        if path.is_file() and path.suffix.lower() in allowed:
            relative = path.relative_to(root).with_suffix("").as_posix()
            found[relative] = str(path)
            by_basename[path.stem].append(str(path))
    # Keep basename lookup only when it is unambiguous. Relative keys are never
    # discarded, so mirrored corpora do not lose colliding utterances.
    for stem, paths in by_basename.items():
        if len(paths) == 1 and stem not in found:
            found[stem] = paths[0]
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
    return index.get(utt.replace(os.sep, "/"), "")


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


def _roi_geometry(path: str) -> tuple[int, int, int] | None:
    if not path.endswith(".npy"):
        return None
    try:
        shape = np.load(path, mmap_mode="r").shape
    except Exception:
        return None
    if len(shape) == 3:
        return int(shape[1]), int(shape[2]), 1
    if len(shape) == 4 and shape[-1] in (1, 3):
        return int(shape[1]), int(shape[2]), int(shape[3])
    return None


def build_manifests(cfg: BuildConfig) -> dict:
    """Build train/dev/test manifests and a machine-readable rejection report."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    accepted, rejected = [], Counter()
    seen_ids, seen_media = set(), set()
    speakers_by_split: Dict[str, set] = defaultdict(set)

    for spec in cfg.sources:
        if not spec.enabled:
            rejected[f"{spec.name}:source_disabled"] += 1
            continue
        if spec.supervision not in {"supervised", "pseudo"}:
            raise ValueError(
                f"{spec.name}: supervision must be supervised or pseudo for CTC training")
        if spec.input_type not in {"raw_scene", "face_crop", "mouth_roi"}:
            raise ValueError(
                f"{spec.name}: input_type must be raw_scene, face_crop or mouth_roi")
        if spec.supervision == "pseudo" and not cfg.allow_pseudo:
            rejected[f"{spec.name}:pseudo_disabled"] += 1
            continue
        index = {} if spec.format in {"cn_cvs", "cmlr"} else _media_index(spec)
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
                split = str(spec.speaker_split_map.get(speaker, "")).lower()
            if split not in {"train", "dev", "test"}:
                split = _split(speaker_key, cfg.seed,
                               cfg.dev_speaker_percent, cfg.test_speaker_percent)
            frames = int(row.get("n_frames", 0) or _n_frames(media))
            roi_geometry = _roi_geometry(media) if spec.input_type == "mouth_roi" else None
            if spec.input_type == "mouth_roi":
                if roi_geometry is None:
                    rejected[f"{spec.name}:invalid_mouth_roi"] += 1; continue
                if spec.fps <= 0:
                    rejected[f"{spec.name}:missing_roi_fps"] += 1; continue
                if roi_geometry[0] != roi_geometry[1] or roi_geometry[2] != 1:
                    rejected[f"{spec.name}:invalid_mouth_roi_geometry"] += 1; continue
            _, syllables, unknown = text_to_pinyin_mixed(text)
            if unknown or not syllables:
                rejected[f"{spec.name}:pinyin_unknown"] += 1; continue
            if frames and frames < max(len(syllables), len(text)) * cfg.min_frames_per_label:
                rejected[f"{spec.name}:insufficient_frames"] += 1; continue
            extra_paths = {}
            for key in ("landmark_path", "face_box_path", "audio_path"):
                value = str(row.get(key, "")).strip()
                if value:
                    resolved = _path(spec.root, value)
                    extra_paths[key] = (os.path.abspath(resolved) if cfg.absolute_paths
                                        else os.path.relpath(resolved, cfg.out_dir))
            landmark_format = str(row.get("landmark_format", "")).strip()
            accepted.append({"id": unique_id, "video": media, "text": text,
                             "speaker_id": speaker_key, "source": spec.name,
                             "split": split, "n_frames": frames,
                             "input_type": spec.input_type,
                             "source_input_type": spec.input_type,
                             "roi_type": ("mouth" if spec.input_type == "mouth_roi"
                                          else ""),
                             **({"roi_spec": spec.roi_spec, "fps": float(spec.fps),
                                "roi_height": roi_geometry[0],
                                "roi_width": roi_geometry[1],
                                "roi_channels": roi_geometry[2]}
                               if roi_geometry is not None else {}),
                             **extra_paths,
                             **({"landmark_format": landmark_format}
                                if landmark_format else {}),
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
