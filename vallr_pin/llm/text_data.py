"""Build a video-independent Chinese text corpus for Stage-II."""

from __future__ import annotations

import csv
from contextlib import ExitStack
import hashlib
import json
import os
import re
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterator, List

from ..text.pinyin import UNK_SYLLABLE, clean_text, text_to_pinyin
from .noise import PinyinNoiseConfig, corrupt_pinyin
from .prompt import build_messages


@dataclass
class TextSource:
    name: str
    path: str
    format: str = "text"  # text | jsonl | delimited
    text_field: str = "text"
    document_field: str = "document_id"
    text_column: int = 0
    document_column: int = -1
    delimiter: str = "\t"
    has_header: bool = False
    encoding: str = "utf-8"


@dataclass
class TextBuildConfig:
    sources: List[TextSource] = field(default_factory=list)
    out_dir: str = "data/stage2_text"
    seed: int = 2026
    val_percent: int = 1
    test_percent: int = 1
    min_chars: int = 4
    max_chars: int = 80
    deduplicate: bool = True
    exclude_paths: List[str] = field(default_factory=list)


_SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")


def _source_rows(spec: TextSource) -> Iterator[Dict[str, str]]:
    if spec.format == "text":
        with open(spec.path, encoding=spec.encoding) as stream:
            for index, line in enumerate(stream):
                yield {"text": line.strip(), "document_id": str(index)}
        return
    if spec.format == "jsonl":
        with open(spec.path, encoding=spec.encoding) as stream:
            for index, line in enumerate(stream):
                if line.strip():
                    row = json.loads(line)
                    yield {"text": str(row.get(spec.text_field, "")),
                           "document_id": str(row.get(spec.document_field, index))}
        return
    if spec.format == "delimited":
        with open(spec.path, encoding=spec.encoding, newline="") as stream:
            reader = csv.reader(stream, delimiter=spec.delimiter)
            if spec.has_header:
                next(reader, None)
            for index, row in enumerate(reader):
                if spec.text_column >= len(row):
                    continue
                doc = (row[spec.document_column]
                       if 0 <= spec.document_column < len(row) else str(index))
                yield {"text": row[spec.text_column], "document_id": doc}
        return
    raise ValueError(f"unsupported Stage-II text format: {spec.format}")


def _sentences(text: str) -> Iterator[str]:
    for fragment in _SENTENCE_SPLIT.split(text):
        value = clean_text(fragment)
        if value:
            yield value


def _split(document_id: str, seed: int, val: int, test: int) -> str:
    value = int(hashlib.sha1(f"{seed}:{document_id}".encode()).hexdigest()[:8], 16) % 100
    return "test" if value < test else ("val" if value < test + val else "train")


def _excluded(paths: List[str]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    row = json.loads(line)
                    text = row.get("text", row.get("ref", ""))
                else:
                    text = line
                values.update(_sentences(str(text)))
    return values


def build_text_corpus(cfg: TextBuildConfig) -> Dict[str, object]:
    """Create clean text/Pinyin rows; Pinyin noise is generated online by SFT."""
    if cfg.val_percent + cfg.test_percent >= 100:
        raise ValueError("val_percent + test_percent must be below 100")
    os.makedirs(cfg.out_dir, exist_ok=True)
    blocked = _excluded(cfg.exclude_paths)
    # Keep fixed-size digests rather than full sentences and stream split files;
    # memory grows with the dedup index, not with decoded rows or output text.
    seen: set[bytes] = set()
    counts = Counter()
    rejected, source_counts = Counter(), Counter()
    with ExitStack() as stack:
        outputs = {
            split: stack.enter_context(open(os.path.join(cfg.out_dir, f"{split}.jsonl"),
                                            "w", encoding="utf-8"))
            for split in ("train", "val", "test")
        }
        for spec in cfg.sources:
            for row in _source_rows(spec):
                document = f"{spec.name}:{row['document_id']}"
                for text in _sentences(row["text"]):
                    if not cfg.min_chars <= len(text) <= cfg.max_chars:
                        rejected["text_length"] += 1
                        continue
                    if text in blocked:
                        rejected["heldout_contamination"] += 1
                        continue
                    digest = hashlib.sha1(text.encode()).digest()
                    if cfg.deduplicate and digest in seen:
                        rejected["duplicate"] += 1
                        continue
                    _, pinyin = text_to_pinyin(text)
                    if not pinyin or UNK_SYLLABLE in pinyin:
                        rejected["pinyin_unknown"] += 1
                        continue
                    split = _split(document, cfg.seed, cfg.val_percent, cfg.test_percent)
                    uid = hashlib.sha1(f"{spec.name}:{text}".encode()).hexdigest()[:20]
                    item = {"id": uid, "text": text, "pinyin": pinyin,
                            "source": spec.name, "document_id": document}
                    outputs[split].write(json.dumps(item, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    source_counts[f"{split}:{spec.name}"] += 1
                    seen.add(digest)
    report = {"counts": {k: counts[k] for k in ("train", "val", "test")},
              "source_counts": dict(source_counts), "rejected": dict(rejected),
              "excluded_texts": len(blocked), "seed": cfg.seed}
    with open(os.path.join(cfg.out_dir, "build_report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def materialize_instruction_data(input_jsonl: str, output_jsonl: str,
                                 noise: PinyinNoiseConfig | None = None,
                                 variants_per_text: int = 1, seed: int = 2026) -> int:
    """Freeze online noise into ``messages`` rows for ms-swift/other trainers."""
    if variants_per_text < 1:
        raise ValueError("variants_per_text must be positive")
    noise = noise or PinyinNoiseConfig()
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    written = 0
    with open(input_jsonl, encoding="utf-8") as source, \
            open(output_jsonl, "w", encoding="utf-8") as target:
        for row_index, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            for variant in range(variants_per_text):
                rng = random.Random(seed + row_index * 1_000_003 + variant)
                corrupted, meta = corrupt_pinyin(row["pinyin"], noise, rng)
                item = {"messages": build_messages(corrupted, answer=row["text"]),
                        "meta": {"id": row.get("id"), "source": row.get("source"),
                                 "noise": meta}}
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
    return written
