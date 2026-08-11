#!/usr/bin/env python3
"""Build leakage-safe Stage-I manifests from a YAML corpus specification."""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vallr_pin.data.corpora import BuildConfig, CorpusSpec, build_manifests  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    raw = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    raw["sources"] = [CorpusSpec(**item) for item in raw.get("sources", [])]
    report = build_manifests(BuildConfig(**raw))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
