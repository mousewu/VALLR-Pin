#!/usr/bin/env python3
"""Build the decoupled Stage-II corpus from independent Chinese text."""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vallr_pin.llm.text_data import TextBuildConfig, TextSource, build_text_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    raw = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    raw["sources"] = [TextSource(**item) for item in raw.get("sources", [])]
    report = build_text_corpus(TextBuildConfig(**raw))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
