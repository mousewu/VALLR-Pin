#!/usr/bin/env python3
"""启动独立下载节点或处理节点。"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vallr_pin.pipeline.db import PipelineDB  # noqa: E402
from vallr_pin.pipeline.worker import Worker  # noqa: E402
from vallr_pin.pipeline.client import PipelineHTTPClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument(
        "--role", choices=["download", "analyze", "render", "process"], required=True,
        help="process 仅用于消费重构前的遗留任务")
    ap.add_argument("--db", default="data/pipeline.sqlite"); ap.add_argument("--data-root", default="data")
    ap.add_argument("--node-id", default=""); ap.add_argument("--poll", type=float, default=2)
    ap.add_argument("--lease", type=int, default=300); ap.add_argument("--once", action="store_true")
    ap.add_argument("--api",default="",help="远程控制面 URL；设置后不直接打开 SQLite")
    ap.add_argument("--token",default=os.environ.get('PIPELINE_TOKEN',''))
    a = ap.parse_args(); control=PipelineHTTPClient(a.api,a.token) if a.api else PipelineDB(a.db)
    Worker(control, a.role, a.data_root, a.node_id,
                                a.poll, a.lease).run_forever(a.once)


if __name__ == "__main__": main()
