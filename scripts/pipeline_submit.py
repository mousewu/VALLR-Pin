#!/usr/bin/env python3
"""批量提交模型无关的采集任务；下载完成后自动进入 analyze。"""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vallr_pin.pipeline.db import PipelineDB  # noqa: E402
from vallr_pin.pipeline.client import PipelineHTTPClient  # noqa: E402
from vallr_pin.pipeline.submission import submit_downloads  # noqa: E402


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("urls", nargs="*"); ap.add_argument("--file")
    ap.add_argument("--db", default="data/pipeline.sqlite"); ap.add_argument("--api", default="")
    ap.add_argument("--token",default=os.environ.get("PIPELINE_TOKEN",""))
    ap.add_argument("--height", type=int, default=1080); ap.add_argument("--speaker-id", default="")
    a=ap.parse_args(); db=PipelineHTTPClient(a.api,a.token) if a.api else PipelineDB(a.db)
    urls=list(a.urls)
    if a.file: urls.extend(open(a.file,encoding="utf-8").read().splitlines())
    ids=submit_downloads(db,urls,height=a.height,speaker_id=a.speaker_id)
    print(json.dumps(ids, indent=2))


if __name__ == "__main__": main()
