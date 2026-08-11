#!/usr/bin/env python3
"""为已完成 analyze 的视频批量提交一个或多个模型适配渲染任务。"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vallr_pin.pipeline.client import PipelineHTTPClient  # noqa: E402
from vallr_pin.pipeline.db import PipelineDB  # noqa: E402
from vallr_pin.pipeline.submission import RENDER_SPECS, submit_renders  # noqa: E402


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("video_ids",nargs="+")
    ap.add_argument("--spec",action="append",required=True,choices=sorted(RENDER_SPECS))
    ap.add_argument("--db",default="data/pipeline.sqlite"); ap.add_argument("--api",default="")
    ap.add_argument("--token",default=os.environ.get("PIPELINE_TOKEN",""))
    ap.add_argument("--speaker-id",default="")
    a=ap.parse_args(); control=PipelineHTTPClient(a.api,a.token) if a.api else PipelineDB(a.db)
    print(json.dumps(submit_renders(control,a.video_ids,a.spec,speaker_id=a.speaker_id),indent=2))


if __name__ == "__main__": main()
