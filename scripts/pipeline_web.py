#!/usr/bin/env python3
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vallr_pin.pipeline.web import serve  # noqa: E402

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--db",default="data/pipeline.sqlite")
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8080)
    p.add_argument("--static",default="web"); p.add_argument("--worker-token",default=os.environ.get('PIPELINE_TOKEN',''))
    a=p.parse_args(); serve(a.db,a.host,a.port,a.static,a.worker_token)
