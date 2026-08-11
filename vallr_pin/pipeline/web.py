"""零第三方依赖的 Pipeline Web 控制台与 JSON API。"""

from __future__ import annotations

import dataclasses
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .db import PipelineDB
from .submission import submit_downloads, submit_renders


class App:
    def __init__(self, db: PipelineDB, static_dir: str, worker_token: str = ""):
        self.db, self.static, self.worker_token = db, Path(static_dir), worker_token

    def handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def json(self, obj, status=200):
                data = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

            def body(self):
                n = int(self.headers.get("Content-Length", 0)); return json.loads(self.rfile.read(n) or b"{}")

            def worker_auth(self):
                return not app.worker_token or self.headers.get("X-Pipeline-Token","") == app.worker_token

            def do_GET(self):
                u = urlparse(self.path); q = parse_qs(u.query)
                if u.path == "/api/summary": return self.json(app.db.summary())
                if u.path == "/api/tasks":
                    rows = app.db.list_tasks(int(q.get("limit", [200])[0]), q.get("status", [""])[0],
                                             q.get("kind", [""])[0])
                    return self.json([dataclasses.asdict(x) for x in rows])
                if u.path.startswith("/api/tasks/"):
                    tid = u.path.split("/")[3]; task = app.db.get(tid)
                    if not task: return self.json({"error":"not found"}, 404)
                    return self.json({"task":dataclasses.asdict(task), "events":app.db.events(tid)})
                if u.path.startswith("/api/worker/task/"):
                    if not self.worker_auth(): return self.json({"error":"unauthorized"},401)
                    task=app.db.get(u.path.rsplit('/',1)[-1])
                    return self.json({"task":dataclasses.asdict(task) if task else None})
                path = "index.html" if u.path == "/" else u.path.lstrip("/")
                target = (app.static / path).resolve()
                if app.static.resolve() not in target.parents and target != app.static.resolve():
                    return self.send_error(403)
                if not target.exists(): return self.send_error(404)
                data=target.read_bytes(); self.send_response(200)
                self.send_header("Content-Type",mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)

            def do_POST(self):
                u=urlparse(self.path)
                try:
                    b=self.body()
                    if u.path.startswith('/api/worker/'):
                        if not self.worker_auth(): return self.json({"error":"unauthorized"},401)
                        action=u.path.rsplit('/',1)[-1]
                        if action=='claim':
                            t=app.db.claim(b['kinds'],b['node_id'],int(b.get('lease_seconds',300)))
                            return self.json({'task':dataclasses.asdict(t) if t else None})
                        if action=='heartbeat':
                            ok=app.db.heartbeat(b['task_id'],b['node_id'],b.get('progress'),b.get('message',''),int(b.get('lease_seconds',300)))
                            return self.json({'ok':ok})
                        if action=='complete': app.db.complete(b['task_id'],b['node_id'],b['result']); return self.json({'ok':True})
                        if action=='fail': app.db.fail(b['task_id'],b['node_id'],b['error']); return self.json({'ok':True})
                        if action=='submit':
                            tid=app.db.submit(b['kind'],b['payload'],int(b.get('priority',0)),int(b.get('max_attempts',3)),b.get('parent_id'),b.get('task_id'))
                            return self.json({'id':tid})
                        if action=='node': app.db.upsert_node(b['node_id'],b['role'],b['hostname'],b['status'],b.get('current_task'),b.get('capabilities')); return self.json({'ok':True})
                        if action=='artifact': app.db.add_artifact(b['task_id'],b['kind'],b['path'],b.get('sha256',''),b.get('meta')); return self.json({'ok':True})
                    if u.path == "/api/tasks":
                        urls=b.get("urls", [b["url"]] if b.get("url") else [])
                        ids=submit_downloads(app.db,urls,height=int(b.get("height",1080)),
                                             speaker_id=b.get("speaker_id", ""))
                        return self.json({"ids":ids,"count":len(ids)},201)
                    if u.path == "/api/renders":
                        ids=submit_renders(app.db,b.get("video_ids",[]),b.get("specs",[]),
                                           speaker_id=b.get("speaker_id", ""))
                        return self.json({"ids":ids,"count":len(ids)},201)
                    if u.path.startswith("/api/tasks/"):
                        parts=u.path.split("/"); tid,action=parts[3],parts[4]
                        app.db.operate(tid,action); return self.json({"ok":True})
                    return self.json({"error":"not found"},404)
                except (KeyError,ValueError) as e: return self.json({"error":str(e)},400)

            def log_message(self, fmt, *args):
                print(f"[web] {self.address_string()} {fmt % args}")
        return Handler


def serve(db_path="data/pipeline.sqlite", host="127.0.0.1", port=8080,
          static_dir="web", worker_token: str = ""):
    app=App(PipelineDB(db_path),static_dir,worker_token); server=ThreadingHTTPServer((host,port),app.handler())
    print(f"Pipeline UI: http://{host}:{port}"); server.serve_forever()
