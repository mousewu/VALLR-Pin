"""下载、分析、渲染 worker；三类 worker 可运行在不同节点。"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

from .db import PipelineDB, Task


class Worker:
    def __init__(self, db: PipelineDB, role: str, data_root: str,
                 node_id: str = "", poll_seconds: float = 2,
                 lease_seconds: int = 300):
        if role not in {"download", "analyze", "render", "process"}:
            raise ValueError("role must be download, analyze, render or legacy process")
        self.db, self.role = db, role
        self.data_root = str(Path(data_root).resolve())
        self.node_id = node_id or f"{role}-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self.poll_seconds, self.lease_seconds = poll_seconds, lease_seconds
        self.handlers: dict[str, Callable[[Task], dict]] = {
            "download": self.download, "analyze": self.analyze,
            "render": self.render, "process": self.process}

    def run_command(self, task: Task, cmd: list[str], message: str,
                    progress: float | None = None) -> str:
        # stdout=PIPE + readline 会在没有换行的长任务上阻塞，导致租约过期。
        # 临时日志文件既避免 pipe 填满，又保证心跳循环永不被子进程输出阻塞。
        log = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False)
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        last = time.time()
        try:
          while p.poll() is None:
            if time.time() - last > 10:
                current = self.db.get(task.id)
                if not current or current.status != "running" or current.leased_by != self.node_id:
                    p.terminate(); raise RuntimeError("task cancelled or lease lost")
                self.db.heartbeat(task.id, self.node_id, progress=progress, message=message,
                                  lease_seconds=self.lease_seconds)
                last = time.time()
            time.sleep(.2)
        finally:
            log.flush(); log.seek(0); output = log.read(); name = log.name; log.close()
            try: os.unlink(name)
            except OSError: pass
        if p.returncode:
            raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{output[-5000:]}")
        return output

    def download(self, task: Task) -> dict:
        p = task.payload; url = p["url"]
        video_id = p.get("video_id")
        if not video_id:
            video_id = self.run_command(task, ["yt-dlp", "--no-playlist", "--print", "%(id)s", url],
                                        "resolving metadata", .05).strip().splitlines()[-1]
        raw = Path(self.data_root) / "raw" / video_id; raw.mkdir(parents=True, exist_ok=True)
        video = raw / "source.mp4"
        if not video.exists():
            self.run_command(task, ["yt-dlp", "--no-playlist", "-f",
                f"bestvideo[height<={p.get('height',1080)}]+bestaudio/best[height<={p.get('height',1080)}]",
                "--merge-output-format", "mp4", "-o", str(video), url], "downloading source", .25)
        # 字幕直接放 raw 临时名，register_source 会以不可变名称登记。
        self.run_command(task, ["yt-dlp", "--no-playlist", "--skip-download", "--write-sub",
            "--sub-langs", p.get("sub_langs", "zh-Hans,zh-CN,zh"), "--sub-format", "json3",
            "-o", str(raw / "downloaded_subs"), url], "downloading subtitles", .8)
        subs = sorted(raw.glob("downloaded_subs.*.json3"))
        if not subs:
            raise RuntimeError("manual Chinese subtitles not found")
        from scripts.register_source import register
        rec = register(str(video), str(Path(self.data_root) / "raw"), video_id,
                       url=url, subtitles=str(subs[0]), complete=True)
        self.db.add_artifact(task.id, "source", str(raw / "source.json"), rec["sha256"])
        analyze_payload = {"video_id": video_id, **p.get("analysis", {})}
        analyze_id = self.db.submit("analyze", analyze_payload, parent_id=task.id,
                                    max_attempts=p.get("analyze_max_attempts", 2))
        return {"video_id": video_id, "source_record": str(raw / "source.json"),
                "analyze_task_id": analyze_id, "bytes": rec["bytes"]}

    def analyze(self, task: Task) -> dict:
        p = task.payload; vid = p["video_id"]
        raw = Path(self.data_root) / "raw" / vid
        rec = json.loads((raw / "source.json").read_text(encoding="utf-8"))
        video = raw / rec["video"]
        tracks = Path(self.data_root) / "tracks" / f"{vid}.npz"
        tracks.parent.mkdir(parents=True, exist_ok=True)
        project = Path(__file__).resolve().parents[2]
        py = p.get("python", sys.executable)
        model = p.get("face_model", str(project / "models" / "face_landmarker.task"))
        if not tracks.exists():
            self.run_command(task, [py, str(project / "scripts" / "extract_tracks.py"),
                str(video), "--model", model, "--out", str(tracks), "--keep-subset"],
                "extracting model-independent landmarks", .65)
        analysis = {"schema_version": 1, "video_id": vid,
                    "source_record": str(raw / "source.json"), "tracks": str(tracks),
                    "subtitles": str(raw / rec["subtitles"]) if rec.get("subtitles") else None,
                    "speaker_id": p.get("speaker_id", "")}
        analysis_path = tracks.with_suffix(".json")
        tmp = analysis_path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(analysis_path)
        self.db.add_artifact(task.id, "analysis", str(analysis_path), meta=analysis)
        self.db.add_artifact(task.id, "tracks", str(tracks))
        return analysis

    def render(self, task: Task) -> dict:
        p = task.payload; vid = p["video_id"]
        raw = Path(self.data_root) / "raw" / vid
        rec = json.loads((raw / "source.json").read_text(encoding="utf-8"))
        if not rec.get("subtitles"):
            raise RuntimeError("source has no subtitles")
        video, subs = raw / rec["video"], raw / rec["subtitles"]
        tracks = Path(self.data_root) / "tracks" / f"{vid}.npz"
        if not tracks.exists():
            raise RuntimeError(f"analysis is not ready: {tracks}")
        analysis_path = tracks.with_suffix(".json")
        analysis = (json.loads(analysis_path.read_text(encoding="utf-8"))
                    if analysis_path.exists() else {})
        project = Path(__file__).resolve().parents[2]
        py = p.get("python", sys.executable)
        spec = p["spec"]
        out = Path(self.data_root) / "derived" / spec / vid
        cmd = [py, str(project / "scripts" / "stream_build.py"), str(video), str(subs),
               str(tracks), "--out-dir", str(out), "--spec", spec, "--source-id", vid,
               "--speaker-id", p.get("speaker_id", analysis.get("speaker_id", "")), "--shard-samples",
               str(p.get("shard_samples", 1000)), "--av-offset-frames",
               str(p.get("av_offset_frames", 0)), "--pad-frames", str(p.get("pad_frames", 2))]
        self.run_command(task, cmd, "streaming samples into shards", .7)
        report = json.loads((out / "build_report.json").read_text(encoding="utf-8"))
        self.db.add_artifact(task.id, "manifest", str(out / "manifest.jsonl"), meta=report)
        self.db.add_artifact(task.id, "tracks", str(tracks))
        return {"video_id": vid, "manifest": str(out / "manifest.jsonl"), **report}

    def process(self, task: Task) -> dict:
        """兼容重构前已入队的 process：先分析，再按原 spec 渲染。"""
        self.analyze(task)
        return self.render(task)

    def run_forever(self, once: bool = False) -> None:
        self.db.upsert_node(self.node_id, self.role, socket.gethostname(), "idle",
                            capabilities={"python": sys.version.split()[0]})
        while True:
            task = self.db.claim([self.role], self.node_id, self.lease_seconds)
            if not task:
                self.db.upsert_node(self.node_id, self.role, socket.gethostname(), "idle")
                if once: return
                time.sleep(self.poll_seconds); continue
            self.db.upsert_node(self.node_id, self.role, socket.gethostname(), "busy", task.id)
            try:
                result = self.handlers[task.kind](task)
                self.db.complete(task.id, self.node_id, result)
            except Exception:
                self.db.fail(task.id, self.node_id, traceback.format_exc())
            finally:
                self.db.upsert_node(self.node_id, self.role, socket.gethostname(), "idle")
            if once: return
