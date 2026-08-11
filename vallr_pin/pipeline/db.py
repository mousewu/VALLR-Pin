"""SQLite 任务状态机：下载、分析和渲染节点共享同一个控制面 API。

SQLite 适合单机或共享 POSIX 文件系统上的小型集群。跨地域/对象存储部署时，可保持
本模块接口不变，将实现替换成 PostgreSQL。
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


TERMINAL = {"succeeded", "failed", "cancelled"}
ACTIVE = {"queued", "running", "paused"}


@dataclass
class Task:
    id: str
    kind: str
    status: str
    payload: Dict[str, Any]
    result: Dict[str, Any]
    progress: float
    message: str
    attempts: int
    max_attempts: int
    leased_by: Optional[str]
    lease_until: Optional[float]
    parent_id: Optional[str]
    created_at: float
    updated_at: float
    error: str = ""


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  payload_json TEXT NOT NULL,
  result_json TEXT NOT NULL DEFAULT '{}',
  progress REAL NOT NULL DEFAULT 0,
  message TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  priority INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  leased_by TEXT,
  lease_until REAL,
  parent_id TEXT REFERENCES tasks(id),
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_claim
  ON tasks(kind,status,priority DESC,created_at);
CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  hostname TEXT NOT NULL,
  status TEXT NOT NULL,
  current_task TEXT,
  capabilities_json TEXT NOT NULL DEFAULT '{}',
  last_seen REAL NOT NULL,
  started_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT,
  level TEXT NOT NULL,
  message TEXT NOT NULL,
  data_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id,id DESC);
CREATE TABLE IF NOT EXISTS artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL,
  UNIQUE(task_id,kind,path)
);
"""


class PipelineDB:
    def __init__(self, path: str = "data/pipeline.sqlite"):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=30000")
        return con

    def init(self) -> None:
        with self.connect() as con:
            con.executescript(SCHEMA)

    def submit(self, kind: str, payload: Dict[str, Any], priority: int = 0,
               max_attempts: int = 3, parent_id: str | None = None,
               task_id: str | None = None) -> str:
        task_id = task_id or uuid.uuid4().hex
        now = time.time()
        with self.connect() as con:
            con.execute("""INSERT INTO tasks
              (id,kind,status,payload_json,priority,max_attempts,parent_id,created_at,updated_at)
              VALUES(?,?,'queued',?,?,?,?,?,?)""",
                        (task_id, kind, json.dumps(payload, ensure_ascii=False), priority,
                         max_attempts, parent_id, now, now))
            self._event(con, task_id, "info", "task submitted", {"kind": kind})
        return task_id

    def claim(self, kinds: Iterable[str], node_id: str,
              lease_seconds: int = 300) -> Optional[Task]:
        kinds = tuple(kinds)
        if not kinds:
            return None
        now = time.time()
        marks = ",".join("?" for _ in kinds)
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            # 过期租约自动回队列；attempts 已在领取时增加，避免死循环。
            con.execute("""UPDATE tasks SET status='queued',leased_by=NULL,lease_until=NULL,
                           message='lease expired; requeued',updated_at=?
                           WHERE status='running' AND lease_until<? AND attempts<max_attempts""",
                        (now, now))
            con.execute("""UPDATE tasks SET status='failed',error='lease expired too many times',
                           leased_by=NULL,lease_until=NULL,updated_at=?
                           WHERE status='running' AND lease_until<? AND attempts>=max_attempts""",
                        (now, now))
            row = con.execute(f"""SELECT * FROM tasks
                WHERE status='queued' AND kind IN ({marks})
                ORDER BY priority DESC,created_at LIMIT 1""", kinds).fetchone()
            if not row:
                con.execute("COMMIT")
                return None
            changed = con.execute("""UPDATE tasks SET status='running', attempts=attempts+1,
                leased_by=?,lease_until=?,updated_at=?,message='claimed by '||?
                WHERE id=? AND status='queued'""",
                                  (node_id, now + lease_seconds, now, node_id, row["id"])).rowcount
            con.execute("COMMIT")
            return self.get(row["id"]) if changed else None

    def heartbeat(self, task_id: str, node_id: str, progress: float | None = None,
                  message: str = "", lease_seconds: int = 300) -> bool:
        now = time.time()
        fields = ["lease_until=?", "updated_at=?"]
        vals: list[Any] = [now + lease_seconds, now]
        if progress is not None:
            fields.append("progress=?")
            vals.append(max(0.0, min(1.0, progress)))
        if message:
            fields.append("message=?")
            vals.append(message)
        vals += [task_id, node_id]
        with self.connect() as con:
            return bool(con.execute(
                f"UPDATE tasks SET {','.join(fields)} WHERE id=? AND leased_by=? AND status='running'",
                vals).rowcount)

    def complete(self, task_id: str, node_id: str, result: Dict[str, Any]) -> None:
        now = time.time()
        with self.connect() as con:
            n = con.execute("""UPDATE tasks SET status='succeeded',progress=1,result_json=?,
                message='completed',leased_by=NULL,lease_until=NULL,updated_at=?
                WHERE id=? AND leased_by=? AND status='running'""",
                            (json.dumps(result, ensure_ascii=False), now, task_id, node_id)).rowcount
            if not n:
                raise RuntimeError("task lease lost before completion")
            self._event(con, task_id, "info", "task completed", result)

    def fail(self, task_id: str, node_id: str, error: str) -> None:
        now = time.time()
        with self.connect() as con:
            row = con.execute("SELECT attempts,max_attempts FROM tasks WHERE id=?", (task_id,)).fetchone()
            retry = row and row["attempts"] < row["max_attempts"]
            status = "queued" if retry else "failed"
            con.execute("""UPDATE tasks SET status=?,error=?,message=?,leased_by=NULL,
                lease_until=NULL,updated_at=? WHERE id=? AND leased_by=?""",
                        (status, error[-8000:], "retry queued" if retry else "failed",
                         now, task_id, node_id))
            self._event(con, task_id, "error", error[-2000:], {"retry": bool(retry)})

    def operate(self, task_id: str, action: str) -> None:
        allowed = {"pause", "resume", "cancel", "retry"}
        if action not in allowed:
            raise ValueError(f"unsupported action: {action}")
        now = time.time()
        with self.connect() as con:
            row = con.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                raise KeyError(task_id)
            status = row["status"]
            target = {"pause": "paused", "resume": "queued", "cancel": "cancelled",
                      "retry": "queued"}[action]
            valid = ((action == "pause" and status == "queued") or
                     (action == "resume" and status == "paused") or
                     (action == "cancel" and status in ACTIVE) or
                     (action == "retry" and status in {"failed", "cancelled"}))
            if not valid:
                raise ValueError(f"cannot {action} task in {status}")
            con.execute("""UPDATE tasks SET status=?,leased_by=NULL,lease_until=NULL,
                error=CASE WHEN ?='queued' THEN '' ELSE error END,updated_at=? WHERE id=?""",
                        (target, target, now, task_id))
            self._event(con, task_id, "info", f"task {action}", {})

    def get(self, task_id: str) -> Optional[Task]:
        with self.connect() as con:
            row = con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._task(row) if row else None

    def list_tasks(self, limit: int = 200, status: str = "", kind: str = "") -> list[Task]:
        where, vals = [], []
        if status:
            where.append("status=?"); vals.append(status)
        if kind:
            where.append("kind=?"); vals.append(kind)
        sql = "SELECT * FROM tasks" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY created_at DESC LIMIT ?"; vals.append(limit)
        with self.connect() as con:
            return [self._task(r) for r in con.execute(sql, vals)]

    def events(self, task_id: str, limit: int = 100) -> list[dict]:
        with self.connect() as con:
            return [dict(r) for r in con.execute(
                "SELECT * FROM events WHERE task_id=? ORDER BY id DESC LIMIT ?",
                (task_id, limit))]

    def upsert_node(self, node_id: str, role: str, hostname: str, status: str,
                    current_task: str | None = None, capabilities: Dict | None = None) -> None:
        now = time.time()
        with self.connect() as con:
            con.execute("""INSERT INTO nodes(id,role,hostname,status,current_task,
                capabilities_json,last_seen,started_at) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET role=excluded.role,hostname=excluded.hostname,
                status=excluded.status,current_task=excluded.current_task,
                capabilities_json=excluded.capabilities_json,last_seen=excluded.last_seen""",
                        (node_id, role, hostname, status, current_task,
                         json.dumps(capabilities or {}), now, now))

    def list_nodes(self) -> list[dict]:
        with self.connect() as con:
            return [dict(r) for r in con.execute("SELECT * FROM nodes ORDER BY role,id")]

    def add_artifact(self, task_id: str, kind: str, path: str, sha256: str = "",
                     meta: Dict | None = None) -> None:
        with self.connect() as con:
            con.execute("""INSERT OR REPLACE INTO artifacts(task_id,kind,path,sha256,meta_json,created_at)
                           VALUES(?,?,?,?,?,?)""",
                        (task_id, kind, path, sha256, json.dumps(meta or {}), time.time()))

    def summary(self) -> dict:
        with self.connect() as con:
            states = {r["status"]: r["n"] for r in con.execute(
                "SELECT status,count(*) n FROM tasks GROUP BY status")}
            kinds = {r["kind"]: r["n"] for r in con.execute(
                "SELECT kind,count(*) n FROM tasks GROUP BY kind")}
        return {"states": states, "kinds": kinds, "nodes": self.list_nodes()}

    @staticmethod
    def _task(row: sqlite3.Row) -> Task:
        return Task(id=row["id"], kind=row["kind"], status=row["status"],
                    payload=json.loads(row["payload_json"]),
                    result=json.loads(row["result_json"]), progress=row["progress"],
                    message=row["message"], attempts=row["attempts"],
                    max_attempts=row["max_attempts"], leased_by=row["leased_by"],
                    lease_until=row["lease_until"], parent_id=row["parent_id"],
                    created_at=row["created_at"], updated_at=row["updated_at"],
                    error=row["error"])

    @staticmethod
    def _event(con: sqlite3.Connection, task_id: str, level: str,
               message: str, data: Dict) -> None:
        con.execute("INSERT INTO events(task_id,level,message,data_json,created_at) VALUES(?,?,?,?,?)",
                    (task_id, level, message, json.dumps(data, ensure_ascii=False), time.time()))
