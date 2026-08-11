"""远程 worker 使用的 HTTP 控制面客户端。"""
from __future__ import annotations
import json
from urllib.request import Request, urlopen
from .db import Task


class PipelineHTTPClient:
    def __init__(self, base: str, token: str = ""):
        self.base=base.rstrip('/'); self.token=token
    def _call(self,path,data=None):
        body=None if data is None else json.dumps(data).encode()
        req=Request(self.base+path,data=body,headers={"Content-Type":"application/json",
                    "X-Pipeline-Token":self.token},method="POST" if data is not None else "GET")
        with urlopen(req,timeout=60) as r: return json.loads(r.read())
    @staticmethod
    def _task(x): return Task(**x) if x else None
    def claim(self,kinds,node_id,lease_seconds=300):
        return self._task(self._call('/api/worker/claim',{"kinds":list(kinds),"node_id":node_id,
                                                         "lease_seconds":lease_seconds}).get('task'))
    def heartbeat(self,task_id,node_id,progress=None,message='',lease_seconds=300):
        return self._call('/api/worker/heartbeat',{'task_id':task_id,'node_id':node_id,
            'progress':progress,'message':message,'lease_seconds':lease_seconds}).get('ok',False)
    def complete(self,task_id,node_id,result): self._call('/api/worker/complete',{'task_id':task_id,'node_id':node_id,'result':result})
    def fail(self,task_id,node_id,error): self._call('/api/worker/fail',{'task_id':task_id,'node_id':node_id,'error':error})
    def get(self,task_id): return self._task(self._call('/api/worker/task/'+task_id).get('task'))
    def submit(self,kind,payload,priority=0,max_attempts=3,parent_id=None,task_id=None):
        return self._call('/api/worker/submit',{'kind':kind,'payload':payload,'priority':priority,
            'max_attempts':max_attempts,'parent_id':parent_id,'task_id':task_id})['id']
    def upsert_node(self,node_id,role,hostname,status,current_task=None,capabilities=None):
        self._call('/api/worker/node',{'node_id':node_id,'role':role,'hostname':hostname,
            'status':status,'current_task':current_task,'capabilities':capabilities})
    def add_artifact(self,task_id,kind,path,sha256='',meta=None):
        self._call('/api/worker/artifact',{'task_id':task_id,'kind':kind,'path':path,
            'sha256':sha256,'meta':meta})
