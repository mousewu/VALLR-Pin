"""规模化控制面、分片与划分回归测试。"""
from __future__ import annotations
import json, os, sys, tempfile
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vallr_pin.pipeline.db import PipelineDB
from vallr_pin.data.shards import TarShardWriter, load_wds_uri
from vallr_pin.data.dataset import _load_video
from scripts.build_dataset_catalog import build as build_catalog
from vallr_pin.pipeline.submission import submit_downloads, submit_renders
from vallr_pin.pipeline.worker import Worker


def test_db_state_machine_and_retry():
    with tempfile.TemporaryDirectory() as d:
        db=PipelineDB(os.path.join(d,'p.sqlite')); tid=db.submit('download',{'url':'x'},max_attempts=2)
        t=db.claim(['download'],'n1',10); assert t.id==tid and t.attempts==1
        assert db.heartbeat(tid,'n1',.4,'working'); db.fail(tid,'n1','temporary')
        assert db.get(tid).status=='queued'
        t=db.claim(['download'],'n2',10); db.complete(tid,'n2',{'ok':1})
        assert db.get(tid).status=='succeeded' and db.get(tid).result['ok']==1


def test_pause_cancel_retry():
    with tempfile.TemporaryDirectory() as d:
        db=PipelineDB(os.path.join(d,'p.sqlite')); tid=db.submit('process',{})
        db.operate(tid,'pause'); assert db.get(tid).status=='paused'
        db.operate(tid,'resume'); db.operate(tid,'cancel'); assert db.get(tid).status=='cancelled'
        db.operate(tid,'retry'); assert db.get(tid).status=='queued'


def test_three_stage_batch_submission_has_no_model_in_collection():
    with tempfile.TemporaryDirectory() as d:
        db=PipelineDB(os.path.join(d,'p.sqlite'))
        ids=submit_downloads(db,["https://youtu.be/a\nhttps://youtu.be/b", "https://youtu.be/a"],
                             speaker_id="speaker-1")
        assert len(ids)==2
        downloads=db.list_tasks(kind="download")
        assert all("spec" not in t.payload and "process" not in t.payload for t in downloads)
        assert all(t.payload["analysis"]["speaker_id"]=="speaker-1" for t in downloads)
        render_ids=submit_renders(db,["a","b"],["vallr_pin","avhubert"])
        assert len(render_ids)==4
        renders=db.list_tasks(kind="render")
        assert {(t.payload["video_id"],t.payload["spec"]) for t in renders} == {
            ("a","vallr_pin"),("a","avhubert"),("b","vallr_pin"),("b","avhubert")}


def test_download_enqueues_analyze_not_render(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db=PipelineDB(os.path.join(d,"p.sqlite")); raw=Path(d)/"raw"/"vid"
        raw.mkdir(parents=True); (raw/"source.mp4").write_bytes(b"video")
        (raw/"downloaded_subs.zh.json3").write_text("{}")
        import scripts.register_source as source_module
        monkeypatch.setattr(source_module,"register",lambda *a,**k: {
            "sha256":"abc","bytes":5,"video_id":"vid"})
        tid=db.submit("download",{"url":"https://youtu.be/vid","video_id":"vid"})
        task=db.claim(["download"],"downloader",30)
        worker=Worker(db,"download",d,"downloader")
        monkeypatch.setattr(worker,"run_command",lambda *a,**k: "")
        result=worker.download(task)
        analyze=db.get(result["analyze_task_id"])
        assert analyze.kind=="analyze" and analyze.payload=={"video_id":"vid"}
        assert db.list_tasks(kind="render")==[] and db.list_tasks(kind="process")==[]


def test_shard_rotation_and_dataset_uri():
    with tempfile.TemporaryDirectory() as d:
        w=TarShardWriter(os.path.join(d,'shards'),max_samples=2)
        uris=[]
        for i in range(5): uris.append(w.write(f'u{i}',np.full((3,8,8),i,np.uint8),{'id':i}))
        paths=w.close(); assert len(paths)==3 and not list(__import__('pathlib').Path(d).rglob('*.partial'))
        a=load_wds_uri('wds://shards/'+uris[3][6:],d); assert a.shape==(3,8,8) and a[0,0,0]==3
        assert _load_video('wds://shards/'+uris[0][6:],d).dtype==np.uint8


def test_content_dedupe_preserves_same_text_different_speakers():
    with tempfile.TemporaryDirectory() as d:
        items=[]
        patterns=[]
        a=np.zeros((4,16,16),np.uint8); a[:,:,:8]=220
        b=np.zeros((4,16,16),np.uint8); b[:,:8,:]=220
        patterns=[a,b,a.copy()]
        for i,arr in enumerate(patterns):
            np.save(os.path.join(d,f'{i}.npy'),arr)
            items.append({'id':str(i),'video':f'{i}.npy','text':'相同文本','n_frames':4,
                          'source_id':f's{i}','speaker_id':f'p{i}'})
        m=os.path.join(d,'m.jsonl'); open(m,'w').write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items))
        r=build_catalog([m],os.path.join(d,'out'))
        # 0 与 2 是同文本同视觉，应删一个；1 是同文本不同视觉，必须保留。
        assert r['after_dedup']==2 and r['duplicates']==1
        rows=[]
        for s in ('train','dev','test'):
            rows += [json.loads(x) for x in open(os.path.join(d,'out',s+'.jsonl'))]
        assert len({x['speaker_id'] for x in rows})==2


if __name__=='__main__':
    fs=[v for k,v in sorted(globals().items()) if k.startswith('test_')]
    for f in fs: f(); print('ok ',f.__name__)
    print(f'\n{len(fs)} passed')
