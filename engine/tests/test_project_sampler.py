import io
import json
import random
from pathlib import Path
import os
import subprocess
import sys
import threading
import numpy as np
import pytest
from PIL import Image
from local_postshot import project
from local_postshot.reconstruction import import_media, reconstruction_key
from local_postshot.training import Sampler, atomic_checkpoint
from local_postshot.worker import Worker


def test_project_relative_artifacts_and_settings(tmp_path):
    p=project.create("Test",tmp_path/"project")
    p["models"].append(project.model_record(tmp_path/"project"/"models"/"m.ply",12))
    project.save(p)
    on_disk=json.loads((tmp_path/"project"/"project.json").read_text())
    assert not on_disk["models"][0]["path"].startswith(str(tmp_path))
    loaded=project.load(p["path"])
    assert loaded["models"][0]["path"]==str(tmp_path/"project"/"models"/"m.ply")
    with pytest.raises(ValueError):
        project.update(p["path"],{"steps":0})


def test_import_hash_cache_identity_and_changed_file(tmp_path):
    p=project.create("Test",tmp_path/"project")
    image=tmp_path/"source.png"
    Image.new("RGB",(64,48),(100,60,20)).save(image)
    p=import_media(p["path"],[str(image)],"images",lambda *_:None,threading.Event())
    first=reconstruction_key(p)
    p=import_media(p["path"],[str(image)],"images",lambda *_:None,threading.Event())
    assert len(p["assets"]["images"])==1 and first==reconstruction_key(p)
    Image.new("RGB",(64,48),(100,10,20)).save(p["assets"]["images"][0]["path"])
    assert reconstruction_key(p)!=first


def test_sampler_resume_next_hundred_indices_exact():
    a=Sampler(list(range(13)),42)
    for _ in range(19): a.next()
    state=a.state_dict()
    expected=[a.next() for _ in range(100)]
    b=Sampler(list(range(13)),999); b.load_state_dict(state)
    assert expected==[b.next() for _ in range(100)]


def test_worker_unknown_and_busy_are_framed_errors():
    sink=io.StringIO(); w=Worker(sink)
    w.active={"id":"train1","method":"train","path":"project"}
    w.receive({"id":"busy","method":"project.create","params":{"name":"wrong"}})
    result=json.loads(sink.getvalue())
    assert result["type"]=="error" and result["error"]["code"]=="BUSY"


@pytest.mark.parametrize("bad_request",[None,[],1,"hello",{"id":"x","method":"status","params":[]}])
def test_worker_non_object_json_stays_alive(bad_request):
    sink=io.StringIO(); w=Worker(sink)
    w.receive(bad_request)
    w.receive({"id":"still-alive","method":"status","params":{}})
    rows=[json.loads(line) for line in sink.getvalue().splitlines()]
    assert rows[0]["type"]=="error" and rows[1]["id"]=="still-alive"


def test_import_failure_is_transactional_and_resize_reimport_updates(tmp_path):
    p=project.create("Transactional",tmp_path/"project")
    good=tmp_path/"good.png"; bad=tmp_path/"bad.txt"
    Image.new("RGB",(100,80),(100,60,20)).save(good);bad.write_text("invalid")
    with pytest.raises(ValueError):
        import_media(p["path"],[str(good),str(bad)],"images",lambda *_:None,threading.Event())
    assert not project.load(p["path"])["assets"]["images"]
    assert not list((Path(p["path"])/"images").glob("*"))
    p=import_media(p["path"],[str(good)],"images",lambda *_:None,threading.Event(),max_image_size=64)
    assert p["assets"]["images"][0]["width"]==64
    p=import_media(p["path"],[str(good)],"images",lambda *_:None,threading.Event(),max_image_size=32)
    assert len(p["assets"]["images"])==1 and p["assets"]["images"][0]["width"]==32


def test_same_image_selected_twice_commits_only_once(tmp_path):
    p=project.create("Duplicates",tmp_path/"project")
    good=tmp_path/"good.png";Image.new("RGB",(32,24),(50,60,70)).save(good)
    p=import_media(p["path"],[str(good),str(good)],"images",lambda *_:None,threading.Event())
    assert len(p["assets"]["images"])==1
    assert len(list((Path(p["path"])/"images").glob("*.png")))==1


def test_photo_import_preserves_camera_prior_and_consumes_orientation(tmp_path):
    p=project.create("EXIF",tmp_path/"project")
    photo=tmp_path/"camera.jpg"
    exif=Image.Exif();exif[271]="Camera Make";exif[272]="Camera Model";exif[37386]=50;exif[274]=6
    Image.new("RGB",(100,80),(20,50,90)).save(photo,exif=exif)
    p=import_media(p["path"],[str(photo)],"images",lambda *_:None,threading.Event(),max_image_size=64)
    item=p["assets"]["images"][0]
    assert (item["width"],item["height"])==(51,64)
    with Image.open(item["path"]) as imported:
        assert imported.getexif()[272]=="Camera Model" and imported.getexif()[37386]==50
        assert imported.getexif().get(274,1)==1


def test_reimport_video_changed_fps_excludes_stale_frames(tmp_path):
    from pathlib import Path
    from local_postshot.runtime import executable
    from local_postshot.reconstruction import process
    try:
        ffmpeg=executable("ffmpeg")
    except RuntimeError:
        pytest.skip("Runtime FFmpeg unavailable")
    source=tmp_path/"portrait.mp4"
    process([ffmpeg,"-hide_banner","-y","-f","lavfi","-i","testsrc=size=64x96:rate=4",
             "-t","2","-pix_fmt","yuv420p","-c:v","libx264",source],lambda *_:None,threading.Event())
    p=project.create("Video reimport",tmp_path/"project")
    p=import_media(p["path"],[str(source)],"video",lambda *_:None,threading.Event(),
                   video_fps=2,max_frames=4,max_image_size=64)
    assert len(p["assets"]["images"])==4
    assert all(max(i["width"],i["height"])<=64 for i in p["assets"]["images"])
    first_sources={i["source"] for i in p["assets"]["images"]}
    p=import_media(p["path"],[str(source)],"video",lambda *_:None,threading.Event(),
                   video_fps=.5,max_frames=8,max_image_size=64)
    assert len(p["assets"]["images"])==1 and len(p["assets"]["videos"])==1
    assert first_sources.isdisjoint(i["source"] for i in p["assets"]["images"])


def test_checkpoint_atomic_failure_preserves_previous(tmp_path,monkeypatch):
    torch=pytest.importorskip("torch")
    file=tmp_path/"latest.pt"
    atomic_checkpoint(file,{"valid":torch.tensor([3.])})
    def broken(*args,**kwargs): raise OSError("disk full")
    monkeypatch.setattr(torch,"save",broken)
    with pytest.raises(OSError): atomic_checkpoint(file,{"bad":1})
    assert torch.load(file,weights_only=False)["valid"].item()==3


def test_real_worker_spz_native_logs_do_not_pollute_protocol(tmp_path):
    pytest.importorskip("spz")
    from local_postshot.formats import write_spz
    cloud={"means":np.array([[0.,0.,0.]],np.float32),"scales":np.full((1,3),-3,np.float32),
           "quats":np.array([[1.,0.,0.,0.]],np.float32),"opacities":np.array([1.],np.float32),
           "sh":np.zeros((1,1,3),np.float32)}
    model=tmp_path/"native.spz";write_spz(model,cloud)
    p=project.create("Native stdout",tmp_path/"project")
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    run=subprocess.run([sys.executable,"-m","local_postshot.worker","--method","model.import",
                        "--params",json.dumps({"path":p["path"],"file":str(model)})],
                       capture_output=True,text=True,encoding="utf-8",env=env,timeout=30)
    assert run.returncode==0,run.stderr
    rows=[json.loads(line) for line in run.stdout.splitlines()]
    assert len(rows)==1 and rows[0]["type"]=="result"
    assert rows[0]["result"]["models"][0]["count"]==1
    failed=subprocess.run([sys.executable,"-m","local_postshot.worker","--method","not-a-method"],
                          capture_output=True,text=True,encoding="utf-8",env=env,timeout=30)
    assert failed.returncode!=0 and json.loads(failed.stdout)["type"]=="error"
