import io
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from local_postshot import project
from local_postshot.locks import operation_lock,is_active,pid_alive
from local_postshot.worker import Worker


def test_other_worker_open_preserves_active_status_and_mutations_are_busy(tmp_path):
    p=project.create("Shared",tmp_path/"shared")
    p["status"]="training";project.save(p)
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    script=("from local_postshot.locks import operation_lock;import sys;"
            "ctx=operation_lock(sys.argv[1],'train');ctx.__enter__();"
            "print('ready',flush=True);sys.stdin.read();ctx.__exit__(None,None,None)")
    child=subprocess.Popen([sys.executable,"-c",script,p["path"]],stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8",env=env)
    try:
        assert child.stdout.readline().strip()=="ready"
        assert is_active(p["path"])
        sink=io.StringIO();worker=Worker(sink)
        worker.receive({"id":"open","method":"project.open","params":{"path":p["path"]}})
        assert json.loads(sink.getvalue())["result"]["status"]=="training"
        assert project.load(p["path"])["status"]=="training"
        sink.seek(0);sink.truncate()
        worker.receive({"id":"update","method":"project.update","params":{"path":p["path"],"settings":{"steps":99}}})
        worker.thread.join(timeout=10)
        assert json.loads(sink.getvalue())["error"]["code"]=="PROJECT_BUSY"
        assert project.load(p["path"])["status"]=="training"
        child.kill();child.wait(timeout=10)
        assert not is_active(p["path"])
        sink.seek(0);sink.truncate()
        worker.receive({"id":"recover","method":"project.open","params":{"path":p["path"]}})
        assert json.loads(sink.getvalue())["result"]["status"]=="interrupted"
        assert project.update(p["path"],{"steps":99})["settings"]["steps"]==99
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=10)
        child.stdin.close();child.stdout.close();child.stderr.close()


def test_lock_is_reentrant_on_task_thread_and_released_after_exception(tmp_path):
    p=project.create("Nested",tmp_path/"project")
    with pytest.raises(RuntimeError):
        with operation_lock(p["path"],"outer"):
            assert is_active(p["path"])
            assert project.update(p["path"],{"steps":88})["settings"]["steps"]==88
            raise RuntimeError("Simulated task failure")
    assert not is_active(p["path"])
    assert pid_alive(os.getpid())


def test_stale_marker_with_alive_reused_pid_does_not_override_os_lock(tmp_path):
    p=project.create("Stale PID",tmp_path/"project")
    (Path(p["path"])/".engine-task.json").write_text(json.dumps({"pid":os.getpid(),"method":"dead task"}))
    assert not is_active(p["path"])
    with operation_lock(p["path"],"new task"):
        assert is_active(p["path"])
