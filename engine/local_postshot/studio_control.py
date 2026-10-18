"""Pure Python identity-bound safe-boundary control protocol (no torch import)."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
from .project import atomic_json

CONTROL_ACTIONS = frozenset({'pause','resume','suspend','unpause','checkpoint','stop'})
REQUEST_NAME, STATE_NAME = 'control-request.json','control-state.json'


def _read_object(path):
    path = Path(path)
    if not path.is_file():
        return None
    for attempt in range(5):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.02)
    if not isinstance(value,dict):
        raise ValueError('Control record must be an object')
    return value


def read_control_state(output):
    return _read_object(Path(output)/STATE_NAME)


def submit_control(output, package_sha256, action):
    if action not in CONTROL_ACTIONS:
        raise ValueError('Unknown training control')
    output = Path(output).resolve()
    state, pending = read_control_state(output), _read_object(output/REQUEST_NAME)
    if not state or state.get('package_sha256') != package_sha256 or not state.get('generation_id'):
        raise RuntimeError('Training control identity differs')
    status = state['status']
    if status in ('stopping','stopped','succeeded','failed','paused'):
        raise RuntimeError('Session ended; paused sessions require checkpoint resume')
    if status=='waiting_gpu' and action!='stop':
        raise RuntimeError('Only stop is available while waiting for GPU')
    if action in ('resume','unpause') and status != 'suspended':
        raise RuntimeError('Training is not suspended')
    if action == 'suspend' and status != 'running':
        raise RuntimeError('Training is not running')
    if pending and pending.get('request_id') != state.get('request_id') and action != 'stop':
        raise RuntimeError('A control request is awaiting a safe boundary')
    request = {'schema_version':1,'request_id':uuid.uuid4().hex,'package_sha256':package_sha256,
               'generation_id':state['generation_id'],'action':action}
    atomic_json(output/REQUEST_NAME,request)
    return request


class TrainingControl:
    def __init__(self, output, package_sha256, *, waiting_for_gpu=False, io_interval=.15):
        self.output, self.package_sha256 = Path(output).resolve(), package_sha256
        self.generation_id = uuid.uuid4().hex
        prior = _read_object(self.output/REQUEST_NAME)
        self.request_id = prior.get('request_id') if prior else None
        self.status, self._last_step = 'waiting_gpu' if waiting_for_gpu else 'running',0
        self.io_interval = io_interval
        self._last_poll = self._last_publication = float('-inf')
        self.revision = 0
        self.on_event = None
        self._publish(self.status,0,request_id=self.request_id)

    def _publish(self,status,step,*,request_id=None,message=None,event=True):
        self.status, self._last_step = status,step
        payload = {'schema_version':1,'package_sha256':self.package_sha256,'generation_id':self.generation_id,
                   'status':status,'step':step,'request_id':request_id,
                   'updated_at':datetime.now(timezone.utc).isoformat()}
        if message:
            payload['message'] = message
        atomic_json(self.output/STATE_NAME,payload)
        self._last_publication = time.monotonic()
        self.revision += 1
        if event and self.on_event is not None:
            self.on_event()

    def _next_request(self):
        now = time.monotonic()
        if now-self._last_poll < self.io_interval:
            return None
        self._last_poll = now
        try:
            request = _read_object(self.output/REQUEST_NAME)
        except (OSError, ValueError):
            return None
        if not request or request.get('request_id') == self.request_id:
            return None
        rid = request.get('request_id')
        if not isinstance(rid,str) or not rid:
            return None
        self.request_id = rid
        if (request.get('schema_version') != 1 or request.get('package_sha256') != self.package_sha256 or
                request.get('generation_id') != self.generation_id or request.get('action') not in CONTROL_ACTIONS):
            self._publish(self.status,self._last_step,request_id=rid,message='identity_or_action_mismatch')
            return None
        return request

    def is_set(self):
        if self.status == 'stopping':
            return True
        if self.status != 'waiting_gpu':
            return False
        request = self._next_request()
        if request:
            self._publish('stopping' if request['action'] == 'stop' else 'waiting_gpu',self._last_step,
                          request_id=request['request_id'])
        return self.status == 'stopping'

    def start_running(self):
        if self.status != 'waiting_gpu':
            raise RuntimeError('Session is not waiting for GPU')
        self._publish('running',self._last_step,request_id=self.request_id)

    def observe(self,step,save,*,target=None,tick=None):
        # Always keep the actual safe boundary in memory, even between disk IO.
        self._last_step = step
        while True:
            request = self._next_request()
            if request:
                action = request['action']
                if action in ('pause','stop'):
                    save()
                    self._publish('paused' if action == 'pause' else 'stopping',step,request_id=self.request_id)
                    return action
                if action == 'checkpoint':
                    save()
                    self._publish(self.status,step,request_id=self.request_id,message='checkpoint_saved')
                elif action == 'suspend':
                    self._publish('suspended',step,request_id=self.request_id)
                elif action in ('resume','unpause'):
                    self._publish('running',step,request_id=self.request_id)
            if self.status != 'suspended':
                if time.monotonic()-self._last_publication >= self.io_interval:
                    self._publish(self.status,step,request_id=self.request_id,event=False)
                return None
            if tick:
                action = tick()
                if action in ('pause','stop'):
                    save()
                    self._publish('paused' if action=='pause' else 'stopping',step,request_id=self.request_id)
                    return action
            time.sleep(.1)

    def complete(self,status,step):
        if status not in ('paused','stopped','succeeded','failed'):
            raise ValueError(status)
        self._publish(status,step,request_id=self.request_id)
