"""CLI-specific Run lock wrapper."""
from functools import wraps
from gsstudio.infrastructure.runtime.run_lock import run_session
from gsstudio.interfaces.cli.paths import scene_path


def run_cli_locked(function):
    @wraps(function)
    def wrapped(location_id, scene_id, run_id, *args, **kwargs):
        scene = scene_path(location_id, scene_id)
        with run_session(scene / run_id):
            return function(location_id, scene_id, run_id, *args, **kwargs)
    return wrapped
