"""Explicit opt-in only; parent owns the shared CUDA slot."""
import os
import sys
from pathlib import Path
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('LOCAL_POSTSHOT_GPU_TESTS')!='1',reason='Parent-scheduled real CUDA test')


def test_full_metadata_offaxis_occlusion_threshold_and_cancellation(tmp_path):
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from studio_selection_cases import run_cases
    assert len(run_cases(tmp_path/'roi-evidence'))==10
