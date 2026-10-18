"""The real Run factory must distinguish measured media from generated views."""
from pathlib import Path
import pytest
from PIL import Image
from gsstudio.application import operations as ops
from gsstudio.infrastructure.persistence.run_repository import load_run
from gsstudio.infrastructure.adapters.realityscan import build_realityscan_align_command


def test_equirect_sequence_keeps_analytical_calibration(tmp_path):
    root=tmp_path/'catalog';root.mkdir()
    ops.create_location(root,'loc','Location');ops.create_scene(root,'loc','scene','Scene')
    images=tmp_path/'originals';images.mkdir()
    files=[]
    for i in range(3):
        p=images/f'{i}.jpg';Image.new('RGB',(128,64),(80+i,100,120)).save(p);files.append(p)
    ops.create_capture(root,'loc','scene','pano','equirect_sequence',files,fps=1,
                       output_width=128,output_height=64)
    run_id=ops.create_and_preprocess(root,'loc','scene','pano',keep_intermediates=True)
    run=load_run(root/'loc'/'scene',run_id)
    assert run.config.reconstruction.fix_intrinsics is True
    assert run.config.reconstruction.primary.panorama is not None


@pytest.mark.parametrize('fixed,prior',[(True,'2'),(False,'1')])
def test_realityscan_calibration_prior_reaches_actual_command(tmp_path,monkeypatch,fixed,prior):
    import gsstudio.infrastructure.adapters.realityscan as adapter
    params=tmp_path/'export.xml';params.write_text('<Configuration/>',encoding='utf-8')
    monkeypatch.setattr(adapter,'export_params_file',lambda:params)
    command=build_realityscan_align_command(tmp_path/'RealityScan.exe',tmp_path/'images',tmp_path/'crash',
        tmp_path/'registration.txt',tmp_path/'project.rsproj',projection_fov_degrees=84,fix_intrinsics=fixed)
    assert f'inpCalibration={prior}' in command
    assert f'inpCalibration={"1" if fixed else "2"}' not in command
    assert '-align' in command and '-exportRegistration' in command
