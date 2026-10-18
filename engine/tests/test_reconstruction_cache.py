from pathlib import Path
import struct
import numpy as np
from local_postshot.reconstruction import artifact_manifest,reconstruction_artifacts_valid,read_sparse


def test_deleted_or_modified_required_artifact_invalidates_reuse(tmp_path):
    work=tmp_path/"work";dense=work/"undistorted"
    (dense/"sparse").mkdir(parents=True);(dense/"images").mkdir()
    cameras=[{"name":"one.png"}]
    paths=[work/"sparse_points.npz",work/"cameras.json",dense/"images"/"one.png"]
    paths += [dense/"sparse"/n for n in ("cameras.bin","images.bin","points3D.bin")]
    for f in paths:f.write_bytes(b"initial")
    p={"reconstruction":{"cameras":cameras,"artifacts":artifact_manifest(work,dense,cameras)}}
    assert reconstruction_artifacts_valid(p,work,dense)
    (work/"sparse_points.npz").unlink()
    assert not reconstruction_artifacts_valid(p,work,dense)
    (work/"sparse_points.npz").write_bytes(b"changed")
    assert not reconstruction_artifacts_valid(p,work,dense)
    (work/"sparse_points.npz").write_bytes(b"initial")
    (dense/"images"/"one.png").unlink()
    assert not reconstruction_artifacts_valid(p,work,dense)


def test_binary_camera_adapter_preserves_pinhole_projection(tmp_path):
    def pack(fmt,*values):return struct.pack("<"+fmt,*values)
    (tmp_path/"cameras.bin").write_bytes(pack("QiiQQdddd",1,5,1,64,48,40.,41.,32.,24.))
    image=pack("Qidddddddi",1,7,1.,0.,0.,0.,0.,0.,0.,5)+b"one.png\0"+pack("Q",0)
    (tmp_path/"images.bin").write_bytes(image)
    (tmp_path/"points3D.bin").write_bytes(pack("QQdddBBBdQ",1,13,1.,1.,2.,10,20,30,.1,0))
    cameras,points,colors=read_sparse(tmp_path)
    np.testing.assert_allclose(points,[[1,-1,-2]])
    c=cameras[0];cv=np.asarray(c["w2c"])@np.r_[points[0],1]
    uv=np.asarray(c["K"])@cv[:3];uv=uv[:2]/uv[2]
    np.testing.assert_allclose(uv,[52,44.5])
    np.testing.assert_allclose(colors,[[10/255,20/255,30/255]])


def test_screen_clean_rule_matches_max_axis_normalized_image_edge():
    from local_postshot.cleaning import radius_exceeds
    radii=np.array([[0,0],[150,50],[151,50],[10,200]])
    assert radius_exceeds(radii,1000,500,.15).tolist()==[False,False,True,True]
