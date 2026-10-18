import math
import numpy as np
import pytest
from local_postshot.formats import read_ply, write_ply, read_spz, write_spz
from local_postshot.geometry import apply_edits, euler_matrix, rotate_sh, sh_basis, quat_matrix


def cloud(n=20,degree=3):
    r = np.random.default_rng(7)
    q = r.normal(size=(n,4)); q/=np.linalg.norm(q,axis=-1,keepdims=True)
    return {"means":r.normal(size=(n,3)).astype(np.float32),"scales":r.uniform(-5,-2,(n,3)).astype(np.float32),
            "quats":q.astype(np.float32),"opacities":r.uniform(-2,2,n).astype(np.float32),
            "sh":r.uniform(-.2,.2,(n,(degree+1)**2,3)).astype(np.float32)}


def test_ply_roundtrip_preserves_all_sh_properties(tmp_path):
    c=cloud(); file=tmp_path/"unicode_测试.ply"
    write_ply(file,c); d=read_ply(file)
    for k in c:
        np.testing.assert_allclose(d[k],c[k],atol=2e-7)


@pytest.mark.parametrize("degree",[0,1,2,3,4])
def test_sh_rotation_preserves_directional_radiance(degree):
    c=cloud(degree=degree)
    r=euler_matrix([.27,-.81,1.41])
    rotated=rotate_sh(c["sh"],r)
    d=np.random.default_rng(9).normal(size=(500,3))
    expected=np.einsum("dk,nkc->ndc",sh_basis(d@r,degree),c["sh"])
    actual=np.einsum("dk,nkc->ndc",sh_basis(d,degree),rotated)
    np.testing.assert_allclose(actual,expected,atol=1e-7)


def test_real_sh_basis_matches_gsplat_degree_one():
    b=sh_basis(np.eye(3),1)
    c=math.sqrt(3/(4*math.pi))
    np.testing.assert_allclose(b[:,1:],[[0,0,-c],[-c,0,0],[0,c,0]],atol=1e-12)


def test_crop_before_transform_and_gaussian_covariance():
    c=cloud(n=20)
    c["means"][:]=np.arange(20)[:,None]
    t={"position":[3,-4,2],"rotation":[.2,-.3,.7],"scale":2.5}
    crop={"enabled":True,"min":[2,2,2],"max":[4,4,4],"invert":False}
    d=apply_edits(c,t,crop)
    assert len(d["means"])==3
    r=euler_matrix(t["rotation"])
    np.testing.assert_allclose(d["means"],c["means"][2:5]@r.T*2.5+[3,-4,2],atol=1e-6)
    def cov(q,s):
        rot=quat_matrix(q)
        return (rot*np.exp(2*s)[:,None,:])@rot.transpose(0,2,1)
    np.testing.assert_allclose(cov(d["quats"],d["scales"]),
                               r@cov(c["quats"][2:5],c["scales"][2:5])@r.T*2.5**2,atol=1e-7)


def test_spz_official_roundtrip_quantization_bounds(tmp_path):
    pytest.importorskip("spz")
    c=cloud(degree=3)
    file=tmp_path/"场景_编码往返.spz"
    write_spz(file,c); d=read_spz(file)
    assert len(c["means"])==len(d["means"])
    np.testing.assert_allclose(d["means"],c["means"],atol=1/4096)
    np.testing.assert_allclose(d["scales"],c["scales"],atol=.04)
    np.testing.assert_allclose(d["sh"][:,0],c["sh"][:,0],atol=.025)
    np.testing.assert_allclose(d["sh"][:,1:],c["sh"][:,1:],atol=.15)
    np.testing.assert_allclose(quat_matrix(d["quats"]),quat_matrix(c["quats"]),atol=.012)


def test_spz_saturated_alpha_decode_is_finite(tmp_path):
    pytest.importorskip("spz")
    c=cloud(n=2);c["opacities"][:]=[-100.,100.]
    file=tmp_path/"opaque.spz";write_spz(file,c);d=read_spz(file)
    assert np.isfinite(d["opacities"]).all()
    np.testing.assert_allclose(1/(1+np.exp(-d["opacities"])),[0,1],atol=1e-6)
