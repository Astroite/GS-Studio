import math
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
from .geometry import SH0


def validate(cloud):
    n = len(cloud["means"])
    for key, value in cloud.items():
        if len(value) != n or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} Gaussian attribute")
    if not n:
        raise ValueError("Gaussian cloud is empty")
    if np.any(np.linalg.norm(cloud["quats"], axis=-1) < 1e-10):
        raise ValueError("Zero Gaussian quaternion")
    return cloud


def write_ply(path, cloud):
    validate(cloud)
    n = len(cloud["means"])
    columns = {name: cloud["means"][:, i] for i, name in enumerate(("x","y","z"))}
    columns.update({name: np.zeros(n) for name in ("nx","ny","nz")})
    columns.update({f"f_dc_{i}": cloud["sh"][:,0,i] for i in range(3)})
    rest = cloud["sh"][:,1:,:].transpose(0,2,1).reshape(n,-1)
    columns.update({f"f_rest_{i}": rest[:,i] for i in range(rest.shape[1])})
    columns["opacity"] = cloud["opacities"].reshape(-1)
    columns.update({f"scale_{i}": cloud["scales"][:,i] for i in range(3)})
    q = cloud["quats"] / np.linalg.norm(cloud["quats"],axis=-1,keepdims=True)
    columns.update({f"rot_{i}": q[:,i] for i in range(4)})
    data = np.empty(n, dtype=[(name,"<f4") for name in columns])
    for name, values in columns.items():
        data[name] = values
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    PlyData([PlyElement.describe(data,"vertex")], text=False,
            comments=["Local Postshot coordinate_system RUB Y-up; SH 3DGS scalar-first quaternions"]).write(str(temp))
    temp.replace(path)


def read_ply(path):
    data = PlyData.read(str(path))["vertex"].data
    names = data.dtype.names
    required = ["x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    if any(k not in names for k in required):
        raise ValueError("Expected standard 3DGS PLY with SH, opacity, log scales and quaternion attributes")
    n = len(data)
    rest_names = sorted([k for k in names if k.startswith("f_rest_")], key=lambda k:int(k[7:]))
    k = len(rest_names)//3
    if len(rest_names) % 3 or round(math.sqrt(k+1)) ** 2 != k+1:
        raise ValueError("PLY SH properties do not form complete bands")
    sh = np.zeros((n,k+1,3),np.float32)
    sh[:,0] = np.stack([data[f"f_dc_{i}"] for i in range(3)],-1)
    if k:
        sh[:,1:] = np.stack([data[name] for name in rest_names],-1).reshape(n,3,k).transpose(0,2,1)
    return validate({"means":np.stack([data[a] for a in ("x","y","z")],-1),
                     "scales":np.stack([data[f"scale_{i}"] for i in range(3)],-1),
                     "quats":np.stack([data[f"rot_{i}"] for i in range(4)],-1),
                     "opacities":np.asarray(data["opacity"]).copy(), "sh":sh})


def _spz():
    try:
        import spz
        return spz
    except ImportError as exc:
        raise RuntimeError("Official Niantic SPZ Python binding is unavailable; run runtime setup") from exc


def read_spz(path):
    spz = _spz()
    options = spz.UnpackOptions()
    options.to_coord = spz.CoordinateSystem.RUB
    # Native narrow-string file streams do not support Chinese Windows paths.
    c = spz.load_spz_from_buffer(Path(path).read_bytes(), options)
    n = c.num_points
    degree = c.sh_degree
    if not n:
        raise ValueError("SPZ decoder returned an empty cloud")
    sh = np.zeros((n,(degree+1)**2,3),np.float32)
    sh[:,0] = np.asarray(c.colors).reshape(n,3)
    if degree:
        sh[:,1:] = np.asarray(c.sh).reshape(n,-1,3)
    # Official codec/PLY writer stores quaternion xyzw and colors as SH DC.
    # Python README labels colors RGB, but native source and interoperability
    # preflight establish that no SH0 conversion belongs here.
    q = np.asarray(c.rotations).reshape(n,4)[:,[3,0,1,2]]
    # The official codec decodes byte alpha 0/255 to logit +/-infinity.
    # Keep our train/render/export model finite while preserving alpha to <1e-6.
    opacities=np.nan_to_num(np.asarray(c.alphas).reshape(n),nan=np.nan,posinf=16.,neginf=-16.)
    return validate({"means":np.asarray(c.positions).reshape(n,3),
                     "scales":np.asarray(c.scales).reshape(n,3), "quats":q,
                     "opacities":opacities, "sh":sh})


def write_spz(path, cloud):
    validate(cloud)
    spz = _spz()
    c = spz.GaussianCloud()
    c.sh_degree = round(math.sqrt(cloud["sh"].shape[1])-1)
    c.antialiased = False
    for key, value in (("positions",cloud["means"]),("scales",cloud["scales"]),
                       ("rotations",cloud["quats"][:,[1,2,3,0]]),
                       ("alphas",cloud["opacities"]),("colors",cloud["sh"][:,0]),
                       ("sh",cloud["sh"][:,1:])):
        setattr(c,key,np.ascontiguousarray(value,dtype=np.float32).reshape(-1))
    options = spz.PackOptions()
    options.from_coord = spz.CoordinateSystem.RUB
    # v3 is compatible with current WebGL viewers; v4 requires a newer loader.
    options.version = 3
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    encoded = spz.save_spz_to_buffer(c, options)
    if not encoded:
        raise RuntimeError("Official SPZ encoder failed")
    temp.write_bytes(encoded)
    temp.replace(path)


def load_cloud(path):
    ext = Path(path).suffix.lower()
    if ext == ".ply":
        return read_ply(path)
    if ext == ".spz":
        return read_spz(path)
    raise ValueError("Only 3DGS PLY and SPZ models are supported")
