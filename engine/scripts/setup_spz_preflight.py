"""Verify native official SPZ encoding and PLY color/quaternion conventions."""
import json
import os
from pathlib import Path
import numpy as np
from plyfile import PlyData
import spz

root = Path(os.environ.get("LOCAL_POSTSHOT_ROOT", "E:/AI/LocalPostshot")) / "evidence/runtime"
root.mkdir(parents=True, exist_ok=True)
cloud = spz.GaussianCloud()
cloud.positions = np.array([1., 2., 3., -1., 0., 1.], np.float32)
cloud.scales = np.array([-3.] * 6, np.float32)
cloud.rotations = np.array([0., 0., 0., 1.] * 2, np.float32)
cloud.alphas = np.array([0., 2.], np.float32)
cloud.colors = np.array([-0.5, 0., 0.7, 0.2, -0.2, 0.3], np.float32)
cloud.sh_degree = 0
cloud.sh = np.empty(0, np.float32)
pack = spz.PackOptions()
pack.from_coord = spz.CoordinateSystem.RUB
unpack = spz.UnpackOptions()
unpack.to_coord = spz.CoordinateSystem.RUB
assert spz.save_spz(cloud, pack, str(root / "native-roundtrip.spz"))
assert spz.save_splat_to_ply(cloud, pack, str(root / "official-output.ply"))
decoded = spz.load_spz(str(root / "native-roundtrip.spz"), unpack)
ply = PlyData.read(root / "official-output.ply")["vertex"]
ply_dc = np.column_stack([ply[f"f_dc_{i}"] for i in range(3)]).reshape(-1)
np.testing.assert_allclose(ply_dc, cloud.colors, atol=1e-6)
np.testing.assert_allclose(decoded.positions, cloud.positions, atol=0.001)
np.testing.assert_allclose(decoded.colors, cloud.colors, atol=0.02)
result = {
    "success": True,
    "module": spz.__file__,
    "source_commit": "affd0ecea7fbb4c265ee119475af7ee5b2997482",
    "source_version": "1.1.0",
    "count": len(decoded),
    "pack_version": pack.version,
    "color_semantics": "SH DC coefficients, not display RGB",
    "cloud_colors": cloud.colors.tolist(),
    "official_ply_f_dc": ply_dc.tolist(),
    "decoded_colors": decoded.colors.tolist(),
    "max_color_error": float(np.max(np.abs(decoded.colors - cloud.colors))),
}
result["version_roundtrips"] = {}
for version in (2, 3, 4):
    pack.version = version
    path = root / f"native-v{version}.spz"
    assert spz.save_spz(cloud, pack, str(path))
    restored = spz.load_spz(str(path), unpack)
    np.testing.assert_allclose(restored.positions, cloud.positions, atol=0.001)
    np.testing.assert_allclose(restored.colors, cloud.colors, atol=0.02)
    result["version_roundtrips"][str(version)] = {"success": True, "bytes": path.stat().st_size, "count": len(restored)}
(root / "spz-preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
