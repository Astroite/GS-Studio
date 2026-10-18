"""Incrementally record local tile/AA wheels without touching other runtimes."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

REPO=Path(__file__).resolve().parents[1]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def record_round4_native_runtime(root):
    root=Path(root).resolve()
    Path(sys.prefix).resolve().relative_to(root/'runtime')
    manifest=root/'provisioning-manifest.json'
    lock=root/'runtime-requirements.lock.txt'
    backup=root/'evidence/runtime'/('round4-before-'+uuid.uuid4().hex)
    backup.mkdir(parents=True)
    for p in (manifest,lock):
        if p.is_file():shutil.copyfile(p,backup/p.name)
    data=json.loads(manifest.read_text(encoding='utf-8-sig'))
    records={}
    for suffix,implementation in [('tiles','tile_support_v1'),('projection','native_floor_v1')]:
        module='local_splat3_'+suffix
        dist=importlib.metadata.distribution('local-splat3-'+suffix)
        files=[Path(dist.locate_file(p)) for p in dist.files if str(p).endswith('.pyd')]
        if len(files)!=1:raise RuntimeError('Expected one binary for '+module)
        binary=files[0].resolve();binary.relative_to(root/'runtime')
        wheel=root/'wheels'/f'{module}-{dist.version}-cp310-cp310-win_amd64.whl'
        if not wheel.is_file():raise FileNotFoundError(wheel)
        src=REPO/'engine/native'/('splat3_'+suffix)
        sources=[p for p in src.rglob('*') if p.is_file() and (p.suffix in ('.py','.h','.cuh','.cu','.cpp') or p.name in ('LICENSE','NOTICE','GLM-LICENSE.txt'))]
        record={'version':dist.version,'implementation':implementation,'binary_sha256':digest(binary),
            'wheel_sha256':digest(wheel),'cuda':'12.9','cuda_arch':'8.9','nvcc_flags':['-O3'],
            'source':src.relative_to(REPO).as_posix(),'original_postshot_code':False,
            'source_sha256':{p.relative_to(REPO).as_posix():digest(p) for p in sources},
            'validation_scope':'Current binary identity; test evidence is bound separately in round4-delivery.json'}
        data[module]=records[module]=record
        data.setdefault('versions',{})['local-splat3-'+suffix]=dist.version
        data.setdefault('artifacts',{})[wheel.relative_to(root).as_posix()]=digest(wheel)
    # Preserve every unrelated pin and all dataset/SPZ provenance verbatim.
    lines=lock.read_text(encoding='utf-8-sig').splitlines() if lock.exists() else []
    lines=[s for s in lines if not s.lower().replace('_','-').startswith(('local-splat3-tiles==','local-splat3-projection=='))]
    lines += [f'{name.replace("_","-")}=={value["version"]}' for name,value in records.items()]
    lock.write_text('\n'.join(sorted(lines,key=str.casefold))+'\n',encoding='utf-8')
    manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    receipt={'passed':True,'backup':str(backup),'modules':records,
             'unrelated_manifest_fields_unchanged':all(data.get(k)==v for k,v in json.loads((backup/manifest.name).read_text(encoding='utf-8-sig')).items() if k not in ('versions','artifacts',*records))}
    (REPO/'evidence/splat3/round4-runtime.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return records

if __name__=='__main__':
    records=record_round4_native_runtime(Path(os.environ.get('LOCAL_POSTSHOT_ROOT','E:/AI/LocalPostshot')))
    print(json.dumps({k:{f:v[f] for f in ('version','binary_sha256','wheel_sha256')} for k,v in records.items()}))
