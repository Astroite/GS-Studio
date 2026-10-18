"""Versioned, reproducible segment QA; never rewrites historical trajectory reports."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gsstudio.infrastructure.persistence.manifests import canonical_hash
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.domain.models import SegmentQAConfig
from gsstudio.pipeline.reconstruction.core import projection_view_specs, projection_world_from_camera


def frame_id(name: str) -> int:
    match = re.search(r"frame_(\d+)", name)
    if not match:
        raise ValueError(f"Image lacks temporal frame ID: {name}")
    return int(match[1])


def image_name(name: str) -> str:
    name = name.replace('\\', '/').removeprefix('./').removeprefix('images/')
    if Path(name).is_absolute() or ':' in name or '..' in Path(name).parts:
        raise ValueError(f"Unsafe image path: {name}")
    return name


def analyze_segments(frames: list[dict], records: list[dict], included: set[str],
                     settings: SegmentQAConfig, view_rotations: list[np.ndarray]) -> dict:
    times = {i: float(r['timestamp_seconds']) for i, r in enumerate(records, 1)}
    ts = list(times.values())
    if not ts or not all(math.isfinite(t) for t in ts) or any(b <= a for a,b in zip(ts,ts[1:])):
        raise ValueError('Missing, duplicate, nonfinite or nonmonotonic timestamps')
    included = {image_name(n) for n in included}
    input_counts = Counter(frame_id(n) for n in included)
    if not set(input_counts) <= set(times):
        raise ValueError('Input inventory references unknown timestamps')
    groups = defaultdict(list)
    seen = set()
    for item in frames:
        name = image_name(item['file_path'])
        if name not in included or name in seen:
            raise ValueError(f'Duplicate or unexpected registered image: {name}')
        seen.add(name)
        f = frame_id(name)
        matrix = np.asarray(item['transform_matrix'], dtype=float)
        if matrix.shape != (4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0,0,0,1]):
            raise ValueError(f'Invalid/nonfinite pose: {name}')
        rotation = matrix[:3,:3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < 0.99:
            raise ValueError(f'Invalid pose rotation: {name}')
        view = re.search(r'view_(\d+)', name)
        if view:
            if int(view[1]) >= len(view_rotations):
                raise ValueError(f'Unknown projection view: {name}')
            # transforms.json uses OpenGL camera axes; remove the virtual lens rotation.
            rig_rotation = rotation @ np.diag([1.,-1.,-1.]) @ view_rotations[int(view[1])].T
        elif view_rotations:
            raise ValueError(f'Unknown projection view: {name}')
        else:
            # A perspective source registers each frame under its own name.
            rig_rotation = rotation
        groups[f].append((name, matrix[:3,3], rig_rotation))
    samples = []
    for f, poses in sorted(groups.items()):
        centers = np.array([p[1] for p in poses])
        center = np.median(centers, axis=0)
        spread = float(np.max(np.linalg.norm(centers-center, axis=1)))
        # Pick the observed rotation with minimum total angular discrepancy (medoid).
        rotations = [p[2] for p in poses]
        rotation = min(rotations, key=lambda r: sum(np.linalg.norm(r-s) for s in rotations))
        samples.append(dict(frame=f, timestamp_seconds=times[f], center=center.tolist(),
                            center_spread=spread, rotation=rotation.tolist(), images=sorted(p[0] for p in poses)))
    extent = float(np.linalg.norm(np.ptp([s['center'] for s in samples], axis=0))) if samples else 0.
    epsilon = max(extent * 1e-6, 1e-9)
    edges = []
    for a,b in zip(samples,samples[1:]):
        dt = b['timestamp_seconds']-a['timestamp_seconds']
        distance = float(np.linalg.norm(np.array(b['center'])-a['center']))
        edges.append(dict(from_frame=a['frame'], to_frame=b['frame'], dt=dt, distance=distance, speed=distance/dt))
    local = [e for e in edges if e['dt'] <= settings.max_gap_seconds]
    speeds = [e['speed'] for e in local]
    median_speed = float(np.median(speeds)) if speeds else 0.
    mad = float(np.median(np.abs(np.array(speeds)-median_speed))) if speeds else 0.
    speed_limit = max(median_speed*settings.speed_multiplier,
                      median_speed+settings.speed_multiplier*1.4826*mad, epsilon)
    nonzero_steps = [e['distance'] for e in local if e['distance'] > epsilon]
    step = float(np.median(nonzero_steps)) if nonzero_steps else epsilon
    spread_limit = max(step*settings.center_spread_step_ratio, epsilon)
    bad_samples = {s['frame'] for s in samples if s['center_spread'] > spread_limit}
    issues = []
    cut_before = set()
    for edge in edges:
        if edge['dt'] > settings.max_gap_seconds:
            issues.append(dict(type='coverage_gap', **edge))
            cut_before.add(edge['to_frame'])
        elif edge['speed'] > speed_limit:
            issues.append(dict(type='speed_jump', limit=speed_limit, **edge))
            cut_before.add(edge['to_frame'])
    for a,b in zip(samples,samples[1:]):
        if b['timestamp_seconds']-a['timestamp_seconds'] > settings.max_gap_seconds:
            continue
        relative = np.array(a['rotation']).T @ b['rotation']
        angle = math.degrees(math.acos(float(np.clip((np.trace(relative)-1)/2,-1,1))))
        if angle > 90:
            issues.append(dict(type='rotation_jump', from_frame=a['frame'], to_frame=b['frame'], degrees=angle))
            cut_before.add(b['frame'])
    for s in samples:
        if s['frame'] in bad_samples:
            issues.append(dict(type='center_spread', frame=s['frame'], spread=s['center_spread'], limit=spread_limit))
    chunks, chunk = [], []
    for s in samples:
        if s['frame'] in cut_before or s['frame'] in bad_samples:
            if chunk:
                chunks.append(chunk)
                chunk = []
        if s['frame'] not in bad_samples:
            chunk.append(s)
    if chunk:
        chunks.append(chunk)
    segments = []
    for index, chunk in enumerate(chunks, 1):
        first,last = chunk[0]['frame'],chunk[-1]['frame']
        names = sorted(n for s in chunk for n in s['images'])
        expected = sum(count for f,count in input_counts.items() if first <= f <= last)
        distinct = []
        for s in chunk:
            p = np.array(s['center'])
            if all(np.linalg.norm(p-q) > epsilon for q in distinct):
                distinct.append(p)
        reasons = []
        ratio = len(names)/expected if expected else 0.
        if len(chunk) < settings.minimum_samples:
            reasons.append('too_few_temporal_samples')
        if len(distinct) < settings.minimum_positions:
            reasons.append('too_few_camera_positions')
        if ratio < settings.registration_threshold:
            reasons.append('registration_below_threshold')
        segments.append(dict(id=f'segment-{index:03d}', status='passed' if not reasons else 'rejected',
            reasons=reasons, start_seconds=times[first], end_seconds=times[last],
            frames=[s['frame'] for s in chunk], images=names, input_images=expected,
            registration_ratio=ratio, distinct_positions=len(distinct)))
    missing = sorted(set(times)-set(groups))
    missing_intervals = []
    for f in missing:
        reason = 'all_views_filtered' if input_counts.get(f,0)==0 else 'no_registered_view'
        if missing_intervals and missing_intervals[-1]['frames'][-1]==f-1 and missing_intervals[-1]['reason']==reason:
            missing_intervals[-1]['frames'].append(f)
            missing_intervals[-1]['end_seconds']=times[f]
        else:
            missing_intervals.append(dict(frames=[f],start_seconds=times[f],end_seconds=times[f],reason=reason,
                boundary='head' if f==1 else 'interior'))
    if missing_intervals and missing_intervals[-1]['frames'][-1]==len(times):
        missing_intervals[-1]['boundary']='whole_route' if missing_intervals[-1]['frames'][0]==1 else 'tail'
    weak = sorted(f for f in times if input_counts.get(f,0) < settings.minimum_views)
    accepted = {f for segment in segments if segment['status']=='passed' for f in segment['frames']}
    usable = [s for s in segments if s['status']=='passed']
    coverage_complete = accepted == set(times) and not issues
    return dict(schema_version=1, integrity='passed', training_status='passed' if usable else 'failed',
        coverage_status='complete' if coverage_complete else 'incomplete',
        expected_samples=len(times), registered_samples=len(samples), missing_frames=missing,
        missing_intervals=missing_intervals,
        weak_input_frames=weak, excluded_frames=sorted(set(times)-accepted),
        median_speed=median_speed, speed_limit=speed_limit, center_spread_limit=spread_limit,
        issues=issues, segments=segments, samples=samples, config=settings.model_dump(mode='json'))


def segment_report(dataset: Path, records: list[dict], included: set[str], attempt,
                   settings: SegmentQAConfig | None = None) -> dict:
    transforms = dataset / 'transforms.json'
    payload = json.loads(transforms.read_text(encoding='utf-8'))
    if (dataset/'colmap-undistortion.json').is_file():
        from gsstudio.pipeline.reconstruction.colmap import media_root
        media_root(dataset)
    rotations = ([projection_world_from_camera(yaw, pitch)
                  for yaw, pitch in projection_view_specs(attempt)] if attempt.panorama else [])
    report = analyze_segments(payload['frames'], records, included, settings or SegmentQAConfig(), rotations)
    report['lineage'] = dict(transforms_sha256=sha256_file(transforms),
        records_sha256=canonical_hash(records), included_sha256=canonical_hash(sorted(included)),
        projection_sha256=canonical_hash(attempt),
        colmap_sha256={p.name:sha256_file(p) for p in [dataset/'colmap'/n for n in ('cameras.bin','images.bin','points3D.bin')] if p.is_file()},
        source_manifests={p.name:sha256_file(p) for p in [dataset/'mask-filter.json',dataset/'mask-final.json',
                          dataset/'colmap-undistortion.json'] if p.is_file()})
    return report


def write_segments(dataset: Path, records: list[dict], included: set[str], attempt,
                   settings: SegmentQAConfig, output: Path | None = None) -> dict:
    report = segment_report(dataset,records,included,attempt,settings)
    path = output or dataset/'segments.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)
    return report


def validate_segments(dataset: Path, records: list[dict], included: set[str], attempt,
                      settings: SegmentQAConfig, path: Path | None = None) -> dict:
    saved = json.loads((path or dataset/'segments.json').read_text(encoding='utf-8'))
    current = segment_report(dataset,records,included,attempt,settings)
    if saved != current:
        raise RuntimeError('Segment QA is stale or modified; regenerate the report')
    if saved['training_status'] != 'passed':
        raise RuntimeError('No segment passed training QA')
    return saved


def write_qa_report(scene: Path, run, resume=False) -> Path:
    """Report the three independent QA results and the training experiments."""
    from gsstudio.pipeline.masks.masking import validate_mask_filter
    from gsstudio.pipeline.masks.finalize import validate_mask_finalization
    from gsstudio.domain.run_state import begin_stage,complete_stage,fail_stage
    from gsstudio.infrastructure.persistence.run_repository import save_run
    if not run.selected_dataset or run.stages['reconstruct'].status.value!='succeeded':
        raise RuntimeError('Segment QA report requires a completed reconstruction')
    begin_stage(run,'qa',resume=resume,force=resume)
    save_run(scene,run)
    path=scene/'qa'/f'{run.id}.md'
    try:
        dataset=scene/run.selected_dataset
        label=run.metrics['selected_attempt']
        records=[json.loads(line) for line in (scene/run.id/f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
        filtered=validate_mask_filter(dataset,verify_hashes=True)
        included={r['image'] for r in filtered['accepted']}
        final=validate_mask_finalization(dataset,included)
        included-=set(final.get('excluded_images',[]))
        report=validate_segments(dataset,records,included,run.config.reconstruction.primary,run.config.segment_qa)
        result={k:report[k] for k in ('integrity','training_status','coverage_status')}
        run.metrics.setdefault('qa',{})['segments']=result
        lines=[f'# QA: {run.id}','',f'配置哈希：`{run.config_hash}`',
            f'选中尝试：`{label}`；刚性 rig：未启用。',
            f'遮罩剔除：严格大于 {run.config.masking.mask_discard_threshold:.2%}，等于时保留。','',
            '| 检查 | 结果 |','| --- | --- |',
            f'| 数据完整性 | {report["integrity"]} |',
            f'| 分段训练适用性 | {report["training_status"]} |',
            f'| 整条路线覆盖 | {report["coverage_status"]} |','',
            '分段通过仅允许训练对应图片；覆盖不完整不等于整条路线验收通过。','',
            '| 分段 | 起止秒 | 图片 | 区间注册率 | 结果 | 排除原因 |',
            '| --- | --- | ---: | ---: | --- | --- |']
        for segment in report['segments']:
            lines.append(f'| {segment["id"]} | {segment["start_seconds"]:.3f}–{segment["end_seconds"]:.3f} | '
                f'{len(segment["images"])} | {segment["registration_ratio"]:.2%} | {segment["status"]} | {", ".join(segment["reasons"])} |')
        counts=Counter(issue['type'] for issue in report['issues'])
        lines+=['',f'注册采样：{report["registered_samples"]}/{report["expected_samples"]}。',
            f'缺失采样：{len(report["missing_frames"])}；异常/覆盖边界：`{json.dumps(counts)}`。',
            f'完整图片清单、缺失原因及来源哈希：[segments.json]({(dataset/"segments.json").resolve().as_posix()})','',
            '## 训练实验','', '| 后端 | 分段 | 光度补偿 | 状态 | 记录 |','| --- | --- | --- | --- | --- |']
        for item in run.metrics.get('training_experiments',[]):
            target=Path(item['output'])/'dispatch.json'
            lines.append(f'| {item["backend"]} | {item["segment"]} | {item.get("photo_comp","unknown")} | {item["status"]} | [dispatch]({target.resolve().as_posix()}) |')
        lines+=['','正式候选保留 SH 3。人物残留、异常高亮、细节和统一验证相机的渲染仍需人工对照；准备完成或短训练测试不代表画质验收。','']
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.md.tmp')
        temporary.write_text('\n'.join(lines),encoding='utf-8')
        temporary.replace(path)
        complete_stage(run,'qa',message=f'Segment report generated; route coverage {report["coverage_status"]}')
    except Exception as error:
        fail_stage(run,'qa',str(error))
        save_run(scene,run)
        raise
    save_run(scene,run)
    return path
