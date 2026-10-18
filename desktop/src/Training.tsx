import { useEffect, useRef, useState } from 'react';
import { modelUrl } from './engine';
import { Field, Check, Json, displayStatus, experimentLabel } from './Workflow';
import { LatestRequest, previewMatches } from './studioState';
import type { StudioWork } from './useStudio';
import { record, readable } from './types';
import type { Metric } from './types';
import { LossChart } from './ActivityPanel';
import { verifiedPreviewBytes } from './previewFiles';
import { checkedTrainingOptions, newTrainingSettings, savedTrainingSettings, trainingOptionsIdentity } from './trainingOptions';

export function LinkedImages({ source, render, version = 0 }: { source: string; render: string; version?: number }) {
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 }), [failed, setFailed] = useState('');
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  useEffect(() => { setFailed(''); }, [source, render]);
  return <div><div className="gs-actions"><span>滚轮缩放 · 拖动平移 · 两侧联动</span><button className="button compact" onClick={() => setView({ zoom: 1, x: 0, y: 0 })}>重置视图</button></div>{failed && <p role="alert">{failed}</p>}<div className="gs-paired">{[source, render].map((path, i) => <div key={i} className="gs-paired-pane" onWheel={e => setView(v => ({ ...v, zoom: Math.max(0.1, Math.min(16, v.zoom * Math.exp(-e.deltaY * 0.001))) }))} onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); drag.current = { x: e.clientX, y: e.clientY, px: view.x, py: view.y }; }} onPointerMove={e => { const d = drag.current; if (d) setView(v => ({ ...v, x: d.px + e.clientX - d.x, y: d.py + e.clientY - d.y })); }} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}><span>{i ? '原生训练渲染' : '输入图像'}</span>{path ? <img draggable={false} src={path.startsWith('blob:') ? path : modelUrl(path, version)} style={{ transform: `translate(${view.x}px,${view.y}px) scale(${view.zoom})` }} onError={() => setFailed('图像无法读取；请刷新当前相机预览')} alt={i ? '匹配当前请求的原生渲染' : '匹配当前请求的源图'} /> : <p>当前相机尚无匹配的预览</p>}</div>)}</div></div>;
}
export function CameraInspection({ cameras }: { cameras: Record<string, unknown>[] }) {
  const [index, setIndex] = useState(0), row = cameras[index] ?? {};
  useEffect(() => setIndex(0), [cameras]);
  const points = Array.isArray(row.sparse_points) ? row.sparse_points : [];
  return <section className="gs-card"><h2>输入相机与稀疏点</h2><Field label="相机"><select value={index} onChange={e => setIndex(Number(e.target.value))}>{cameras.map((c, i) => <option value={i} key={String(c.image ?? c.source_image ?? i)}>{String(c.source_image ?? c.image)} · {String(c.split ?? '')} · 组 {String(c.group ?? '')}</option>)}</select></Field>{typeof row.source_path === 'string' && <svg className="gs-camera-image" viewBox={`0 0 ${row.width} ${row.height}`}><image href={modelUrl(row.source_path)} width={Number(row.width)} height={Number(row.height)} />{points.slice(0, 5000).map((p, i) => Array.isArray(p) && <circle key={i} cx={Number(p[0])} cy={Number(p[1])} r={Number(row.width) / 500} fill="#55d1b6" />)}</svg>}<details><summary>实际相机参数 / split</summary><Json value={row} /></details></section>;
}
export function TrainingPreview({ studio, output, digest, loadFiles = loadPreviewFiles }: { studio: StudioWork; output: string; digest: string; loadFiles?: (p: Record<string, unknown>) => Promise<[string, string]> }) {
  const [cameras, setCameras] = useState<Record<string, unknown>[]>([]), [image, setImage] = useState(''), [pair, setPair] = useState<Record<string, unknown> | null>(null), [message, setMessage] = useState('选择输入相机');
  const [offset, setOffset] = useState(0), [total, setTotal] = useState(0);
  const requests = useRef(new LatestRequest());
  useEffect(() => setOffset(0), [output, digest]);
  useEffect(() => {
    let live = true; setCameras([]); setImage(''); setPair(null); requests.current.invalidate();
    if (output && digest) void studio.read('sources.list', { output, package_sha256: digest, offset, limit: 200 }).then(value => { if (!live) return; const next = Array.isArray(value.cameras) ? value.cameras.map(record) : []; setCameras(next); setTotal(Number(value.total ?? next.length)); }).catch(e => { if (live) setMessage(readable(e)); });
    return () => { live = false; requests.current.invalidate(); };
  }, [output, digest, offset]);
  useEffect(() => {
    setPair(null); if (!image || !output || !digest) return;
    const token = requests.current.next(), request_id = crypto.randomUUID().replaceAll('-', ''), expected = { request_id, package_sha256: digest, image };
    let stopped = false, polling = false;
    let objectUrls: string[] = [], shown = '';
    setMessage('正在准备当前相机的原生预览…');
    const poll = async () => {
      if (polling || stopped) return; polling = true;
      try {
        const value = await studio.read('preview.read', { output, request_id });
        const preview = record(value.preview);
        if (requests.current.current(token) && previewMatches(preview, expected)) {
          const stamp = `${preview.request_id}:${preview.step}:${preview.source_sha256}:${preview.render_sha256}`;
          if (shown !== stamp) {
            const urls = await loadFiles(preview);
            if (stopped || !requests.current.current(token)) { urls.filter(v => v.startsWith('blob:')).forEach(v => URL.revokeObjectURL(v)); return; }
            objectUrls.forEach(url => URL.revokeObjectURL(url)); objectUrls = urls.filter(v => v.startsWith('blob:')); shown = stamp;
            setPair({ ...preview, source_path: urls[0], render_path: urls[1] }); setMessage(`同视角 · step ${preview.step}`);
          }
        }
      } catch (e) { if (!stopped) setMessage(readable(e)); }
      finally { polling = false; }
    };
    void studio.read('preview.request', { output, ...expected }).then(() => { if (!stopped) void poll(); }).catch(e => { if (!stopped) setMessage(readable(e)); });
    const timer = window.setInterval(() => void poll(), 1500);
    return () => { stopped = true; requests.current.invalidate(); window.clearInterval(timer); objectUrls.forEach(url => URL.revokeObjectURL(url)); };
  }, [output, digest, image]);
  return <section className="gs-card"><h2>源图 / 原生渲染对照</h2><div className="gs-actions"><button className="button compact" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 200))}>上一页相机</button><span>{offset + 1}–{Math.min(offset + 200, total)} / {total}</span><button className="button compact" disabled={offset + 200 >= total} onClick={() => setOffset(offset + 200)}>下一页相机</button></div><Field label="训练包内相机"><select value={image} onChange={e => setImage(e.target.value)}><option value="">选择相机</option>{cameras.map((c, i) => <option key={i} value={String(c.image)}>{String(c.split)} · {String(c.source_image ?? c.image)}</option>)}</select></Field><p role="status">{message}</p><LinkedImages key={`${output}:${image}`} source={String(pair?.source_path ?? '')} render={String(pair?.render_path ?? '')} version={Number(pair?.step ?? 0)} /><details><summary>当前相机参数</summary><Json value={cameras.find(v => v.image === image) ?? {}} /></details></section>;
}
export function TrainingSettings({ value, change, disabled, resolved }: { value: Record<string, unknown>; change: (v: Record<string, unknown>) => void; disabled: boolean; resolved: Record<string, unknown> | null }) {
  const update = (patch: Record<string, unknown>) => change({ ...value, ...patch });
  const photoMode = value.use_bilateral_grid === undefined ? 'auto' : value.photo_comp ? value.use_bilateral_grid ? 'bilateral' : 'camera' : 'off';
  return <fieldset disabled={disabled}><div className="gs-grid two">
    <Field label="训练预设"><select value={String(value.training_preset)} onChange={e => update({ training_preset: e.target.value })}><option value="auto">自动 · 按素材来源</option><option value="photo">普通照片 · 10k / 最大边 1600</option><option value="gs">GS · 原分辨率 / 正式预算</option></select></Field>
    <Field label="原生配方"><select value={String(value.training_recipe)} onChange={e => update({ training_recipe: e.target.value })}><option value="quality_v2">公开基线 v2 · 默认</option><option value="observed_v5">投影合成 v5 · 已发布</option><option value="observed_v6">椭圆裁剪 v6 · 实验</option></select></Field>
    <Field label="步数 · 留空使用预设预算"><input type="number" min={1} max={10000000} step={1} value={value.steps === undefined ? '' : String(value.steps)} onChange={e => update({ steps: e.target.value ? Number(e.target.value) : undefined })} /></Field>
    <Field label="光度补偿方式"><select value={photoMode} onChange={e => update(e.target.value === 'auto' ? { photo_comp: undefined, use_bilateral_grid: undefined } : { photo_comp: e.target.value !== 'off', use_bilateral_grid: e.target.value === 'bilateral' })}><option value="auto">使用预设默认值</option><option value="bilateral">双边网格（不叠加相机补偿）</option><option value="camera">仅相机补偿</option><option value="off">关闭</option></select></Field>
  </div><Check label="启用已验证的稀疏深度锚" checked={resolved?.use_sparse_depth === true} disabled={!resolved?.sparse_depth_available} change={v => update({ use_sparse_depth: v })} /><button type="button" className="button compact" disabled={value.use_sparse_depth === undefined} onClick={() => update({ use_sparse_depth: undefined })}>深度恢复预设默认值</button>
    <p className="gs-note">深度{value.use_sparse_depth === undefined ? '沿用预设' : '使用显式设置'} · {resolved ? resolved.sparse_depth_reason == null ? `已验证 ${resolved.sparse_depth_observations} 个训练观测` : String(resolved.sparse_depth_reason) : '等待当前训练包验证'}</p>
    {value.training_recipe === 'observed_v5' && <p className="gs-note">v5 为已发布配方，使用独立合成器与冻结投影配置；旧检查点保留原配方。</p>}{value.training_recipe === 'observed_v6' && <p className="gs-warning">v6 尚未通过 R13 画质门槛，属于实验配置。</p>}<p className="gs-note">恢复沿用已保存配置，修改设置须新建实验。</p></fieldset>;
}
export function Training({ studio, openModel, focus = {} }: { studio: StudioWork; openModel: (value: Record<string, unknown>) => Promise<void>; focus?: { segment?: string; output?: string } }) {
  const [choices, setChoices] = useState<Record<string, unknown>>({}), [segment, setSegment] = useState(focus.segment ?? ''), [settings, setSettings] = useState<Record<string, unknown>>(newTrainingSettings);
  const [experiments, setExperiments] = useState<Record<string, unknown>[]>([]), [output, setOutput] = useState(focus.output ?? ''), [snapshot, setSnapshot] = useState<{ output: string; value: Record<string, unknown> } | null>(null), [pendingControl, setPendingControl] = useState('');
  const [restart, setRestart] = useState(false);
  const [openingModel, setOpeningModel] = useState(false), openingModelRef = useRef(false);
  const [cameraInspection, setCameraInspection] = useState<Record<string, unknown>[]>([]);
  const reloadRequests = useRef(new LatestRequest());
  const x = studio.context, external = !!x.import_id;
  const viewKey = JSON.stringify([x.location_id, x.scene_id, x.run_id, x.import_id, focus.segment, focus.output]);
  const [newExperiment, setNewExperiment] = useState<string | null>(null), creating = newExperiment === viewKey;
  const state = snapshot?.output === output ? snapshot.value : {};
  const selectOutput = (next: string) => { setOutput(next); setSnapshot(null); setNewExperiment(null); setPendingControl(''); setRestart(false); setCameraInspection([]); };
  const [options, setOptions] = useState<{ key: string; value: Record<string, unknown> } | null>(null), [optionsError, setOptionsError] = useState(''), [checking, setChecking] = useState(false), [optionsRevision, setOptionsRevision] = useState(0);
  const checkingRef = useRef(false);
  const target = external ? { root: studio.catalog.root, import_id: x.import_id, location_id: undefined, scene_id: undefined, run_id: undefined, capture_id: undefined, segment: undefined } : { root: studio.catalog.root, ...x, import_id: undefined, segment };
  const source = external ? studio.catalog.external_imports.find(v => v.id === x.import_id) : studio.catalog.captures.find(v => v.id === x.capture_id && v.location_id === x.location_id && v.scene_id === x.scene_id);
  const run = studio.catalog.runs.find(v => v.id === x.run_id && v.location_id === x.location_id && v.scene_id === x.scene_id);
  const optionsKey = JSON.stringify([viewKey, creating, output, target, settings, source?.manifest_sha256, source?.package_sha256, run?.config_hash, run?.selected_dataset, optionsRevision]);
  const currentOptionsKey = useRef(optionsKey); currentOptionsKey.current = optionsKey;
  const mounted = useRef(true), currentOutput = useRef(output); currentOutput.current = output;
  const optionsQueue = useRef<Promise<unknown>>(Promise.resolve());
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const openSelectedModel = async () => {
    if (openingModelRef.current || !output) return;
    const selected = output, key = viewKey;
    openingModelRef.current = true; setOpeningModel(true);
    try {
      const value = await studio.read('model_open', { output: selected });
      if (mounted.current && currentOutput.current === selected && currentViewKey.current === key) await openModel(value);
    } catch (e) { if (mounted.current && currentOutput.current === selected) studio.report(e); }
    finally { openingModelRef.current = false; if (mounted.current) setOpeningModel(false); }
  };
  const currentViewKey = useRef(viewKey); currentViewKey.current = viewKey;
  const resolved = options?.key === optionsKey ? options.value : null;
  const hasTarget = external || !!(x.location_id && x.scene_id && x.run_id && segment);
  const queueOptions = (request: Record<string, unknown>, key: string) => {
    const next = optionsQueue.current.catch(() => {}).then(async () => {
      if (!mounted.current || currentOptionsKey.current !== key) return null;
      return checkedTrainingOptions(await studio.read('training_options', request), request);
    });
    // training_options can create CPU packages and occupies the worker's write
    // slot. Coalesce superseded choices before sending another request.
    optionsQueue.current = next;
    return next;
  };
  useEffect(() => {
    let live = true; setOptions(null); setOptionsError('');
    if (creating && hasTarget) void queueOptions({ ...target, ...settings }, optionsKey).then(v => {
      if (v && live && currentOptionsKey.current === optionsKey) setOptions({ key: optionsKey, value: v });
    }).catch(e => { if (live && currentOptionsKey.current === optionsKey) setOptionsError(readable(e)); });
    return () => { live = false; };
  }, [optionsKey, hasTarget, creating]);
  const reload = async () => {
    const token = reloadRequests.current.next();
    try {
      if (external) { const result = await studio.read('list_external_imports'); if (!reloadRequests.current.current(token)) return; const item = (Array.isArray(result.imports) ? result.imports.map(record) : []).find(v => v.id === x.import_id); setExperiments(Array.isArray(item?.experiments) ? item.experiments.map(record) : []); }
      else if (x.run_id) { const [choice, list] = await Promise.all([studio.read('segment_choices'), studio.read('experiments')]); if (!reloadRequests.current.current(token)) return; setChoices(choice); setExperiments(Array.isArray(list.experiments) ? list.experiments.map(record) : []); }
    } catch (e) { studio.report(e); }
  };
  useEffect(() => { setChoices({}); setExperiments([]); selectOutput(focus.output ?? ''); setSegment(focus.segment ?? ''); void reload(); return () => reloadRequests.current.invalidate(); }, [viewKey]);
  const resultOutput = studio.result.output ?? record(studio.result.detail).output;
  const lastResultOutput = useRef(resultOutput);
  useEffect(() => {
    if (resultOutput !== lastResultOutput.current && typeof resultOutput === 'string') selectOutput(resultOutput);
    lastResultOutput.current = resultOutput;
  }, [resultOutput]);
  useEffect(() => {
    let stopped = false, running = false; setSnapshot(null); setPendingControl('');
    const poll = async () => {
      if (running || !output) return; running = true;
      try { const value = await studio.read('training_state', { output }); if (!stopped) { setSnapshot({ output, value }); const control = record(value.state); if (control.request_id) setPendingControl(old => old === control.request_id ? '' : old); } }
      catch (e) { if (!stopped) studio.report(e); } finally { running = false; }
    };
    void poll(); const timer = window.setInterval(() => void poll(), 2000);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [output]);
  const dispatch = record(state.dispatch), controlState = record(state.state), training = record(state.training);
  const digest = String(dispatch.package_sha256 ?? training.package_sha256 ?? ''), generation = String(controlState.generation_id ?? ''), controlStatus = String(controlState.status ?? training.status ?? 'unknown');
  const active = ['running', 'waiting_gpu', 'suspended', 'training'].includes(controlStatus) && !!generation && !!digest;
  const sendControl = async (control_action: string) => { const selected = output; const result = await studio.control({ output, control_action, package_sha256: digest, generation_id: generation }); if (result && currentOutput.current === selected) setPendingControl(String(result.request_id)); };
  const start = async (resume: boolean) => {
    if (checkingRef.current) return;
    if (resume && !training.checkpoint_sha256 && !restart) { studio.report(new Error('首次检查点尚不存在；请明确选择从第 0 步重试')); return; }
    if (!resume && (!creating || !resolved)) return;
    const requestKey = optionsKey, saved = experiments.find(v => v.output === output);
    if (resume && (!saved || !external && !saved.segment)) { studio.report(new Error('请先选择属于当前项目的已记录实验，以核验恢复身份')); return; }
    const submitSettings = resume ? savedTrainingSettings(dispatch, record(state.config)) : settings;
    const submitTarget = { ...target, ...(external ? {} : { segment: resume ? saved?.segment ?? segment : segment }) };
    checkingRef.current = true; setChecking(true);
    try {
      const fresh = await queueOptions({ ...submitTarget, ...submitSettings }, requestKey);
      if (!fresh || !mounted.current || currentOptionsKey.current !== requestKey || resume && currentOutput.current !== output) return;
      if (!resume && resolved && trainingOptionsIdentity(fresh) !== trainingOptionsIdentity(resolved)) {
        setOptions({ key: requestKey, value: fresh }); studio.report(new Error('训练包或预算已变化，请检查更新后的设置再提交')); return;
      }
      const value = await studio.operate(external ? 'train_external_import' : 'train_segment', { ...submitTarget, ...submitSettings, ...(external ? { experiment_id: resume ? saved?.id : undefined } : { output: resume ? output : undefined }), resume, restart_from_step_zero: resume && restart });
      if (!mounted.current || currentOptionsKey.current !== requestKey) return;
      if (value?.output) selectOutput(String(value.output)); await reload();
    } catch (e) { if (currentOptionsKey.current === requestKey) studio.report(e); }
    finally { checkingRef.current = false; setChecking(false); }
  };
  return <div className="gs-page"><span className="eyebrow">EXPERIMENTS</span><h1>分段训练与实验</h1><p>{external ? `外部导入 ${x.import_id} · 独立实验，无 Run QA 声明` : `${x.run_id ?? '请先选择 Run'} · 分段通过且 QA 接受后才能训练`}</p>
    <section className="gs-card" aria-label="当前实验"><h2>当前实验 / 检查点</h2>
      <div className="gs-actions"><button className="button compact" onClick={() => void reload()}>读取实验清单</button><Field label="选择已记录实验"><select value={output} title={output} onChange={e => selectOutput(e.target.value)}><option value="">选择实验</option>{experiments.map((v, i) => <option key={i} value={String(v.output ?? '')}>{experimentLabel(v)} · {displayStatus(v.status)}</option>)}</select></Field></div>
      {output ? <><p className="gs-status" role="status">{experimentLabel({ output })} · 实际状态：{displayStatus(controlStatus)}{pendingControl ? ' · 控制请求等待步边界确认' : ''}</p>
        <div className="gs-actions"><button className="button primary" disabled={!active || !!pendingControl} onClick={() => void sendControl('pause')}>保存并暂停 / 释放 GPU</button><button className="button" disabled={studio.locked || checking || active || !snapshot} onClick={() => void start(true)}>检查并恢复已保存实验</button><button className="button" disabled={!active || !!pendingControl} onClick={() => void sendControl('checkpoint')}>保存检查点</button><button className="button" disabled={!active || !!pendingControl} onClick={() => void sendControl('stop')}>提前结束并保存</button></div>
        <TrainingMetrics state={state} />
        <SavedConfiguration dispatch={dispatch} config={record(state.config)} />
        <details><summary>高级控制 / 首次检查点前恢复</summary><div className="gs-actions"><button className="button" disabled={!active || !!pendingControl || controlStatus === 'suspended'} onClick={() => void sendControl('suspend')}>热挂起（保留 GPU）</button><button className="button" disabled={!active || !!pendingControl || controlStatus !== 'suspended'} onClick={() => void sendControl('unpause')}>解除热挂起</button></div><Check label="若首次检查点前失败，在身份核验后明确从第 0 步重试" checked={restart} change={setRestart} /></details>
        <div className="gs-actions"><button className="button" disabled={openingModel || studio.locked || active} onClick={() => void openSelectedModel()}>{openingModel ? '正在核验模型与训练数据…' : '核验模型身份并打开编辑'}</button></div>
      </> : <p className="gs-note">选择已记录实验以查看状态与控制；新实验须显式准备训练包。</p>}
      <details><summary>诊断：实验路径 / 已保存配置 / 控制身份</summary><Field label="当前实验目录（训练事件自动填入）"><input value={output} onChange={e => selectOutput(e.target.value)} /></Field><Json value={state} /></details>
    </section>
    {!external && <><Field label="通过的分段"><select value={segment} onChange={e => { setSegment(e.target.value); selectOutput(''); }}><option value="">选择分段</option>{(Array.isArray(choices.segments) ? choices.segments.map(record) : []).map(v => <option key={String(v.id)} value={String(v.id)}>{String(v.id)}</option>)}</select></Field><div className="gs-evidence"><span>可训练性：{displayStatus(choices.training_status)}</span><span>路线覆盖：{displayStatus(choices.coverage_status)}</span></div></>}
    <section className="gs-card"><div className="gs-actions"><h2>新实验</h2><button className="button" aria-expanded={creating} disabled={studio.locked || checking} onClick={() => setNewExperiment(creating ? null : viewKey)}>{creating ? '收起新实验' : '新建实验'}</button></div>
      {creating ? <><TrainingSettings value={settings} change={setSettings} disabled={studio.locked || checking} resolved={resolved} />
        {resolved ? <div role="status"><p>已解析 {String(resolved.training_preset)} · 来源 {String(resolved.source_kind)} · {String(resolved.steps)} 步 · SH {String(resolved.sh_degree)} · 上限 {String(resolved.max_splats)} 点 · seed {String(resolved.seed)}</p><p>图像尺寸：{Number(resolved.max_image_size) ? `最大边 ${resolved.max_image_size} px（等比缩放上限）` : '保留源图分辨率'} · {resolved.photo_comp && resolved.use_bilateral_grid ? '双边网格（相机补偿不叠加）' : resolved.photo_comp ? '相机补偿' : '光度补偿关闭'} · 稀疏深度 {resolved.use_sparse_depth ? '启用' : '关闭'} / {String(resolved.sparse_depth_observations)} 个有效观测</p><details><summary>诊断：新实验包路径 / 身份 / 解析结果</summary><Json value={resolved} /></details></div> : <p role="status">{optionsError || (hasTarget ? '正在解析当前训练包、预算与深度可用性…' : '选择导入项目或已接受 QA 的分段以解析设置')}</p>}
        <div className="gs-actions"><button className="button compact" disabled={studio.locked || checking || !hasTarget} onClick={() => setOptionsRevision(v => v + 1)}>重新验证训练选项</button><button className="button primary" disabled={studio.locked || checking || !resolved} onClick={() => void start(false)}>创建 GS 训练实验</button></div>
      </> : <p className="gs-note">点击“新建实验”后准备训练包、预算与深度选项；查看已有实验无需准备。</p>}
    </section>
    {output && digest && <><TrainingPreview studio={studio} output={output} digest={digest} /><div className="gs-actions"><button className="button" onClick={() => void studio.read('sources.list', { output, package_sha256: digest }).then(v => { if (currentOutput.current === output) setCameraInspection(Array.isArray(v.cameras) ? v.cameras.map(record) : []); }).catch(studio.report)}>检查训练包相机参数与输入图</button></div>{cameraInspection.length > 0 && <CameraInspection cameras={cameraInspection} />}</>}
  </div>;
}
function numeric(...values: unknown[]): number | undefined { return values.find(v => typeof v === 'number' && Number.isFinite(v)) as number | undefined; }
const count = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('zh-CN', { maximumFractionDigits: 0 }) : '—';
function SavedConfiguration({ dispatch, config }: { dispatch: Record<string, unknown>; config: Record<string, unknown> }) {
  const saved = { ...config, ...dispatch }, value = (key: string) => saved[key] == null ? '—' : String(saved[key]);
  const enabled = (key: string) => typeof saved[key] === 'boolean' ? saved[key] ? '启用' : '关闭' : '—';
  return <><h3>已保存的解析配置</h3><dl className="gs-summary-grid">
    <div><dt>预设 / 配方</dt><dd>{value('training_preset')} / {value('training_recipe')}</dd></div>
    <div><dt>预算 / SH</dt><dd>{count(saved.requested_steps ?? saved.steps)} 步 / {value('sh_degree')}</dd></div>
    <div><dt>图像尺寸</dt><dd>{saved.max_image_size === 0 ? '保留源图分辨率' : saved.max_image_size ? `最大边 ${count(saved.max_image_size)} px` : '—'}</dd></div>
    <div><dt>光度补偿</dt><dd>{saved.photo_comp === false ? '关闭' : saved.photo_comp === true ? saved.use_bilateral_grid === true ? '双边网格（相机补偿不叠加）' : saved.use_bilateral_grid === false ? '相机补偿' : '启用（方式未记录）' : '—'}</dd></div>
    <div><dt>稀疏深度（有效）</dt><dd>{enabled('use_sparse_depth')}</dd></div>
    <div><dt>点数上限 / seed</dt><dd>{count(saved.max_splats)} / {value('seed')}</dd></div>
  </dl></>;
}
function TrainingMetrics({ state }: { state: Record<string, unknown> }) {
  const training = record(state.training), control = record(state.state), dispatch = record(state.dispatch), config = record(state.config);
  const metrics = metricsFromLog(String(state.log_tail ?? '')), latest = metrics.at(-1);
  const knownSteps = [control.step, training.completed_steps, training.step, latest?.step].filter(v => typeof v === 'number' && Number.isFinite(v)) as number[];
  const step = knownSteps.length ? Math.max(...knownSteps) : undefined, budget = numeric(dispatch.requested_steps, config.steps, training.target_steps);
  const elapsed = numeric(latest?.elapsed_seconds, training.seconds_this_run), splats = numeric(latest?.splats, training.count);
  return <div aria-label="实测训练指标"><dl className="gs-summary-grid">
    <div><dt>最新步数 / 预算</dt><dd>{count(step)} / {count(budget)}</dd></div>
    <div><dt>本次耗时</dt><dd>{elapsed === undefined ? '—' : `${Math.floor(elapsed / 60)} 分 ${Math.floor(elapsed % 60)} 秒`}</dd></div>
    <div><dt>高斯点数</dt><dd>{count(splats)}</dd></div>
    <div><dt>最新损失</dt><dd>{latest?.loss.toPrecision(4) ?? '—'}</dd></div>
  </dl>{step !== undefined && budget !== undefined && budget > 0 && <progress aria-label="训练预算进度" max={budget} value={Math.min(step, budget)} />}
    <details><summary>实测曲线 / 评估指标</summary><LossChart metrics={metrics} /><Json value={state.metrics ?? {}} /></details>
    <details><summary>实际日志（有界读取 32 KiB）</summary><p className="gs-note">{String(state.log_path ?? '')}</p><pre className="gs-json" role="log">{String(state.log_tail ?? '等待训练日志')}</pre></details>
  </div>;
}
function metricsFromLog(text: string): Metric[] {
  const result: Metric[] = [];
  for (const line of text.split('\n')) {
    try {
      const value = record(JSON.parse(line)), data = record(value.data ?? value);
      if (typeof data.step === 'number' && Number.isFinite(data.step) && typeof data.loss === 'number' && Number.isFinite(data.loss)) result.push({ step: data.step, loss: data.loss, splats: numeric(data.splats), elapsed_seconds: numeric(data.elapsed_seconds) });
    } catch { /* Human log lines are displayed verbatim, never interpreted as success. */ }
  }
  return result.slice(-360);
}

async function loadPreviewFiles(preview: Record<string, unknown>): Promise<[string, string]> {
  const bytes = await verifiedPreviewBytes([modelUrl(String(preview.source_path)), modelUrl(String(preview.render_path))], [String(preview.source_sha256), String(preview.render_sha256)]);
  return bytes.map(value => URL.createObjectURL(new Blob([value], { type: 'image/png' }))) as [string, string];
}
