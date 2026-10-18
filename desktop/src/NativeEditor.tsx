import { useEffect, useRef, useState } from 'react';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { engineCall, desktop, modelUrl } from './engine';
import { Viewport } from './Viewport';
import type { Gizmo, ViewportAPI } from './Viewport';
import { projectRenderPolicy } from './renderPolicy';
import { identityTransform, validEdit, editorPreviewArgs } from './editorCoordinates';
import type { EditTransform, EditCrop } from './editorCoordinates';
import type { Workbench } from './useWorkbench';
import { normalizeProject, record } from './types';
import { LatestRequest } from './studioState';
import { Field, Check, Json } from './Workflow';
import { reconstructionCamera } from './cameras';
import type { CameraView, ReconstructionCamera } from './cameras';

export function NativeEditor({ work, blocked, onBusyChange, onDirtyChange, rpc = engineCall, Scene = Viewport }: { work: Workbench; blocked: boolean; onBusyChange?: (busy: boolean) => void; onDirtyChange?: (dirty: boolean) => void; rpc?: typeof engineCall; Scene?: typeof Viewport }) {
  const [state, setState] = useState<Record<string, unknown>>({}), [pending, setPending] = useState(false), [mode, setMode] = useState('view'), [xray, setXray] = useState(false), [highlight, setHighlight] = useState(true);
  const [transform, setTransform] = useState<EditTransform>(identityTransform), [crop, setCrop] = useState<EditCrop>({ min: [-1, -1, -1], max: [1, 1, 1], keep_inside: true }), [cropEnabled, setCropEnabled] = useState(false), [dirty, setDirty] = useState(false), [previewReady, setPreviewReady] = useState(false);
  const [native, setNative] = useState(''), [cameraVersion, setCameraVersion] = useState(0), [points, setPoints] = useState<number[][]>([]), [renderFailed, setRenderFailed] = useState(false);
  // Camera-only renders reuse the persisted edit revision and file path.
  // Each accepted frame needs its own URL so WebView cannot display an older
  // camera's cached pixels while native selection uses the current camera.
  const [nativeVersion, setNativeVersion] = useState(() => Date.now());
  const [cleanRadius, setCleanRadius] = useState(0.15);
  const cameraRevision = useRef(0);
  const draftRevision = useRef(0);
  const viewport = useRef<ViewportAPI | null>(null), requests = useRef(new LatestRequest()), pendingRef = useRef(false), previewPromise = useRef<Promise<void> | null>(null);
  const project = work.project, model = project?.models.find(v => v.id === work.selectedId), identity = `${project?.path}:${model?.id}`;
  const reconstruction = record(project?.reconstruction);
  const cameras = (Array.isArray(reconstruction.cameras) ? reconstruction.cameras : []).map(reconstructionCamera).filter((c): c is ReconstructionCamera => c !== null);
  const [reference, setReference] = useState<CameraView | null>(null), [referenceIndex, setReferenceIndex] = useState('');
  const referenceVersion = useRef(0), referenceKey = JSON.stringify([project?.path, cameras[0] ?? null]);
  useEffect(() => { setReferenceIndex(cameras.length ? '0' : ''); setReference(cameras.length ? { camera: cameras[0], version: ++referenceVersion.current } : null); }, [referenceKey]);
  const latestIdentity = useRef(identity); latestIdentity.current = identity;
  const dirtyRef = useRef(dirty); dirtyRef.current = dirty;
  useEffect(() => { onBusyChange?.(pending); }, [pending, onBusyChange]);
  useEffect(() => { onDirtyChange?.(dirty); }, [dirty, onDirtyChange]);
  const locked = blocked || pending || !!work.operation || !model;
  const fromModel = (p = project) => {
    const m = p?.models.find(v => v.id === model?.id); if (!m) return;
    setTransform({ ...identityTransform(), translation: m.transform.position, rotation_degrees: m.transform.rotation.map(v => v * 180 / Math.PI) as [number, number, number], scale: m.transform.scale });
    setCrop({ min: m.crop.min, max: m.crop.max, keep_inside: !m.crop.invert }); setCropEnabled(m.crop.enabled);
  };
  const call = async (action: string, args: Record<string, unknown> = {}, accept = true, guard: () => boolean = () => true) => {
    if (!project || !model) return null;
    const scope = identity, token = requests.current.next(), viewRevision = cameraRevision.current;
    const result = record(await rpc('editor.operation', { path: project.path, model_id: model.id, action, args }));
    if (!requests.current.current(token) || latestIdentity.current !== scope || !guard()) return null;
    setState(result); if (typeof result.native_render === 'string' && viewRevision === cameraRevision.current) { setNative(result.native_render); setNativeVersion(v => Math.max(v + 1, Date.now())); }
    if (accept && result.project) { const p = normalizeProject(result.project); work.acceptProject(p); if (action !== 'preview') fromModel(p); }
    return result;
  };
  const perform = async (action: string, args: Record<string, unknown> = {}) => {
    if (pendingRef.current || blocked || work.operation) return;
    pendingRef.current = true; setPending(true);
    try { await call(action, { ...args, ...(viewport.current?.camera() ? { camera: viewport.current.camera() } : {}) }); if (['commit', 'cancel_preview', 'undo', 'redo'].includes(action)) { setDirty(false); setPreviewReady(false); } }
    catch (e) { setRenderFailed(true); work.report(e); }
    finally { pendingRef.current = false; setPending(false); }
  };
  const cleanCopy = async () => {
    if (!project || !model || locked || dirty) return;
    pendingRef.current = true; setPending(true);
    try {
      await work.save();
      // Core cleans the committed editor state and extends its original-index
      // undo history. Reimporting a baked export would discard that history.
      const result = normalizeProject(await rpc('model.clean', { path: project.path, model_id: model.id, max_screen_radius: cleanRadius }));
      work.acceptProject(result);
      const derivative = result.models.find(v => v.visible && !project.models.some(old => old.id === v.id));
      if (!derivative) throw new Error('清理没有返回新的独立模型身份');
      work.setSelectedId(derivative.id);
      work.setNotice('已生成包含当前全部已提交编辑的独立清理副本');
    } catch (e) {
      work.report(e);
      try { work.acceptProject(normalizeProject(await rpc('project.open', { path: project.path }))); } catch { /* Preserve last known evidence. */ }
    } finally { pendingRef.current = false; setPending(false); }
  };
  const cancelDraft = async () => {
    if (pendingRef.current || blocked) return;
    ++draftRevision.current; setPreviewReady(false); setDirty(false); setMode('view');
    pendingRef.current = true; setPending(true);
    try { await previewPromise.current; await call('cancel_preview', { ...(viewport.current?.camera() ? { camera: viewport.current.camera() } : {}) }); fromModel(); }
    catch (e) { work.report(e); }
    finally { pendingRef.current = false; setPending(false); }
  };
  useEffect(() => { requests.current.invalidate(); setState({}); setNative(''); setPoints([]); setDirty(false); setPreviewReady(false); fromModel(); if (model && !blocked) void perform('cancel_preview'); return () => { requests.current.invalidate(); onBusyChange?.(false); onDirtyChange?.(false); if (dirtyRef.current && project && model) { const path = project.path, model_id = model.id; void (async () => { await previewPromise.current; try { await rpc('editor.operation', { path, model_id, action: 'cancel_preview' }); } catch (e) { work.report(e); } })(); } }; }, [identity, blocked]);
  useEffect(() => {
    if (blocked || !model || !desktop || pending || native || renderFailed || dirty) return;
    const timer = window.setTimeout(() => { if (!pendingRef.current) void perform('render'); }, 350);
    return () => window.clearTimeout(timer);
  }, [identity, cameraVersion, blocked, pending, native, renderFailed, dirty]);
  useEffect(() => {
    const draftToken = ++draftRevision.current;
    if (!dirty || blocked || !validEdit(transform, cropEnabled ? crop : undefined)) return;
    setPreviewReady(false); setNative(''); setPoints([]);
    const timer = window.setTimeout(() => {
      const promise = (async () => {
        if (previewPromise.current) await previewPromise.current;
        if (draftToken !== draftRevision.current) return;
        const args = { ...editorPreviewArgs(transform, crop, cropEnabled), ...(viewport.current?.camera() ? { camera: viewport.current.camera() } : {}) };
        try { const value = await call('preview', args, true, () => draftToken === draftRevision.current); if (value) setPreviewReady(true); } catch (e) { work.report(e); }
      })();
      previewPromise.current = promise; void promise.finally(() => { if (previewPromise.current === promise) previewPromise.current = null; });
    }, 300);
    return () => window.clearTimeout(timer);
  }, [transform, crop, cropEnabled, dirty, blocked, identity, cameraVersion]);
  useEffect(() => {
    const escape = (e: KeyboardEvent) => { if (e.key === 'Escape' && dirty) { e.preventDefault(); void cancelDraft(); } };
    window.addEventListener('keydown', escape); return () => window.removeEventListener('keydown', escape);
  }, [dirty, identity]);
  useEffect(() => {
    let live = true; setPoints([]); const path = state.selection_preview;
    if (typeof path === 'string' && highlight) new PLYLoader().load(modelUrl(path, Number(state.revision)), geometry => { if (live) { const p = geometry.getAttribute('position'); const next: number[][] = []; for (let i = 0; i < Math.min(p.count, 10000); i++) next.push([p.getX(i), p.getY(i), p.getZ(i)]); setPoints(next); } geometry.dispose(); }, undefined, e => { if (live) work.report(e); });
    return () => { live = false; };
  }, [state.selection_preview, state.revision, highlight]);
  const vector = (label: string, value: [number, number, number], change: (v: [number, number, number]) => void) => <Field label={label}><div className="gs-grid three">{value.map((v, i) => <input key={i} type="number" step="0.01" aria-label={`${label} ${'XYZ'[i]}`} value={v} disabled={locked} onChange={e => { const next = [...value] as [number, number, number]; next[i] = Number(e.target.value); change(next); setDirty(true); }} />)}</div></Field>;
  let gizmo: Gizmo | undefined;
  if (['translate', 'rotate', 'scale', 'crop-move', 'crop-scale'].includes(mode) && model && !blocked) gizmo = { mode: mode === 'crop-move' ? 'translate' : mode === 'crop-scale' ? 'scale' : mode as Gizmo['mode'], kind: mode.startsWith('crop') ? 'crop' : 'transform', transform, crop, change: (t, c) => { setTransform(t); setCrop(c); if (mode.startsWith('crop')) setCropEnabled(true); setDirty(true); } };
  const camera = viewport.current?.camera();
  const projected = camera ? points.map(p => projectPoint(p, camera)).filter((v): v is number[] => !!v) : [];
  return <div className="gs-editor"><div className="gs-editor-main"><div className="gs-editor-toolbar"><select value={work.selectedId ?? ''} disabled={locked || dirty} onChange={e => work.setSelectedId(e.target.value)}><option value="">选择模型</option>{project?.models.map(v => <option key={v.id} value={v.id}>{v.name}</option>)}</select><select aria-label="编辑工具" value={mode} disabled={locked} onChange={e => setMode(e.target.value)}><option value="view">观察</option><option value="select">矩形选择</option><option value="translate">移动手柄</option><option value="rotate">旋转手柄</option><option value="scale">统一缩放手柄</option><option value="crop-move">裁剪框移动</option><option value="crop-scale">裁剪框尺寸</option></select><Check label="穿透 X-ray" checked={xray} change={v => { setXray(v); setNative(''); setCameraVersion(v => v + 1); }} disabled={locked} /><Check label="高亮选择" checked={highlight} change={setHighlight} /><span>{String(state.selection_count ?? 0)} 选中</span></div>
      {cameras.length > 0 && <div className="gs-editor-toolbar"><Field label="训练相机视角"><select value={referenceIndex} disabled={locked || dirty} onChange={e => { const index = e.target.value; setReferenceIndex(index); setReference({ camera: cameras[Number(index)], version: ++referenceVersion.current }); }}>{cameras.map((c, i) => <option key={i} value={i}>{i + 1} · {c.name}</option>)}</select></Field></div>}
      <div className="gs-editor-scene"><Scene models={project?.models ?? []} selectedId={work.selectedId} scope={project?.path ?? ''} desktop={desktop} canImport={false} onImport={() => {}} reference={reference} renderPolicy={projectRenderPolicy(project)} apiRef={viewport} onCameraChange={() => { ++cameraRevision.current; setNative(''); setRenderFailed(false); setCameraVersion(v => v + 1); }} selection={{ enabled: mode === 'select' && !locked && !dirty, precise: !xray || !!record(model?.editor).deleted_count || dirty, nativeRender: native, revision: nativeVersion, onSelect: (camera, rectangle) => void perform('select_rect', { camera, rectangle, xray }) }} gizmo={gizmo} />
        {highlight && camera && projected.length > 0 && <svg className="gs-selection-highlight" viewBox={`0 0 ${camera.width} ${camera.height}`} preserveAspectRatio="none">{projected.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={2} fill="#b4eb59" opacity={0.8} />)}</svg>}
      </div><div className="gs-actions gs-editor-actions">{[['invert', '反选'], ['clear', '清空'], ['delete', '删除选中'], ['undo', '撤销'], ['redo', '重做'], ['state', '重开已保存历史']].map(([action, title]) => <button key={action} className="button compact" disabled={locked || dirty || action === 'undo' && !state.can_undo || action === 'redo' && !state.can_redo || action === 'delete' && !state.selection_count} onClick={() => void perform(action)}>{title}</button>)}<button className="button primary" disabled={locked || dirty} onClick={() => void work.exportModel()}>导出已提交编辑</button></div></div>
    <aside className="gs-editor-inspector"><h2>变换 / 裁剪预览</h2><p className="gs-note">拖动手柄或调整数值后确认，写入一次撤销；Esc 取消。原模型和训练检查点保持独立。</p>{vector('位置', transform.translation, v => setTransform(t => ({ ...t, translation: v })))}{vector('旋转 °', transform.rotation_degrees, v => setTransform(t => ({ ...t, rotation_degrees: v })))}<Field label="统一缩放"><input type="number" min="0.0001" step="0.01" value={transform.scale} disabled={locked} onChange={e => { setTransform(t => ({ ...t, scale: Number(e.target.value) })); setDirty(true); }} /></Field><Check label="启用模型局部裁剪框" checked={cropEnabled} disabled={locked} change={v => { setCropEnabled(v); setDirty(true); }} />{vector('裁剪最小值', crop.min, v => setCrop(c => ({ ...c, min: v })))}{vector('裁剪最大值', crop.max, v => setCrop(c => ({ ...c, max: v })))}<Check label="保留框内（取消则保留框外）" checked={crop.keep_inside} disabled={locked} change={v => { setCrop(c => ({ ...c, keep_inside: v })); setDirty(true); }} /><div className="gs-actions"><button className="button primary" disabled={locked || !dirty || !previewReady || !validEdit(transform, cropEnabled ? crop : undefined)} onClick={() => void perform('commit')}>确认一次编辑</button><button className="button" disabled={locked || !dirty} onClick={() => void cancelDraft()}>取消 / Esc</button></div><p role="status">{pending ? '等待原生编辑…' : dirty ? previewReady ? '预览已就绪，等待确认' : '预览更新中…' : `已保存 · revision ${state.revision ?? '—'}`}</p><section className="gs-card"><h2>投影几何清理</h2><Field label="最大投影半径 / 图像边长"><input type="number" min="0.001" max={1} step="0.01" value={cleanRadius} disabled={locked || dirty} onChange={e => setCleanRadius(Number(e.target.value))} /></Field><button className="button" disabled={locked || dirty || !(Array.isArray(record(project?.reconstruction).cameras) && (record(project?.reconstruction).cameras as unknown[]).length) || cleanRadius <= 0 || cleanRadius > 1} onClick={() => void cleanCopy()}>按训练相机生成清理副本</button><p className="gs-note">清理输出独立模型和诊断报告，原模型保留。</p></section><details><summary>实际视口相机 K / w2c</summary><Json value={camera} /></details></aside>
  </div>;
}
function projectPoint(p: number[], c: ReconstructionCamera): number[] | null {
  const v = c.w2c.slice(0, 3).map(row => row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3]);
  if (v[2] <= 0) return null; const x = c.K[0][0] * v[0] / v[2] + c.K[0][2], y = c.K[1][1] * v[1] / v[2] + c.K[1][2];
  return x >= 0 && x <= c.width && y >= 0 && y <= c.height ? [x, y] : null;
}
