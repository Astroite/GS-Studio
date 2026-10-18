import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { selectionRectangle, viewportCamera } from './editorCoordinates';
import type { EditCrop, EditTransform } from './editorCoordinates';
import { SparkRenderer, SplatMesh, SplatEdit, SplatEditSdf, SplatEditSdfType } from '@sparkjsdev/spark';
import { Box, Crosshair, Grid3X3, House, ImagePlus, MousePointer2, RefreshCw, Scan, TriangleAlert } from 'lucide-react';
import { modelUrl } from './engine';
import { formatCount, readable } from './types';
import type { Model } from './types';
import { cameraVerticalFov, cameraWorldPose } from './cameras';
import type { CameraView, ReconstructionCamera } from './cameras';
import type { ViewportPolicy } from './renderPolicy';

interface Entry { mesh: SplatMesh; key: string; edit: SplatEdit; sdf: SplatEditSdf; box: THREE.Box3Helper; removed: boolean; loaded: boolean; meshDisposed: boolean }
interface Session {
  scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; spark: SparkRenderer;
  controls: OrbitControls; grid: THREE.GridHelper; entries: Map<string, Entry>; frame: (id?: string | null) => void; didFrame: boolean; scope: string; policy: ViewportPolicy;
  changed: () => void; selecting: boolean;
}
export interface ViewportAPI { camera: () => ReconstructionCamera | null }
export interface Gizmo { mode: 'translate' | 'rotate' | 'scale'; kind: 'transform' | 'crop'; transform: EditTransform; crop: EditCrop; change: (transform: EditTransform, crop: EditCrop) => void }
interface Props { models: Model[]; selectedId: string | null; scope: string; onImport: () => void; canImport: boolean; desktop: boolean; reference?: CameraView | null; renderPolicy: ViewportPolicy;
  apiRef?: RefObject<ViewportAPI | null>; onCameraChange?: () => void; selection?: { enabled: boolean; precise: boolean; nativeRender?: string; revision: number; onSelect: (camera: ReconstructionCamera, rectangle: number[]) => void }; gizmo?: Gizmo }

function disposeMesh(entry: Entry) {
  if (entry.meshDisposed) return;
  entry.meshDisposed = true; entry.mesh.dispose();
}
function disposeEntry(entry: Entry) {
  if (entry.removed) return;
  entry.removed = true; entry.mesh.removeFromParent();
  entry.box.geometry.dispose(); (entry.box.material as THREE.Material).dispose();
  if (entry.mesh.isInitialized) disposeMesh(entry);
  // Pending decoder results are disposed by the initialized continuation below.
}
function applyModel(entry: Entry, model: Model, selected: boolean) {
  const { mesh, sdf, edit, box } = entry;
  mesh.visible = model.visible;
  mesh.position.fromArray(model.transform.position);
  mesh.rotation.set(...model.transform.rotation, 'XYZ');
  mesh.scale.setScalar(model.transform.scale);
  const min = new THREE.Vector3().fromArray(model.crop.min), max = new THREE.Vector3().fromArray(model.crop.max);
  sdf.position.copy(min).add(max).multiplyScalar(0.5);
  // Spark BOX.scale is half-extents. It is parented to the model, so edits use model coordinates.
  sdf.scale.copy(max).sub(min).multiplyScalar(0.5).max(new THREE.Vector3(1e-8, 1e-8, 1e-8));
  // opacity=0 deletes the affected region: normal crop deletes outside, inverted crop deletes inside.
  edit.invert = !model.crop.invert;
  mesh.edits = model.crop.enabled ? [edit] : [];
  box.box.set(min, max); box.visible = model.crop.enabled && selected;
  mesh.updateMatrixWorld(true);
}

function resetOrbit(state: Session, up: THREE.Vector3, target: THREE.Vector3) {
  state.controls.dispose();
  state.camera.up.copy(up);
  state.controls = new OrbitControls(state.camera, state.renderer.domElement);
  state.controls.enableDamping = true; state.controls.dampingFactor = 0.08;
  state.controls.minDistance = 0.01; state.controls.maxDistance = 100000;
  state.controls.target.copy(target); state.controls.update();
  state.controls.enabled = !state.selecting;
  state.controls.addEventListener('end', state.changed);
}

function referenceView(state: Session, reference: ReconstructionCamera) {
  const pose = cameraWorldPose(reference), camera = state.camera;
  camera.position.setFromMatrixPosition(pose);
  camera.quaternion.setFromRotationMatrix(pose);
  camera.fov = cameraVerticalFov(reference); camera.near = state.policy.near; camera.far = state.policy.far;
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
  const bounds = new THREE.Box3();
  for (const entry of state.entries.values()) {
    if (entry.loaded && entry.mesh.visible) bounds.union(entry.mesh.getBoundingBox().applyMatrix4(entry.mesh.matrixWorld));
  }
  const distance = bounds.isEmpty() ? 5 : Math.max(0.1, bounds.getCenter(new THREE.Vector3()).sub(camera.position).dot(forward));
  resetOrbit(state, up, camera.position.clone().addScaledVector(forward, distance));
  camera.updateProjectionMatrix(); state.didFrame = true; state.changed();
}

export function Viewport({ models, selectedId, scope, onImport, canImport, desktop, reference, renderPolicy, apiRef, onCameraChange, selection, gizmo }: Props) {
  const host = useRef<HTMLDivElement>(null), session = useRef<Session | null>(null);
  const [generation, setGeneration] = useState(0), [ready, setReady] = useState(0);
  const [grid, setGrid] = useState(true), [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({ fps: 0, count: 0, loaded: 0, loading: 0 });
  const [assetErrors, setAssetErrors] = useState<Record<string, string>>({});
  const latest = useRef({ models, selectedId }); latest.current = { models, selectedId };
  const latestPolicy = useRef(renderPolicy); latestPolicy.current = renderPolicy;
  const latestGrid = useRef(grid); latestGrid.current = grid;
  const latestCameraChange = useRef(onCameraChange); latestCameraChange.current = onCameraChange;
  const latestGizmo = useRef(gizmo); latestGizmo.current = gizmo;
  const nativeMode = useRef(selection?.precise ?? false); nativeMode.current = selection?.precise ?? false;
  const [rectangle, setRectangle] = useState<number[] | null>(null), selectionStart = useRef<[number, number] | null>(null);
  const gizmoSession = useRef<{ controls: TransformControls; object: THREE.Object3D; parent: THREE.Object3D; dragging: boolean } | null>(null);

  useEffect(() => {
    if (apiRef) apiRef.current = { camera: () => { const s = session.current; return s ? viewportCamera(s.camera, s.renderer.domElement.clientWidth, s.renderer.domElement.clientHeight) : null; } };
    return () => { if (apiRef) apiRef.current = null; };
  }, [apiRef, ready]);
  useEffect(() => {
    const s = session.current; if (!s) return;
    s.selecting = !!selection?.enabled;
    s.controls.enabled = !selection?.enabled;
  }, [selection?.enabled, ready, reference]);
  useEffect(() => {
    const s = session.current; if (!s || !gizmo) return;
    const controls = new TransformControls(s.camera, s.renderer.domElement), object = new THREE.Object3D(), parent = new THREE.Object3D();
    const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: '#b4eb59', wireframe: true, depthTest: false }));
    if (gizmo.kind === 'crop') object.add(box);
    parent.add(object); s.scene.add(parent); s.scene.add(controls.getHelper()); controls.attach(object); controls.setMode(gizmo.mode); controls.setSpace(gizmo.kind === 'crop' ? 'local' : 'world');
    const current = { controls, object, parent, dragging: false }; gizmoSession.current = current;
    const dragging = (event: { value: unknown }) => { current.dragging = !!event.value; s.controls.enabled = !event.value; };
    const change = () => {
      if (!current.dragging) return; const g = latestGizmo.current; if (!g) return;
      if (g.kind === 'transform') {
        const axisIndex = controls.axis === 'Y' ? 1 : controls.axis === 'Z' ? 2 : 0;
        const scale = Math.max(0.0001, g.mode === 'scale' ? object.scale.getComponent(axisIndex) : g.transform.scale); object.scale.setScalar(scale);
        g.change({ ...g.transform, translation: object.position.toArray(), rotation_degrees: [object.rotation.x, object.rotation.y, object.rotation.z].map(v => v * 180 / Math.PI) as [number, number, number], scale }, g.crop);
      }
      else { const half = new THREE.Vector3(Math.abs(object.scale.x), Math.abs(object.scale.y), Math.abs(object.scale.z)).multiplyScalar(0.5).max(new THREE.Vector3(0.0001, 0.0001, 0.0001)); g.change(g.transform, { ...g.crop, min: object.position.clone().sub(half).toArray(), max: object.position.clone().add(half).toArray() }); }
    };
    controls.addEventListener('dragging-changed', dragging); controls.addEventListener('objectChange', change);
    return () => { controls.removeEventListener('dragging-changed', dragging); controls.removeEventListener('objectChange', change); controls.detach(); controls.dispose(); controls.getHelper().removeFromParent(); parent.removeFromParent(); box.geometry.dispose(); box.material.dispose(); gizmoSession.current = null; s.controls.enabled = !selection?.enabled; };
  }, [gizmo?.mode, gizmo?.kind, ready]);
  useEffect(() => {
    const current = gizmoSession.current; if (!current || !gizmo || current.dragging) return;
    if (gizmo.kind === 'transform') { current.object.position.fromArray(gizmo.transform.translation); current.object.rotation.set(...gizmo.transform.rotation_degrees.map(v => v * Math.PI / 180) as [number, number, number], 'XYZ'); current.object.scale.setScalar(gizmo.transform.scale); }
    else { current.parent.position.fromArray(gizmo.transform.translation); current.parent.rotation.set(...gizmo.transform.rotation_degrees.map(v => v * Math.PI / 180) as [number, number, number], 'XYZ'); current.parent.scale.setScalar(gizmo.transform.scale); current.object.position.fromArray(gizmo.crop.min).add(new THREE.Vector3().fromArray(gizmo.crop.max)).multiplyScalar(0.5); current.object.rotation.set(0, 0, 0); current.object.scale.fromArray(gizmo.crop.max).sub(new THREE.Vector3().fromArray(gizmo.crop.min)); }
  }, [gizmo, ready]);

  useEffect(() => {
    if (!host.current) return;
    const element = host.current;
    let renderer: THREE.WebGLRenderer | undefined, spark: SparkRenderer | undefined, controls: OrbitControls | undefined;
    let resize: ResizeObserver | undefined, stopped = false;
    let onLost: ((event: Event) => void) | undefined;
    setError(null); setAssetErrors({});
    try {
      const scene = new THREE.Scene(); scene.background = new THREE.Color('#101619');
      const policy = latestPolicy.current;
      const camera = new THREE.PerspectiveCamera(50, 1, policy.near, policy.far);
      camera.up.set(0, 1, 0); camera.position.set(5, 3.2, 5);
      renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'high-performance' });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.domElement.setAttribute('aria-label', 'Gaussian Splat 三维视口');
      element.appendChild(renderer.domElement);
      const webgl = renderer;
      // Drive async updates explicitly so rejected sorting/GPU work is surfaced in this viewport.
      spark = new SparkRenderer({ renderer, maxStdDev: Math.sqrt(8), autoUpdate: false,
        preBlurAmount: policy.preBlurAmount, blurAmount: policy.blurAmount }); scene.add(spark);
      const splats = spark;
      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true; controls.dampingFactor = 0.08; controls.target.set(0, 0.25, 0);
      controls.minDistance = 0.01; controls.maxDistance = 100000;
      const floor = new THREE.GridHelper(40, 80, '#284741', '#1c2b2d');
      floor.position.y = -0.015;
      (floor.material as THREE.Material).transparent = true; (floor.material as THREE.Material).opacity = 0.48;
      scene.add(floor);
      const state: Session = { scene, camera, renderer, spark, controls, grid: floor, entries: new Map(), didFrame: false, scope: '', policy, frame: () => {}, changed: () => latestCameraChange.current?.(), selecting: false };
      controls.addEventListener('end', state.changed);
      state.frame = (id) => {
        const bounds = new THREE.Box3();
        const target = new THREE.Vector3(0, 0.25, 0);
        for (const [entryId, entry] of state.entries) {
          if (id && state.entries.has(id) && entryId !== id) continue;
          if (!entry.loaded || latest.current.models.find(v => v.id === entryId)?.visible === false) continue;
          entry.mesh.updateMatrixWorld(true);
          bounds.union(entry.mesh.getBoundingBox().applyMatrix4(entry.mesh.matrixWorld));
        }
        if (bounds.isEmpty()) { camera.position.set(5, 3.2, 5); }
        else {
          const center = bounds.getCenter(new THREE.Vector3()), radius = Math.max(bounds.getSize(new THREE.Vector3()).length() / 2, 0.02);
          const fov = Math.min(camera.fov * Math.PI / 180, 2 * Math.atan(Math.tan(camera.fov * Math.PI / 360) * camera.aspect));
          const distance = radius / Math.sin(fov / 2) * 1.15;
          camera.position.copy(center).add(new THREE.Vector3(1, 0.65, 1).normalize().multiplyScalar(distance));
          target.copy(center); camera.near = state.policy.near; camera.far = state.policy.far;
        }
        resetOrbit(state, new THREE.Vector3(0, 1, 0), target); camera.updateProjectionMatrix();
        state.changed();
      };
      session.current = state;
      const size = () => {
        if (stopped) return;
        const width = Math.max(element.clientWidth, 1), height = Math.max(element.clientHeight, 1);
        webgl.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
        state.changed();
      };
      resize = new ResizeObserver(size); resize.observe(element); size();
      const fail = (message: string) => { if (!stopped) { setError(message); webgl.setAnimationLoop(null); } };
      renderer.debug.onShaderError = (gl, program) => fail(`GPU 着色器编译失败：${gl.getProgramInfoLog(program) ?? '未知错误'}`);
      onLost = event => { event.preventDefault(); fail('GPU 上下文已丢失。请重试以重新创建视口。'); };
      renderer.domElement.addEventListener('webglcontextlost', onLost);
      let frames = 0, previous = performance.now(), updating = false;
      renderer.setAnimationLoop(() => {
        if (stopped) return;
        try {
          state.controls.update();
          scene.background = nativeMode.current ? null : new THREE.Color('#101619');
          state.spark.visible = !nativeMode.current;
          state.grid.visible = !nativeMode.current && latestGrid.current;
          for (const [id, entry] of state.entries) entry.mesh.visible = !nativeMode.current && (latest.current.models.find(v => v.id === id)?.visible ?? false);
          if (!updating) {
            updating = true;
            void splats.update({ scene, camera }).catch(err => fail(readable(err))).finally(() => { updating = false; });
          }
          webgl.render(scene, camera); ++frames;
          const now = performance.now();
          if (now - previous >= 1000) {
            const entries = [...state.entries.values()];
            setStats({ fps: Math.round(frames * 1000 / (now - previous)),
              count: entries.filter(e => e.loaded && e.mesh.visible).reduce((sum, e) => sum + e.mesh.numSplats, 0),
              loaded: entries.filter(e => e.loaded).length, loading: entries.filter(e => !e.loaded).length });
            frames = 0; previous = now;
          }
        } catch (err) { fail(readable(err)); }
      });
      setReady(value => value + 1);
    } catch (err) { setError(`无法创建三维视口：${readable(err)}`); }
    return () => {
      stopped = true; resize?.disconnect(); renderer?.setAnimationLoop(null);
      if (renderer && onLost) renderer.domElement.removeEventListener('webglcontextlost', onLost);
      if (session.current) {
        session.current.controls.dispose();
        for (const entry of session.current.entries.values()) disposeEntry(entry);
        session.current.grid.geometry.dispose(); (session.current.grid.material as THREE.Material).dispose();
        session.current = null;
      }
      controls?.dispose(); spark?.dispose(); renderer?.dispose(); renderer?.domElement.remove();
    };
  }, [generation]);

  useEffect(() => { if (session.current) session.current.grid.visible = grid; }, [grid, ready]);
  useEffect(() => {
    const state = session.current; if (!state) return;
    state.policy = renderPolicy;
    state.camera.near = renderPolicy.near; state.camera.far = renderPolicy.far;
    state.camera.updateProjectionMatrix();
    state.spark.preBlurAmount = renderPolicy.preBlurAmount; state.spark.blurAmount = renderPolicy.blurAmount;
  }, [renderPolicy.near, renderPolicy.far, renderPolicy.mode, ready]);
  useEffect(() => {
    const state = session.current; if (!state) return;
    if (state.scope !== scope) { state.scope = scope; state.didFrame = false; }
    const wanted = new Set(models.map(model => model.id));
    for (const [id, entry] of state.entries) {
      const model = models.find(m => m.id === id);
      if (!wanted.has(id) || entry.key !== `${model?.path}:${model?.previewStep ?? ''}`) {
        disposeEntry(entry); state.entries.delete(id);
      }
    }
    for (const model of models) {
      let entry = state.entries.get(model.id);
      if (!entry) {
        if (!model.path) { setAssetErrors(errors => ({ ...errors, [model.id]: '模型没有可读取的本地路径' })); continue; }
        try {
          const mesh = new SplatMesh({ url: modelUrl(model.path, model.previewStep), lod: false });
          const edit = new SplatEdit({ softEdge: 0, sdfSmooth: 0 });
          const sdf = new SplatEditSdf({ type: SplatEditSdfType.BOX, opacity: 0 });
          edit.add(sdf); mesh.add(edit);
          const box = new THREE.Box3Helper(new THREE.Box3(), '#b4eb59');
          (box.material as THREE.LineBasicMaterial).depthTest = false; (box.material as THREE.LineBasicMaterial).transparent = true;
          (box.material as THREE.LineBasicMaterial).opacity = 0.55; box.renderOrder = 10; mesh.add(box);
          entry = { mesh, edit, sdf, box, key: `${model.path}:${model.previewStep ?? ''}`, removed: false, loaded: false, meshDisposed: false };
          const loadedEntry = entry;
          state.entries.set(model.id, entry); state.scene.add(mesh);
          void mesh.initialized.then(() => {
            if (loadedEntry.removed) { disposeMesh(loadedEntry); return; }
            loadedEntry.loaded = true;
            const current = latest.current.models.find(m => m.id === model.id);
            if (current) applyModel(loadedEntry, current, latest.current.selectedId === current.id);
            setAssetErrors(errors => { const next = { ...errors }; delete next[model.id]; return next; });
            if (!state.didFrame && mesh.visible) { state.frame(); state.didFrame = true; }
          }).catch(err => {
            if (!loadedEntry.removed) {
              setAssetErrors(errors => ({ ...errors, [model.id]: `${model.name}：${readable(err)}` }));
              disposeEntry(loadedEntry); state.entries.delete(model.id);
            }
            // Decode failures can leave partially allocated buffers, including stale pending loads.
            disposeMesh(loadedEntry);
          });
        } catch (err) { setAssetErrors(errors => ({ ...errors, [model.id]: readable(err) })); }
      }
      if (entry) applyModel(entry, model, selectedId === model.id);
    }
  }, [models, selectedId, scope, ready]);

  useEffect(() => {
    if (!reference || !session.current) return;
    try { referenceView(session.current, reference.camera); }
    catch (error) { setError(`无法使用重建相机：${readable(error)}`); }
  }, [reference, scope, ready]);

  return <section className="viewport" aria-label="三维工作区">
    <div className="viewport-toolbar"><div className="view-title"><Scan size={14} /><span title={reference?.camera.name}>{reference ? '重建相机视角' : '透视视图'}</span><span className="quiet">/</span><span className="quiet">Gaussian Splatting</span></div>
      <div className="view-tools"><button className={grid ? 'icon-button active' : 'icon-button'} title="切换地面网格" aria-label="切换地面网格" onClick={() => setGrid(v => !v)}><Grid3X3 size={16} /></button>
        <button className="icon-button" title="聚焦选中模型（未选择时聚焦整个场景）" aria-label="聚焦选中模型" onClick={() => session.current?.frame(selectedId)}><Crosshair size={16} /></button>
        <button className="icon-button" title="回到默认视角" aria-label="回到默认视角" onClick={() => { const s = session.current; if (s) { if (reference) referenceView(s, reference.camera); else { s.camera.position.set(5, 3.2, 5); resetOrbit(s, new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0.25, 0)); } } }}><House size={16} /></button></div>
    </div>
    <div ref={host} className="canvas-host" />
    {selection?.precise && <div className="native-viewport">{selection.nativeRender ? <img draggable={false} src={modelUrl(selection.nativeRender, selection.revision)} alt="与精确选择一致的原生渲染" /> : <span>当前视角原生渲染更新中…</span>}</div>}
    {selection?.enabled && <div className="selection-canvas" onPointerDown={e => {
      if (selection.precise && !selection.nativeRender) return;
      e.currentTarget.setPointerCapture(e.pointerId); selectionStart.current = [e.clientX, e.clientY]; setRectangle([e.nativeEvent.offsetX, e.nativeEvent.offsetY, e.nativeEvent.offsetX, e.nativeEvent.offsetY]);
    }} onPointerMove={e => { const start = selectionStart.current; if (!start) return; const bounds = e.currentTarget.getBoundingClientRect(); setRectangle(selectionRectangle(start, [e.clientX, e.clientY], bounds, bounds.width, bounds.height)); }} onPointerUp={e => {
      const start = selectionStart.current; selectionStart.current = null; setRectangle(null); const camera = apiRef?.current?.camera();
      if (start && camera) { const rect = selectionRectangle(start, [e.clientX, e.clientY], e.currentTarget.getBoundingClientRect(), camera.width, camera.height); if (rect[2] - rect[0] >= 2 && rect[3] - rect[1] >= 2) selection.onSelect(camera, rect); }
    }} onPointerCancel={() => { selectionStart.current = null; setRectangle(null); }}>{rectangle && <div className="selection-rectangle" style={{ left: rectangle[0], top: rectangle[1], width: rectangle[2] - rectangle[0], height: rectangle[3] - rectangle[1] }} />}</div>}
    <div className="viewport-vignette" />
    {!models.length && !error && <div className="empty-viewport">
      <div className="empty-symbol"><div className="symbol-orbit orbit-one" /><div className="symbol-orbit orbit-two" /><Box size={40} strokeWidth={1} /><i /><i /><i /></div>
      <span className="eyebrow">YOUR NEXT DIMENSION</span><h1>从真实世界，走进三维空间</h1>
      <p>导入多视角照片或视频，重建相机并训练<br />属于你的 3D Gaussian Splatting 场景。</p>
      <button className="button primary" disabled={!canImport} title={canImport ? '选择照片或视频开始' : desktop ? '先新建或打开项目' : '请在桌面程序中打开以连接本地引擎'} onClick={onImport}><ImagePlus size={16} />导入素材<span className="button-arrow">↗</span></button>
      <div className="format-note">JPG · PNG · MP4 <span>/</span> 支持直接导入 PLY / SPZ 模型</div>
    </div>}
    {(error || Object.keys(assetErrors).length > 0) && <div className="viewport-error" role="alert"><TriangleAlert size={22} /><strong>视口加载失败</strong><p>{error ?? Object.values(assetErrors).join('\n')}</p><button className="button" onClick={() => { setStats({ fps: 0, count: 0, loaded: 0, loading: 0 }); setGeneration(v => v + 1); }}><RefreshCw size={14} />重新加载视口</button></div>}
    <div className="axis-widget" aria-label="右手坐标系，Y 轴向上"><span className="axis-y">Y</span><span className="axis-z">Z</span><span className="axis-x">X</span><i /></div>
    <div className="viewport-bottom"><span><MousePointer2 size={12} />左键旋转 <b>·</b> 右键平移 <b>·</b> 滚轮缩放</span><span className="mono">{stats.loading ? '加载中 · ' : ''}{selection?.precise ? 'NATIVE / EXACT CAMERA' : stats.loaded ? `${formatCount(stats.count)} splats` : 'Y-UP'}<b>·</b>{stats.loaded && !error ? `${stats.fps} FPS` : 'PERSPECTIVE'}</span></div>
  </section>;
}
