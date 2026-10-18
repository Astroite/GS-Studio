import { useCallback, useEffect, useRef, useState } from 'react';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { desktop, desktopMessage, engineCall, pickDirectory, pickExport, pickImport, subscribeEngine } from './engine';
import { finite, normalizeModel, normalizeProject, readable, record } from './types';
import type { EngineEvent, Health, LogEntry, Metric, Model, Operation, Progress, Project, Settings } from './types';

const lastProjectKey = 'gsstudio:last-model-project';
const titles: Record<string, string> = { 'project.import': '导入素材', 'model.import': '导入模型', reconstruct: '相机重建', train: '训练模型', resume: '恢复训练', 'model.export': '导出模型', 'model.clean': '清理大投影' };
export function useWorkbench() {
  const [project, setProject] = useState<Project | null>(null);
  const projectRef = useRef(project); projectRef.current = project;
  const [projects, setProjects] = useState<Project[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [connection, setConnection] = useState<'browser' | 'connecting' | 'ready' | 'error'>(desktop ? 'connecting' : 'browser');
  const [connectionError, setConnectionError] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState('');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [preview, setPreview] = useState<Model | null>(null);
  const [operation, setOperation] = useState<Operation | null>(null);
  const operationRef = useRef<Operation | null>(null);
  const [projectBusy, setProjectBusy] = useState(false);
  const projectBusyRef = useRef(false);
  const [control, setControl] = useState<string | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saveState, setSaveState] = useState<'saved' | 'pending' | 'saving' | 'error'>('saved');
  const revision = useRef(0), savedRevision = useRef(0), saving = useRef<Promise<void> | null>(null);
  const logId = useRef(0), alive = useRef(true);
  const appendLog = useCallback((level: string, message: string) => {
    setLogs(previous => [...previous, { id: ++logId.current, time: new Date().toLocaleTimeString('zh-CN', { hour12: false }), level, message: message.slice(0, 16384) }].slice(-500));
  }, []);
  const report = useCallback((err: unknown) => { const message = readable(err); setError(message); appendLog('error', message); }, [appendLog]);
  const acceptProject = useCallback((next: Project) => {
    projectRef.current = next; setProject(next);
    setProjects(list => [next, ...list.filter(p => p.path !== next.path)]);
    setSelectedId(id => next.models.some(m => m.id === id) ? id : next.models.find(m => m.visible)?.id ?? next.models[0]?.id ?? null);
    setDirty(false); setSaveState('saved'); savedRevision.current = ++revision.current;
    try { localStorage.setItem(lastProjectKey, next.path); } catch { /* Storage can be disabled. */ }
  }, []);
  const saveProject = useCallback(async () => {
    if (saving.current) await saving.current;
    const current = projectRef.current;
    if (!current || savedRevision.current === revision.current) return;
    const version = revision.current;
    setSaveState('saving');
    const promise = (async () => {
      try {
        const result = normalizeProject(await engineCall('project.update', { path: current.path, models: current.models, settings: current.settings }));
        if (projectRef.current?.path !== current.path) return;
        savedRevision.current = version;
        if (version === revision.current) {
          projectRef.current = result; setProject(result); setDirty(false); setSaveState('saved');
          setProjects(list => list.map(p => p.path === result.path ? result : p));
        } else setSaveState('pending');
      } catch (err) { setSaveState('error'); throw err; }
    })();
    saving.current = promise;
    try { await promise; } finally { if (saving.current === promise) saving.current = null; }
  }, []);
  useEffect(() => {
    if (!dirty || operation || projectBusy || saveState === 'error') return;
    const timer = window.setTimeout(() => { void saveProject().catch(report); }, 700);
    return () => window.clearTimeout(timer);
  }, [project, dirty, operation, projectBusy, saveState, saveProject, report]);

  useEffect(() => {
    if (!desktop) return;
    const nativeWindow = getCurrentWindow();
    let disposed = false, closing = false, allowClose = false;
    let unlisten: (() => void) | undefined;
    void nativeWindow.onCloseRequested(event => {
      if (allowClose || savedRevision.current === revision.current) return;
      event.preventDefault();
      if (closing) return;
      closing = true;
      void saveProject().then(async () => {
        allowClose = true;
        await nativeWindow.close();
      }).catch(error => { closing = false; allowClose = false; report(error); });
    }).then(dispose => { if (disposed) dispose(); else unlisten = dispose; }).catch(report);
    return () => { disposed = true; unlisten?.(); };
  }, [saveProject, report]);

  const eventHandler = useRef<(event: EngineEvent) => void>(() => {});
  eventHandler.current = event => {
    const data = event.data;
    if (event.event === 'log') { appendLog(String(data.level ?? 'info'), String(data.message ?? data.text ?? readable(data))); return; }
    if (!operationRef.current) return;
    if (event.event === 'progress') setProgress({ stage: String(data.stage ?? operationRef.current.title),
      progress: typeof data.progress === 'number' && Number.isFinite(data.progress) ? Math.max(0, Math.min(1, data.progress)) : undefined,
      message: String(data.message ?? '') });
    if (event.event === 'metric' && typeof data.loss === 'number' && Number.isFinite(data.loss)) {
      const metric: Metric = { step: finite(data.step), loss: data.loss };
      for (const key of ['psnr', 'splats', 'elapsed_seconds', 'gpu_memory_mb'] as const) {
        if (typeof data[key] === 'number' && Number.isFinite(data[key])) metric[key] = data[key];
      }
      setMetrics(list => [...list, metric].slice(-360));
    }
    if (event.event === 'preview' && ['train', 'resume'].includes(operationRef.current.method) && typeof data.path === 'string') {
      setPreview({ ...normalizeModel({ id: '__preview__', name: '训练实时预览', path: data.path, count: data.count }), previewStep: finite(data.step) });
    }
  };

  const refresh = useCallback(async (restore = false) => {
    if (!desktop) { setConnectionError(desktopMessage); return; }
    setConnection('connecting'); setConnectionError('');
    try {
      const runtime = record(await engineCall('health'));
      if (!alive.current) return;
      setHealth(runtime); setConnection('ready');
      appendLog('info', '本地引擎已连接；运行环境信息已更新');
      const result = await engineCall('project.list');
      const list = Array.isArray(result) ? result : record(result).projects;
      const normalized = Array.isArray(list) ? list.map(normalizeProject) : [];
      if (!alive.current) return;
      setProjects(normalized);
      let last: string | null = null;
      try { last = localStorage.getItem(lastProjectKey); } catch { /* Optional session persistence. */ }
      if (restore && last && !projectRef.current) {
        try { acceptProject(normalizeProject(await engineCall('project.open', { path: last }))); }
        catch (err) { appendLog('warn', `无法恢复上次项目：${readable(err)}`); }
      }
    } catch (err) {
      if (!alive.current) return;
      setConnection('error'); setConnectionError(readable(err)); appendLog('error', readable(err));
    }
  }, [acceptProject, appendLog]);
  useEffect(() => {
    alive.current = true;
    let dispose: (() => void) | undefined;
    void (async () => {
      try {
        const unlisten = await subscribeEngine(event => eventHandler.current(event));
        if (!alive.current) { unlisten(); return; }
        dispose = unlisten;
        await refresh(true);
      } catch (err) { if (alive.current) { setConnection('error'); setConnectionError(readable(err)); } }
    })();
    return () => { alive.current = false; dispose?.(); };
  }, [refresh]);

  const editProject = (change: (p: Project) => Project) => {
    if (!projectRef.current || operationRef.current || projectBusyRef.current) return;
    const next = change(projectRef.current); projectRef.current = next; setProject(next);
    ++revision.current; setDirty(true); setSaveState('pending');
  };
  const editModel = (id: string, change: (m: Model) => Model) => editProject(p => ({ ...p, models: p.models.map(m => m.id === id ? change(m) : m) }));
  const updateSettings = (settings: Partial<Settings>) => editProject(p => ({ ...p, settings: { ...p.settings, ...settings } }));
  const openProject = async (path?: string) => {
    if (operationRef.current || projectBusyRef.current) return;
    projectBusyRef.current = true; setProjectBusy(true);
    try {
      const target = path ?? await pickDirectory(); if (!target) return;
      await saveProject();
      acceptProject(normalizeProject(await engineCall('project.open', { path: target })));
      setMetrics([]); setPreview(null); setProgress(null); setNotice('项目已打开');
    } catch (err) { report(err); }
    finally { projectBusyRef.current = false; setProjectBusy(false); }
  };
  const createProject = async (name: string, path?: string) => {
    if (operationRef.current || projectBusyRef.current) return false;
    projectBusyRef.current = true; setProjectBusy(true);
    try {
      await saveProject();
      acceptProject(normalizeProject(await engineCall('project.create', { name: name.trim(), ...(path ? { path } : {}) })));
      setMetrics([]); setPreview(null); setProgress(null); setNotice('项目已创建'); return true;
    } catch (err) { report(err); return false; }
    finally { projectBusyRef.current = false; setProjectBusy(false); }
  };
  const run = async (method: string, params: Record<string, unknown> = {}) => {
    const current = projectRef.current;
    if (!current || operationRef.current || projectBusyRef.current) return;
    const op: Operation = { method, title: titles[method] ?? method, path: current.path, started: Date.now(), dispatched: false };
    operationRef.current = op; setOperation(op); setProgress(null);
    if (method === 'train' || method === 'resume') { setMetrics([]); setPreview(null); }
    try {
      await saveProject();
      const dispatched = { ...op, dispatched: true };
      operationRef.current = dispatched; setOperation(dispatched);
      appendLog('info', `${op.title} · ${current.name}`);
      const result = await engineCall(method, { path: current.path, ...params });
      if (method === 'model.export') {
        const exported = record(result); setNotice(`模型已导出：${exported.path ?? params.file}`); appendLog('info', `模型已导出：${exported.path ?? params.file}`);
      } else {
        const next = normalizeProject(result);
        acceptProject(next);
        if (method === 'model.clean') {
          const added = next.models.find(model => model.visible && !current.models.some(old => old.id === model.id));
          if (added) setSelectedId(added.id);
        }
      }
      setNotice(`${op.title}已结束；项目状态：${method === 'model.export' ? '已导出' : projectRef.current?.status}`);
    } catch (err) {
      report(err);
      // Read the real final status, including cancellation and checkpoint state.
      if (operationRef.current?.dispatched) {
        try { acceptProject(normalizeProject(await engineCall('project.open', { path: current.path }))); } catch { /* Keep last known state. */ }
      }
    } finally {
      operationRef.current = null; setOperation(null); setControl(null);
      // Keep a received preview if the engine returned no model (e.g. a paused run).
      if (projectRef.current?.models.length) setPreview(null);
    }
  };
  const importFiles = async (kind: 'images' | 'video' | 'model') => {
    if (operationRef.current || projectBusyRef.current) return;
    try {
      const files = await pickImport(kind); if (!files.length) return;
      if (kind === 'model') {
        if (!projectRef.current && !await createProject(files[0].split(/[\\/]/).at(-1)?.replace(/\.(ply|spz)$/i, '') || '独立模型')) return;
        await run('model.import', { file: files[0] });
      }
      else throw new Error('照片和视频须从 GS Capture 登记，不能导入独立模型项目');
    } catch (err) { report(err); }
  };
  const exportModel = async () => {
    const model = projectRef.current?.models.find(m => m.id === selectedId);
    if (!model || operationRef.current || projectBusyRef.current) return;
    try {
      const file = await pickExport(model.name); if (!file) return;
      const normalizedPath = (value: string) => value.replaceAll('\\', '/').toLowerCase();
      if (projectRef.current?.models.some(source => normalizedPath(source.path) === normalizedPath(file))) throw new Error('导出必须选择新文件，不能覆盖原始模型');
      const extension = file.split('.').at(-1)?.toLowerCase();
      if (extension !== 'ply' && extension !== 'spz') throw new Error('请使用 .ply 或 .spz 文件扩展名选择导出格式');
      await run('model.export', { model_id: model.id, file, format: extension, bake_edits: true });
    } catch (err) { report(err); }
  };
  const controlOperation = async (method: 'pause' | 'cancel' | 'suspend' | 'unpause' | 'checkpoint' | 'stop') => {
    const active = operationRef.current; if (!active?.dispatched || control) return;
    setControl(method);
    try {
      const status = record(await engineCall('status'));
      const job = record(status.active ?? status.job ?? status);
      if (job.path !== active.path || !job.id || !job.generation_id) throw new Error('活动任务身份已变化；控制请求已取消');
      const result = await engineCall(method, { path: active.path, job_id: job.id, generation_id: job.generation_id });
      appendLog('info', `${method === 'pause' ? '暂停' : '取消'}请求返回：${readable(result)}`);
      setNotice(method === 'pause' ? '暂停请求已提交，等待检查点保存并释放 GPU' : `${method} 请求已提交；以引擎实际状态为准`);
      if (['suspend', 'unpause', 'checkpoint'].includes(method)) setControl(null);
    } catch (err) { report(err); setControl(null); }
  };
  return { project, projects, health, connection, connectionError, refresh, error, setError, notice, setNotice, logs, metrics,
    preview, operation, projectBusy, progress, control, selectedId, setSelectedId, dirty, saveState, editModel, updateSettings, openProject,
    createProject, acceptProject, run, importFiles, exportModel, controlOperation, save: () => saveProject().catch(report), report };
}
export type Workbench = ReturnType<typeof useWorkbench>;
