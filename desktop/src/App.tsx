import { Component, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { Activity, ArrowDownToLine, ArrowUpRight, Box, Camera, Check, ChevronDown, ChevronRight, CircleHelp, Cpu, Eye, EyeOff, FilePlus2, Film, Folder, FolderOpen, HardDrive, ImagePlus, Images, Layers3, LoaderCircle, Pause, Play, Plus, RefreshCw, Save, ScanLine, Sparkles, Square, TriangleAlert, X } from 'lucide-react';
import { ActivityPanel } from './ActivityPanel';
import { Inspector } from './Inspector';
import { Viewport } from './Viewport';
import { projectRenderPolicy } from './renderPolicy';
import { SourceInspection } from './SourceInspection';
import { reconstructionCamera } from './cameras';
import type { CameraView, ReconstructionCamera } from './cameras';
import { desktop, desktopMessage, pickDirectory } from './engine';
import { formatCount, healthMissing, projectCounts, readable, record } from './types';
import type { Workbench } from './useWorkbench';
import './styles.css';

const statusLabels: Record<string, string> = { new: '新建', created: '已创建', empty: '等待素材', imported: '素材已就绪', ready: '就绪', reconstructed: '重建完成', training: '训练中', paused: '已暂停', completed: '已完成', trained: '训练完成', cancelled: '已取消', error: '发生错误', failed: '任务失败', unknown: '状态未知' };
function Brand({ large = false }: { large?: boolean }) { return <div className={`brand ${large ? 'large' : ''}`}><div className="brand-mark"><span /><span /><span /><i /></div><div><strong>GS STUDIO</strong><span>GAUSSIAN WORKSTATION</span></div></div>; }
function ToolButton({ icon, children, onClick, reason, className = '' }: { icon: ReactNode; children: ReactNode; onClick: () => void; reason?: string; className?: string }) {
  return <button className={`tool-button ${className}`} disabled={!!reason} title={reason || (typeof children === 'string' ? children : undefined)} onClick={onClick}>{icon}<span>{children}</span></button>;
}
function Modal({ title, children, onClose, wide = false, busy = false }: { title: string; children: ReactNode; onClose: () => void; wide?: boolean; busy?: boolean }) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    if (!dialogRef.current?.contains(document.activeElement)) dialogRef.current?.querySelector<HTMLElement>('input:not(:disabled), button:not(:disabled)')?.focus();
    return () => { if (previous?.isConnected) previous.focus(); };
  }, []);
  useEffect(() => {
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape' && !busy) onClose(); };
    document.addEventListener('keydown', escape); return () => document.removeEventListener('keydown', escape);
  }, [onClose, busy]);
  return <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget && !busy) onClose(); }}>
    <div ref={dialogRef} className={`modal ${wide ? 'wide' : ''}`} role="dialog" aria-modal="true" aria-label={title} onKeyDown={event => {
      if (event.key === 'Tab') {
        const elements = event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]');
        const first = elements[0], last = elements[elements.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    }}><div className="modal-heading"><strong>{title}</strong><button className="icon-button" aria-label="关闭对话框" title="关闭" disabled={busy} onClick={onClose}><X size={17} /></button></div>{children}</div>
  </div>;
}
function NewProject({ create, close, report }: { create: (name: string, path?: string) => Promise<boolean>; close: () => void; report: (error: unknown) => void }) {
  const [name, setName] = useState(''), [path, setPath] = useState(''), [creating, setCreating] = useState(false);
  return <Modal title="新建项目" onClose={close} busy={creating}><form onSubmit={event => { event.preventDefault(); if (!name.trim()) return; setCreating(true); void create(name, path.trim() || undefined).then(ok => { setCreating(false); if (ok) close(); }); }}>
    <div className="modal-body"><p className="modal-description">为你的下一次重建创建一个独立工作空间。</p><label className="form-label">项目名称<input autoFocus required maxLength={100} placeholder="例如：庭院扫描" value={name} disabled={creating} onChange={event => setName(event.target.value)} /></label><label className="form-label">项目目录 <span className="quiet">可选</span><div className="path-picker"><input placeholder="留空使用引擎默认项目目录" value={path} disabled={creating} onChange={event => setPath(event.target.value)} /><button type="button" className="icon-button" disabled={creating} aria-label="选择项目目录" title="选择项目目录" onClick={() => void pickDirectory().then(selected => { if (selected) setPath(selected); }).catch(report)}><FolderOpen size={16} /></button></div></label><div className="inline-note"><HardDrive size={14} /><span>素材、模型和训练检查点保存在本地项目目录。</span></div></div>
    <div className="modal-footer"><button type="button" className="button" disabled={creating} onClick={close}>取消</button><button className="button primary" type="submit" disabled={!name.trim() || creating}>{creating ? <LoaderCircle size={14} className="spin" /> : <Plus size={14} />}{creating ? '创建中…' : '创建项目'}</button></div>
  </form></Modal>;
}
function ImportModal({ choose, close, disabled }: { choose: (kind: 'images' | 'video' | 'model') => void; close: () => void; disabled: boolean }) {
  return <Modal title="导入到当前项目" onClose={close}><div className="modal-body import-options">{([
    ['images', <Images size={24} />, '多视角照片', '选择一组覆盖同一场景、相互重叠的照片', 'JPG / PNG / WEBP / TIFF'],
    ['video', <Film size={24} />, '场景视频', '按项目中的采样帧率提取真实图像', 'MP4 / MOV / MKV / AVI / WEBM'],
    ['model', <Box size={24} />, '已有高斯模型', '直接查看、编辑并导出 Gaussian Splat', 'PLY / SPZ'],
  ] as const).map(([kind, icon, title, description, format]) => <button key={kind} className="import-option" disabled={disabled} onClick={() => { close(); choose(kind); }}><div className="import-icon">{icon}</div><div><strong>{title}</strong><p>{description}</p><small className="mono">{format}</small></div><ArrowUpRight size={17} /></button>)}</div></Modal>;
}

export default function App({ work, onCapture, onEditor }: { work: Workbench; onCapture: (kind: string) => void; onEditor: () => void }) {
  const [modal, setModal] = useState<'new' | 'import' | 'health' | 'help' | 'assets' | null>(null);
  const [projectMenu, setProjectMenu] = useState(false);
  const [assetsExpanded, setAssetsExpanded] = useState(true), [modelsExpanded, setModelsExpanded] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const [cameraView, setCameraView] = useState<CameraView | null>(null);
  const cameraVersion = useRef(0);
  const reconstruction = record(work.project?.reconstruction);
  const firstCamera = reconstructionCamera(Array.isArray(reconstruction.cameras) ? reconstruction.cameras[0] : null);
  const referenceKey = `${work.project?.path ?? ''}:${String(reconstruction.path ?? '')}:${firstCamera?.name ?? ''}`;
  useEffect(() => { setCameraView(firstCamera ? { camera: firstCamera, version: ++cameraVersion.current } : null); }, [referenceKey]);
  const viewCamera = (camera: ReconstructionCamera) => { setCameraView({ camera, version: ++cameraVersion.current }); setModal(null); };
  const counts = projectCounts(work.project), missing = healthMissing(work.health);
  const ready = work.connection === 'ready';
  const projectReason = !desktop ? desktopMessage : !ready ? '请先连接本地引擎' : work.projectBusy ? '正在打开或创建项目' : work.operation ? '请等待当前任务结束' : '';
  const operationReason = projectReason || (!work.project ? '请先新建或打开项目' : '');
  const selected = work.project?.models.find(model => model.id === work.selectedId);
  const isTraining = work.operation?.method === 'train' || work.operation?.method === 'resume';
  const gpu = work.health?.gpu;
  const gpuLabel = typeof gpu === 'string' ? gpu : String(record(gpu).name ?? record(gpu).device_name ?? (Array.isArray(gpu) ? record(gpu[0]).name ?? '' : ''));
  const runtimeLabel = !desktop ? '浏览器预览' : work.connection === 'connecting' ? '连接引擎中' : work.connection === 'error' ? '引擎未连接' : missing.length ? `缺少 ${missing.length} 项依赖` : '本地引擎已连接';
  useEffect(() => {
    if (!work.operation) { setElapsed(0); return; }
    const started = work.operation.started;
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [work.operation]);
  useEffect(() => { if (!work.notice) return; const timer = window.setTimeout(() => work.setNotice(''), 6500); return () => window.clearTimeout(timer); }, [work.notice, work.setNotice]);
  useEffect(() => {
    if (!desktop) return;
    const shortcut = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); if (work.project && !work.operation && !work.projectBusy) void work.save(); }
    };
    window.addEventListener('keydown', shortcut); return () => window.removeEventListener('keydown', shortcut);
  }, [work.project, work.operation, work.projectBusy, work.save]);

  return <div className="app-shell">
    <header className="app-header"><Brand /><div className="header-divider" /><button className="project-switcher" disabled={!!projectReason} title={projectReason || '切换项目'} onClick={() => setProjectMenu(value => !value)}><Folder size={15} /><span>{work.project?.name ?? '未打开项目'}</span><ChevronDown size={12} /></button>
      {work.project && <span className="project-status"><i />{statusLabels[work.project.status] ?? work.project.status}</span>}
      <div className="header-spacer" /><span className="local-badge"><HardDrive size={12} />本地工作空间</span>
      <button className={`runtime-chip ${ready && !missing.length ? 'online' : 'offline'}`} onClick={() => setModal('health')} title="查看真实运行环境"><i className={work.connection === 'connecting' ? 'pulse' : ''} />{runtimeLabel}<ChevronRight size={12} /></button>
      <button className="icon-button help-button" aria-label="使用帮助" title="使用帮助" onClick={() => setModal('help')}><CircleHelp size={17} /></button>
    </header>
    {projectMenu && <><div className="popover-dismiss" onClick={() => setProjectMenu(false)} /><div className="project-popover"><div className="popover-heading">最近项目</div>{work.projects.length ? work.projects.map(project => <button key={project.path} title={project.path} onClick={() => { setProjectMenu(false); void work.openProject(project.path); }}><Folder size={15} /><span><strong>{project.name}</strong><small>{project.path}</small></span>{work.project?.path === project.path && <Check size={13} />}</button>) : <p>暂无项目</p>}<div className="popover-divider" /><button onClick={() => { setProjectMenu(false); void work.openProject(); }}><FolderOpen size={15} />打开项目文件夹</button><button onClick={() => { setProjectMenu(false); setModal('new'); }}><Plus size={15} />新建项目</button></div></>}
    <nav className="main-toolbar" aria-label="项目和训练操作"><div className="toolbar-group">
      <ToolButton icon={<FilePlus2 size={16} />} reason={projectReason} onClick={() => setModal('new')}>新建</ToolButton>
      <ToolButton icon={<FolderOpen size={16} />} reason={projectReason} onClick={() => void work.openProject()}>打开</ToolButton>
      <ToolButton icon={<ImagePlus size={16} />} reason={projectReason} onClick={() => setModal('import')}>导入<ChevronDown size={11} /></ToolButton>
    </div><span className="toolbar-separator" /><div className="toolbar-group">
      <ToolButton icon={<ScanLine size={16} />} reason={operationReason || (counts.images === 0 ? '请先导入照片或视频' : '')} onClick={() => void work.run('reconstruct')}>重建</ToolButton>
      <ToolButton className="train-button" icon={<Play size={14} fill="currentColor" />} reason={operationReason} onClick={() => void work.run('train', { steps: work.project?.settings.steps, max_splats: work.project?.settings.max_splats, resume: false })}>开始训练</ToolButton>
      <ToolButton icon={<Pause size={15} />} reason={!isTraining ? '仅可暂停正在运行的训练' : !work.operation?.dispatched ? '正在保存项目，尚未启动任务' : work.control ? '正在等待控制请求完成' : ''} onClick={() => void work.controlOperation('pause')}>保存暂停</ToolButton>
      <ToolButton icon={<Play size={14} />} reason={operationReason} onClick={() => void work.run('resume', { steps: work.project?.settings.steps })}>恢复</ToolButton>
      <ToolButton icon={<Square size={13} />} reason={!work.operation ? '没有运行中的任务' : !work.operation.dispatched ? '正在保存项目，尚未启动任务' : work.control ? '正在等待控制请求完成' : ''} onClick={() => void work.controlOperation('cancel')}>取消</ToolButton>
    </div>    <details className="gs-local-controls"><summary>高级控制</summary><div>{[['suspend', '热挂起 · 保留 GPU'], ['unpause', '解除热挂起'], ['checkpoint', '保存检查点'], ['stop', '提前结束并保存']].map(([method, title]) => <button key={method} className="button compact" disabled={!isTraining || !!work.control} onClick={() => void work.controlOperation(method as 'suspend' | 'unpause' | 'checkpoint' | 'stop')}>{title}</button>)}</div></details><div className="toolbar-spacer" /><div className="toolbar-group">
      <ToolButton icon={<Save size={15} />} reason={operationReason} onClick={() => void work.save()}>保存</ToolButton>
      <ToolButton className="export-button" icon={<ArrowDownToLine size={15} />} reason={operationReason || (!selected ? '请先选择已保存的模型' : '')} onClick={() => void work.exportModel()}>导出模型</ToolButton>
    </div></nav>
    <div className="workspace">
      <aside className="assets-panel panel"><div className="panel-heading"><Layers3 size={14} /><span>项目资源</span><button className="icon-button" title={operationReason || '添加素材或模型'} disabled={!!operationReason} aria-label="添加素材或模型" onClick={() => setModal('import')}><Plus size={15} /></button></div>
        <div className="project-card"><div className="project-folder"><FolderOpen size={23} strokeWidth={1.4} /></div><div><strong title={work.project?.name}>{work.project?.name ?? '你的三维工作空间'}</strong><span>{work.project ? '本地项目' : '创建项目，开始一次新的重建'}</span></div></div>
        {!work.project && <div className="sidebar-start"><button className="button compact" disabled={!!projectReason} title={projectReason || '新建项目'} onClick={() => setModal('new')}><Plus size={13} />新建项目</button><button className="text-button" disabled={!!projectReason} title={projectReason || '打开项目文件夹'} onClick={() => void work.openProject()}>打开已有项目<ArrowUpRight size={12} /></button></div>}
        <div className="asset-tree"><button className="tree-heading" onClick={() => setAssetsExpanded(v => !v)}><ChevronDown size={12} className={assetsExpanded ? '' : 'chevron-closed'} /><span>源素材</span><span className="count-tag">{counts.images ?? '—'}</span></button>
          {assetsExpanded && <div className="source-items"><div className="source-item"><Images size={14} /><span>图像</span><span className="mono">{formatCount(counts.images)}</span></div><div className="source-item"><Camera size={14} /><span>已注册相机</span><span className="mono">{formatCount(counts.cameras)}</span></div>{counts.images === undefined && <p>导入后显示引擎返回的素材数量</p>}<button className="text-button source-inspect-button" disabled={!work.project} title={work.project ? '查看实际图像和已求解相机参数' : '打开项目后可检查素材'} onClick={() => setModal('assets')}>检查素材与相机<ChevronRight size={12} /></button></div>}
          <button className="tree-heading" onClick={() => setModelsExpanded(v => !v)}><ChevronDown size={12} className={modelsExpanded ? '' : 'chevron-closed'} /><span>场景模型</span><span className="count-tag">{work.project?.models.length ?? 0}</span></button>
          {modelsExpanded && <div className="model-list">{work.project?.models.map(model => <div key={model.id} className={`model-item ${work.selectedId === model.id ? 'selected' : ''}`}><button className="model-select" title={model.path} onClick={() => work.setSelectedId(model.id)}><Box size={15} /><span><strong>{model.name}</strong><small className="mono">{formatCount(model.count)} splats</small></span></button><button className="icon-button visibility" aria-label={`${model.visible ? '隐藏' : '显示'} ${model.name}`} title={work.operation ? '任务运行时不可修改可见性' : model.visible ? '隐藏模型' : '显示模型'} disabled={!!work.operation} onClick={() => work.editModel(model.id, m => ({ ...m, visible: !m.visible }))}>{model.visible ? <Eye size={13} /> : <EyeOff size={13} />}</button></div>)}
            {work.preview && <div className={`model-item preview-item ${work.selectedId === '__preview__' ? 'selected' : ''}`}><button className="model-select" onClick={() => work.setSelectedId('__preview__')}><Activity size={15} /><span><strong>训练实时预览</strong><small className="mono">STEP {work.preview.previewStep} · {formatCount(work.preview.count)}</small></span></button><i className="live-dot" /></div>}
            {!work.project?.models.length && !work.preview && <div className="empty-models"><Box size={27} strokeWidth={1} /><span>场景尚未添加模型</span><small>训练完成或导入 PLY / SPZ 后显示</small></div>}
          </div>}
        </div>
        <div className="sidebar-bottom"><div className="workflow-heading"><span>工作流程</span><span className="small-label">01 — 03</span></div><div className="workflow-step"><span>01</span><div><strong>导入素材</strong><p>多视角照片或场景视频</p></div><ImagePlus size={15} /></div><div className="workflow-step"><span>02</span><div><strong>重建与训练</strong><p>恢复相机，优化三维高斯</p></div><Sparkles size={15} /></div><div className="workflow-step"><span>03</span><div><strong>编辑与导出</strong><p>调整、裁剪并保存模型</p></div><ArrowUpRight size={15} /></div><div className="sidebar-footnote"><HardDrive size={11} />素材与计算均在本机</div></div>
      </aside>
      <main className="center-workspace">
        {(work.connection !== 'ready' || missing.length > 0) && <div className={`connection-banner ${work.connection === 'connecting' ? 'connecting' : ''}`}><span>{work.connection === 'connecting' ? <LoaderCircle size={14} className="spin" /> : <TriangleAlert size={14} />}{!desktop ? desktopMessage : work.connection === 'connecting' ? '正在连接本地 Python 引擎…' : missing.length && ready ? `运行环境缺少依赖：${missing.join('、')}` : '本地引擎未连接，查看运行环境以了解原因'}</span><button onClick={() => setModal('health')}>{!desktop ? '运行说明' : '查看详情'}<ChevronRight size={12} /></button></div>}
        <Viewport models={work.preview ? [work.preview] : work.project?.models ?? []} selectedId={work.selectedId} scope={work.project?.path ?? ''} desktop={desktop} canImport={!operationReason} onImport={() => setModal('import')} reference={cameraView} renderPolicy={projectRenderPolicy(work.project)} />
        <ActivityPanel work={work} />
      </main>
      <div className="gs-base-inspector"><button className="button primary" disabled={!selected || !!work.operation} onClick={onEditor}>打开原生选择 / 编辑 / 历史</button><Inspector work={work} settingsOnly /></div>
    </div>
    <footer className="status-bar"><div><span className={`status-light ${ready ? 'online' : ''}`} /><span>{runtimeLabel}</span>{gpuLabel && <><span className="status-divider" /><Cpu size={11} /><span title={gpuLabel}>{gpuLabel}</span></>}</div><div>{work.operation ? <><LoaderCircle size={11} className="spin" /><span>{work.control === 'pause' ? '等待检查点保存' : work.control === 'cancel' ? '等待取消完成' : work.progress?.stage ?? work.operation.title}</span><span className="mono">{Math.floor(elapsed / 60).toString().padStart(2, '0')}:{(elapsed % 60).toString().padStart(2, '0')}</span></> : <><span className="mono">GS STUDIO</span><span className="status-divider" /><span className="mono">v0.3.0-rc.1</span></>}</div></footer>
    {work.notice && <div className="toast" role="status"><Check size={15} /><span>{work.notice}</span><button className="icon-button" title="关闭通知" aria-label="关闭通知" onClick={() => work.setNotice('')}><X size={13} /></button></div>}
    {modal === 'new' && <NewProject create={work.createProject} close={() => setModal(null)} report={work.report} />}
    {modal === 'import' && <ImportModal choose={kind => { if (kind === 'model') void work.importFiles(kind); else onCapture(kind); }} disabled={!!projectReason} close={() => setModal(null)} />}
    {modal === 'assets' && work.project && <Modal title="素材与相机检查" wide onClose={() => setModal(null)}><SourceInspection project={work.project} onViewCamera={viewCamera} /></Modal>}
    {modal === 'health' && <Modal title="本地运行环境" wide onClose={() => setModal(null)}><div className="modal-body"><div className="runtime-summary"><div className={`health-icon ${ready ? 'online' : ''}`}><Cpu size={25} /></div><div><strong>{runtimeLabel}</strong><p>{!desktop ? desktopMessage : ready ? '以下信息由当前运行的 Python 引擎返回' : '应用使用运行环境配置中的独立 Python'}</p></div><button className="button compact" disabled={!desktop || work.connection === 'connecting'} title={!desktop ? desktopMessage : '重新连接并检查健康状态'} onClick={() => void work.refresh()}><RefreshCw size={13} className={work.connection === 'connecting' ? 'spin' : ''} />重新检查</button></div>
      {work.connectionError && <div className="error-details">{work.connectionError}</div>}{missing.length > 0 && <div className="missing-deps"><strong>引擎报告缺失的依赖</strong>{missing.map((item, i) => <p key={i}><TriangleAlert size={13} />{item}</p>)}</div>}
      {work.health && <details className="health-details" open><summary>完整运行环境响应</summary><pre>{readable(work.health)}</pre></details>}
      <div className="runtime-paths"><span>运行环境配置</span><code>GSSTUDIO_RUNTIME</code><small>程序旁 runtime/runtime.json；可通过环境变量或用户运行配置覆盖</small><span>引擎模块目录</span><code>runtime.json: engine_root / pipeline_root</code><small>默认程序旁 engine/；开发时使用仓库 engine/</small></div></div></Modal>}
    {modal === 'help' && <Modal title="工作空间指南" onClose={() => setModal(null)}><div className="modal-body help-content"><Brand large /><p>在本机从照片或视频创建、编辑和导出真实的 Gaussian Splatting 场景。</p><ol><li><strong>创建项目并导入素材</strong><span>围绕静态场景拍摄，保持清晰和相邻视角重叠。</span></li><li><strong>重建相机，然后开始训练</strong><span>在「训练设置」调整步数与点数上限。暂停后可从完整检查点恢复。</span></li><li><strong>调整模型并导出</strong><span>编辑自动保存；导出文件扩展名决定 PLY 或 SPZ 格式。</span></li></ol><div className="inline-note"><Save size={14} /><span>Ctrl + S 保存当前项目编辑</span></div></div></Modal>}
    {work.error && <Modal title="操作未完成" wide onClose={() => work.setError(null)}><div className="modal-body"><div className="error-title"><TriangleAlert size={20} /><span>本地引擎或文件操作返回错误</span></div><pre className="error-details">{work.error}</pre><p className="modal-description">详细信息已记录在运行日志中。检查运行环境和项目素材后可重试。</p></div><div className="modal-footer"><button className="button primary" autoFocus onClick={() => work.setError(null)}>知道了</button></div></Modal>}
  </div>;
}

export class AppErrorBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state: { error: string | null } = { error: null };
  static getDerivedStateFromError(error: unknown) { return { error: readable(error) }; }
  render() {
    if (this.state.error) return <div className="fatal-error"><Brand large /><TriangleAlert size={30} /><h1>界面遇到错误</h1><pre>{this.state.error}</pre><button className="button primary" onClick={() => window.location.reload()}><RefreshCw size={14} />重新打开工作空间</button></div>;
    return this.props.children;
  }
}
