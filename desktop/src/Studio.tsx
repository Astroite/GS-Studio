import { useEffect, useState } from 'react';
import { Box, Camera, FolderOpen, Layers3, LoaderCircle, RefreshCw, ScanLine, Scissors, ShieldCheck, Sparkles, TriangleAlert } from 'lucide-react';
import App from './App';
import { useWorkbench } from './useWorkbench';
import { useStudio } from './useStudio';
import { Workflow, RunReview, SelectSharp } from './Workflow';
import { MaskReview } from './MaskReview';
import { ExternalImport } from './ExternalImport';
import { Training } from './Training';
import { NativeEditor } from './NativeEditor';
import { desktop, engineCall, requestStageStop } from './engine';
import type { Row } from './studioState';
import { normalizeProject, record } from './types';
import './studio.css';
import { Catalog } from './Catalog';

export default function Studio() {
  const work = useWorkbench();
  const [editorBusy, setEditorBusy] = useState(false), [editorDirty, setEditorDirty] = useState(false);
  const studio = useStudio(!!work.operation || work.projectBusy || editorBusy || editorDirty);
  const [tab, changeTab] = useState('capture'), [kind, setKind] = useState(''), [showEvents, setShowEvents] = useState(false);
  const [controlNotice, setControlNotice] = useState('');
  useEffect(() => setControlNotice(''), [studio.busy]);
  const setTab = (next: string) => { if (tab === 'editor' && next !== 'editor' && (editorBusy || editorDirty)) { studio.setError('请先确认或取消当前编辑预览，等待原生操作完成后再切换工作区'); return; } changeTab(next); };
  const x = studio.context;
  const [focus, setFocus] = useState<{ segment?: string; output?: string }>({});
  const chooseTraining = (row: Row, segment?: string, output?: string) => { if (row.import_id) studio.setContext({ import_id: String(row.import_id) }); else studio.setContext({ location_id: String(row.location_id), scene_id: String(row.scene_id), run_id: row.id, capture_id: String(row.capture_id ?? '') }); setFocus({ segment, output }); setTab('training'); };
  const capture = (kind: string) => { setKind(kind === 'images' ? 'perspective_images' : 'perspective_video'); setTab('capture'); };
  const openModel = async (value: Record<string, unknown>) => {
    const path = value.project_path;
    if (typeof path !== 'string' || !value.model_sha256) throw new Error('模型打开响应缺少已验证的项目 / 模型身份');
    const project = normalizeProject(await engineCall('project.open', { path })); work.acceptProject(project); setTab('editor');
  };
  const chooseRun = (row: Row) => { studio.setContext({ location_id: String(row.location_id), scene_id: String(row.scene_id), capture_id: String(row.capture_id ?? ''), run_id: row.id }); setTab('run'); };
  const nav = [['capture', '素材与组织', Camera], ['run', 'Run / QA', ScanLine], ['masks', '遮罩审核', ShieldCheck], ['training', '训练实验', Sparkles], ['external', '外部 COLMAP', FolderOpen], ['sharp', '独立选帧', Scissors], ['models', '模型工作台', Box], ['editor', '原生编辑', Layers3]] as const;
  const cancelJob = async () => {
    try { const status = record(await engineCall('status')); const job = record(status.active ?? status.job ?? status); if (!job.id || !job.generation_id) throw new Error('没有可取消的活动任务'); await engineCall('cancel', { job_id: job.id, generation_id: job.generation_id, path: job.path }); } catch (e) { studio.report(e); }
  };
  const stopStage = async () => {
    try { const status = record(await engineCall('status')), active = record(status.active); if (typeof active.id !== 'string' || typeof active.generation_id !== 'string') throw new Error('当前没有活动 Run'); await requestStageStop(active.id, active.generation_id); setControlNotice('已请求当前阶段后停止；等待清单确认'); } catch (e) { studio.report(e); }
  };
  return <div className="gs-shell"><header className="gs-topbar"><div className="brand"><div className="brand-mark"><span /><span /><span /><i /></div><div><strong>GS STUDIO</strong><span>GAUSSIAN WORKSTATION</span></div></div><span className="gs-root mono" title={studio.catalog.root}>{studio.catalog.root}</span><div className="header-spacer" /><span className="local-badge">{desktop ? '本地计算 · 原始素材只读' : '浏览器预览 · 未连接本地引擎'}</span><button className="icon-button" title="刷新真实项目清单" onClick={() => void studio.refresh()}><RefreshCw size={16} /></button></header>
    <nav className="gs-nav" aria-label="GS 工作区">{nav.map(([key, title, Icon]) => <button className={tab === key ? 'selected' : ''} key={key} onClick={() => setTab(key)}><Icon size={15} />{title}</button>)}</nav>
    <div className="gs-layout"><Catalog studio={studio} work={work} go={setTab} chooseRun={chooseRun} chooseTraining={chooseTraining} />
      <main className="gs-content">{(studio.error || work.error) && <div className="gs-error" role="alert"><TriangleAlert size={16} /><pre>{studio.error || work.error}</pre><button className="button compact" onClick={() => { studio.setError(''); work.setError(null); }}>关闭</button></div>}
        {tab === 'capture' && <Workflow key={kind} studio={studio} initialKind={kind} />}{tab === 'run' && <RunReview key={`${x.location_id}/${x.scene_id}/${x.run_id}`} studio={studio} goMasks={() => setTab('masks')} goTraining={() => setTab('training')} />}{tab === 'masks' && <MaskReview studio={studio} />}{tab === 'external' && <ExternalImport studio={studio} goTraining={() => setTab('training')} />}{tab === 'sharp' && <SelectSharp studio={studio} />}{tab === 'training' && <Training studio={studio} openModel={openModel} focus={focus} />}{tab === 'models' && <App work={{ ...work, projectBusy: work.projectBusy || !!studio.busy }} onCapture={capture} onEditor={() => setTab('editor')} />}{tab === 'editor' && <NativeEditor work={work} blocked={!!studio.busy} onBusyChange={setEditorBusy} onDirtyChange={setEditorDirty} />}
      </main></div>
    <footer className="gs-footer"><span>{studio.busy ? <LoaderCircle size={12} className="spin" /> : <i className="status-light" />}{studio.busy ? `${studio.busy} · 等待结构化状态` : '就绪 · 以清单与哈希证据为准'}</span>{['create_run', 'resume_run'].includes(studio.busy ?? '') && <button onClick={() => void stopStage()}>当前阶段后停止</button>}{studio.busy && <button onClick={() => void cancelJob()}>取消活动任务</button>}{studio.busy && controlNotice && <span role="status">{controlNotice}</span>}<div className="header-spacer" /><button onClick={() => setShowEvents(v => !v)}>活动 / 实测指标 / 日志 ({studio.events.length})</button></footer>
    {showEvents && <section className="gs-event-drawer" role="log"><div className="gs-actions"><h2>活动与训练实测数据</h2><button className="button compact" onClick={() => setShowEvents(false)}>收起</button></div>{studio.events.slice(-150).map((e, i) => <pre key={i}>{JSON.stringify(e, null, 2)}</pre>)}</section>}
  </div>;
}
