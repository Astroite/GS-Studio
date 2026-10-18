import { useEffect, useRef, useState } from 'react';
import { Activity, ChevronDown, ChevronUp, ScrollText, Terminal } from 'lucide-react';
import { formatCount } from './types';
import type { Metric } from './types';
import type { Workbench } from './useWorkbench';

export function LossChart({ metrics }: { metrics: Metric[] }) {
  const width = 600, height = 112, pad = 10;
  const low = Math.min(...metrics.map(m => m.loss)), high = Math.max(...metrics.map(m => m.loss));
  const start = metrics[0]?.step ?? 0, end = metrics.at(-1)?.step ?? 1;
  const points = metrics.map(m => `${pad + (m.step - start) / Math.max(1, end - start) * (width - pad * 2)},${height - pad - (m.loss - low) / Math.max(1e-8, high - low) * (height - pad * 2)}`).join(' ');
  return <div className="loss-chart"><div className="chart-labels mono"><span>{metrics.length ? high.toPrecision(3) : 'LOSS'}</span><span>{metrics.length ? low.toPrecision(3) : '—'}</span></div>
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label={metrics.length ? `训练损失曲线，共 ${metrics.length} 个实测采样` : '暂无训练损失数据'}>
      <defs><linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#45c6ad" stopOpacity="0.2" /><stop offset="100%" stopColor="#45c6ad" stopOpacity="0" /></linearGradient></defs>
      {[0.25, 0.5, 0.75].map(y => <line key={y} x1={0} y1={height * y} x2={width} y2={height * y} stroke="#263033" strokeWidth="0.6" strokeDasharray="3 5" />)}
      {metrics.length > 1 && <><polygon points={`${pad},${height} ${points} ${width - pad},${height}`} fill="url(#lossFill)" /><polyline points={points} fill="none" stroke="#55d1b6" strokeWidth="1.6" vectorEffect="non-scaling-stroke" /></>}
      {metrics.length === 1 && <circle cx={pad} cy={height - pad} r={3} fill="#55d1b6" />}
    </svg>
    {!metrics.length && <div className="chart-empty">训练开始后显示实时损失曲线</div>}
    <div className="chart-axis mono"><span>{metrics.length ? `STEP ${start}` : 'STEP —'}</span><span>{metrics.length ? end : '—'}</span></div>
  </div>;
}
export function ActivityPanel({ work }: { work: Workbench }) {
  const [tab, setTab] = useState<'metrics' | 'logs'>('metrics'), [collapsed, setCollapsed] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  const metric = work.metrics.at(-1);
  useEffect(() => { if (tab === 'logs') end.current?.scrollIntoView({ behavior: 'instant', block: 'end' }); }, [work.logs, tab]);
  const progress = work.progress?.progress;
  return <section className={`activity-panel ${collapsed ? 'collapsed' : ''}`}>
    <div className="activity-heading"><div className="activity-tabs"><button className={tab === 'metrics' ? 'selected' : ''} onClick={() => { setTab('metrics'); setCollapsed(false); }}><Activity size={13} />训练监测</button><button className={tab === 'logs' ? 'selected' : ''} onClick={() => { setTab('logs'); setCollapsed(false); }}><Terminal size={13} />运行日志<span className="count-tag">{work.logs.length}</span></button></div>
      <span className={`operation-label ${work.operation ? 'running' : ''}`}><i />{work.operation?.title ?? '无运行任务'}</span><button className="icon-button" title={collapsed ? '展开监测面板' : '收起监测面板'} aria-label={collapsed ? '展开监测面板' : '收起监测面板'} onClick={() => setCollapsed(v => !v)}>{collapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</button></div>
    {!collapsed && <>
      {work.operation && <div className="progress-strip"><span>{work.progress?.stage ?? work.operation.title}</span><div className="progress-track">{progress !== undefined ? <i style={{ width: `${progress * 100}%` }} /> : <i className="indeterminate" />}</div><span className="mono">{progress !== undefined ? `${(progress * 100).toFixed(1)}%` : '等待引擎'}</span><span className="progress-message" title={work.progress?.message}>{work.progress?.message ?? '等待引擎返回进度事件'}</span></div>}
      {tab === 'metrics' ? <div className="metrics-content"><div className="chart-section"><div className="chart-heading"><span><i />训练损失 <span className="quiet">Loss</span></span><strong className="mono">{metric ? metric.loss.toPrecision(4) : '—'}</strong></div><LossChart metrics={work.metrics} /></div>
        <div className="metric-grid"><div><span>当前步数</span><strong className="mono">{metric ? formatCount(metric.step) : '—'}<small>{work.project ? ` / ${formatCount(work.project.settings.steps)}` : ''}</small></strong></div><div><span>高斯点数</span><strong className="mono">{formatCount(metric?.splats)}</strong></div><div><span title="当前采样训练图的 PSNR，独立留出评估见评估报告">训练 PSNR <small>dB</small></span><strong className="mono">{metric?.psnr?.toFixed(2) ?? '—'}</strong></div><div><span title="torch.max_memory_allocated()，仅训练张量峰值；不包含 CUDA context、驱动和其他进程">张量峰值 <small>MiB</small></span><strong className="mono">{metric?.gpu_memory_mb !== undefined ? formatCount(Math.round(metric.gpu_memory_mb)) : '—'}</strong></div></div></div>
        : <div className="logs-scroll" role="log" aria-label="真实引擎运行日志">{work.logs.length ? work.logs.map(log => <div className={`log-line ${log.level.toLowerCase()}`} key={log.id}><time className="mono">{log.time}</time><span className="log-level mono">{log.level.toUpperCase()}</span><span>{log.message}</span></div>) : <div className="logs-empty"><ScrollText size={22} /><span>等待本地引擎日志</span></div>}<div ref={end} /></div>}
    </>}
  </section>;
}
