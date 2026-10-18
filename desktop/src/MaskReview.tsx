import { useEffect, useRef, useState } from 'react';
import { modelUrl } from './engine';
import { Field, Check, Json } from './Workflow';
import type { StudioWork } from './useStudio';
import { record } from './types';
import { LatestRequest } from './studioState';

type Point = [number, number];
export function MaskReview({ studio }: { studio: StudioWork }) {
  const [attempt, setAttempt] = useState('primary'), [offset, setOffset] = useState(0), [page, setPage] = useState<Record<string, unknown>>({});
  const [selected, setSelected] = useState<number | null>(null), [media, setMedia] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState('unreviewed'), [note, setNote] = useState(''), [opacity, setOpacity] = useState(0.5), [showMask, setShowMask] = useState(true);
  const [polygonMode, setPolygonMode] = useState<'view' | 'ignore' | 'keep'>('view'), [points, setPoints] = useState<Point[]>([]), [polygons, setPolygons] = useState<{ points: Point[]; ignore: boolean }[]>([]);
  const [confirmed, setConfirmed] = useState(false), [dirty, setDirty] = useState(false), [loading, setLoading] = useState(false);
  const request = useRef(new LatestRequest()), svg = useRef<SVGSVGElement>(null);
  const x = studio.context, scope = `${x.location_id}/${x.scene_id}/${x.run_id}/${attempt}`;
  const load = async () => {
    const token = request.current.next(); setLoading(true); setConfirmed(false);
    try { const value = await studio.read('masks.review', { attempt, offset, limit: 50 }); if (request.current.current(token)) { setPage(value); setSelected(null); setMedia({}); setDirty(false); setPolygons([]); setPoints([]); } }
    catch (e) { if (request.current.current(token)) studio.report(e); }
    finally { if (request.current.current(token)) setLoading(false); }
  };
  useEffect(() => { setPage({}); setMedia({}); setSelected(null); setDirty(false); setConfirmed(false); request.current.invalidate(); if (x.run_id) void load(); return () => request.current.invalidate(); }, [scope, offset]);
  const select = async (id: number) => {
    if (dirty && !window.confirm('当前审核草稿尚未提交，放弃草稿并切换图片？')) return;
    const token = request.current.next(); setSelected(id); setMedia({}); setLoading(true); setPoints([]); setPolygons([]); setConfirmed(false); setDirty(false);
    try {
      const value = await studio.read('masks.read', { attempt, image_id: id });
      if (!request.current.current(token)) return;
      if (value.dataset_identity !== page.dataset_identity) throw new Error('遮罩数据已变化，请刷新审核列表');
      setMedia(value); const review = record(value.review); setStatus(String(review.status ?? 'unreviewed')); setNote(String(review.note ?? ''));
    } catch (e) { if (request.current.current(token)) studio.report(e); }
    finally { if (request.current.current(token)) setLoading(false); }
  };
  const commit = async () => {
    if (!media.mask_sha256 || !media.image_sha256 || !page.dataset_identity || selected === null || points.length) return;
    const result = await studio.operate('masks.commit', { attempt, dataset_identity: page.dataset_identity, edits: [{ image_id: selected, image_sha256: media.image_sha256, mask_sha256: media.mask_sha256, status, note, ...(polygons.length ? { polygons } : {}) }] });
    if (result) { setDirty(false); setPoints([]); setPolygons([]); await load(); }
  };
  const items = Array.isArray(page.items) ? page.items.map(record) : [], total = Number(page.total ?? 0), summary = record(page.summary);
  const counts = record(summary.counts ?? summary), unresolved = ['unreviewed', 'stale', 'false_positive', 'false_negative'].reduce((n, key) => n + Number(counts[key] ?? 0), 0);
  const canFinalize = total > 0 && unresolved === 0 && Number(counts.ok ?? 0) + Number(counts.exclude ?? 0) === total && !dirty && confirmed;
  return <div className="gs-page mask-review"><span className="eyebrow">HUMAN REVIEW</span><h1>原图与遮罩审核</h1><p>每次提交绑定图像、遮罩哈希与数据集身份。修复和 primary 分开固定，不自动接受审核。</p><div className="gs-actions"><Field label="当前尝试"><select value={attempt} disabled={dirty || studio.locked} onChange={e => { setAttempt(e.target.value); setOffset(0); }}><option value="primary">Primary</option><option value="repair">Repair</option></select></Field><span>{x.run_id ?? '请从 Run 页选择一个 Run'}</span><button className="button" disabled={!x.run_id || loading || dirty} onClick={() => void load()}>刷新当前哈希</button></div>
    <div className="gs-card"><Json value={summary} /><div className="gs-actions"><button className="button compact" disabled={offset === 0 || dirty || loading} onClick={() => setOffset(Math.max(0, offset - 50))}>上一页</button><span>{offset + 1}–{Math.min(offset + 50, total)} / {total}</span><button className="button compact" disabled={offset + 50 >= total || dirty || loading} onClick={() => setOffset(offset + 50)}>下一页</button></div></div>
    <div className="gs-review-grid"><div className="gs-image-list">{items.map(item => <button className={selected === Number(item.image_id) ? 'selected' : ''} key={String(item.image_id)} disabled={loading} onClick={() => void select(Number(item.image_id))}><strong>{String(item.source_image ?? item.image_id)}</strong><small>{String(record(item.review).status ?? item.status ?? 'unreviewed')} · {(Number(item.masked_fraction ?? 0) * 100).toFixed(2)}%</small></button>)}</div><section className="gs-card"><div className="gs-actions"><Check label="原图叠加遮罩" checked={showMask} change={setShowMask} /><Field label="叠加透明度"><input type="range" min={0} max={1} step="0.05" value={opacity} onChange={e => setOpacity(Number(e.target.value))} /></Field><Field label="画多边形"><select value={polygonMode} onChange={e => { setPolygonMode(e.target.value as typeof polygonMode); setPoints([]); }}><option value="view">观察</option><option value="ignore">增加忽略区域</option><option value="keep">恢复保留区域</option></select></Field></div>
      {media.image_path && media.mask_path ? <svg ref={svg} className="gs-mask-canvas" viewBox={`0 0 ${media.width} ${media.height}`} onClick={e => {
        if (polygonMode === 'view' || studio.locked) return; const ctm = svg.current?.getScreenCTM(); if (!ctm) return;
        const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
        if (p.x >= 0 && p.y >= 0 && p.x <= Number(media.width) && p.y <= Number(media.height)) { setPoints(v => [...v, [p.x, p.y]]); setDirty(true); }
      }}><defs><filter id="maskInvert"><feComponentTransfer><feFuncR type="table" tableValues="1 0" /><feFuncG type="table" tableValues="1 0" /><feFuncB type="table" tableValues="1 0" /></feComponentTransfer></filter><mask id="reviewMask" maskUnits="userSpaceOnUse" x="0" y="0" width={Number(media.width)} height={Number(media.height)} style={{ maskType: 'luminance' }}><image href={modelUrl(String(media.mask_path))} width={Number(media.width)} height={Number(media.height)} filter={media.mask_semantics === 'black_is_ignored' ? 'url(#maskInvert)' : undefined} /></mask></defs><image href={modelUrl(String(media.image_path))} width={Number(media.width)} height={Number(media.height)} />{showMask && <rect width="100%" height="100%" fill="#ff405d" opacity={opacity} mask="url(#reviewMask)" />}{polygons.map((p, i) => <polygon key={i} points={p.points.map(v => v.join(',')).join(' ')} fill={p.ignore ? '#ff405d66' : '#55d1b666'} stroke={p.ignore ? '#ff405d' : '#55d1b6'} strokeWidth={Number(media.width) / 500} />)}{points.length > 0 && <polyline points={points.map(v => v.join(',')).join(' ')} fill="none" stroke="#b4eb59" strokeWidth={Number(media.width) / 400} />}</svg> : <div className="gs-empty">{loading ? '读取当前哈希与原图…' : '选择一张实际图像开始审核'}</div>}
      {points.length > 0 && <div className="gs-actions"><button className="button" disabled={points.length < 3} onClick={() => { setPolygons(v => [...v, { points, ignore: polygonMode === 'ignore' }]); setPoints([]); }}>闭合多边形</button><button className="button" onClick={() => setPoints([])}>取消当前多边形</button></div>}
      <fieldset disabled={studio.locked || !media.mask_sha256}><Field label="审核决定"><select value={status} onChange={e => { setStatus(e.target.value); setDirty(true); setConfirmed(false); }}><option value="unreviewed">未审核</option><option value="ok">已核验 / 遮罩正确</option><option value="false_positive">误遮挡 · 仍待修正</option><option value="false_negative">漏遮挡 · 仍待修正</option><option value="exclude">排除此图像</option></select></Field><Field label="审核说明"><textarea value={note} onChange={e => { setNote(e.target.value); setDirty(true); }} /></Field><p className="gs-note">遮罩 {String(media.mask_sha256 ?? '—')}<br />原图 {String(media.image_sha256 ?? '—')}</p><button className="button primary" disabled={!dirty || !!points.length} onClick={() => void commit()}>提交当前审核与遮罩修改</button></fieldset>
    </section></div><div className="gs-card"><Check label="我已逐图审核此尝试，确认固定当前遮罩身份" checked={confirmed} change={setConfirmed} disabled={studio.locked || dirty} /><button className="button primary" disabled={studio.locked || !canFinalize} onClick={() => void studio.operate('finalize_masks', { attempt, dataset_identity: page.dataset_identity }).then(v => { if (v) { setConfirmed(false); void load(); } })}>固定 {attempt} 遮罩</button><p className="gs-note">所有图像须明确通过或排除，后端再次验证当前哈希。固定后请从 Run 页显式继续。</p></div>
  </div>;
}
