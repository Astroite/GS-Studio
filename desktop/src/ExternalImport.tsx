import { useRef, useState } from 'react';
import { PathField, Field, Check, Json } from './Workflow';
import { CameraInspection } from './Training';
import type { StudioWork } from './useStudio';
import { LatestRequest } from './studioState';
import { record } from './types';

export function ExternalImport({ studio, goTraining }: { studio: StudioWork; goTraining: () => void }) {
  const [mode, setMode] = useState('colmap');
  const [source, setSource] = useState(''), [images, setImages] = useState(''), [model, setModel] = useState(''), [masks, setMasks] = useState(''), [groups, setGroups] = useState('');
  const [independent, setIndependent] = useState(false), [whiteIgnore, setWhiteIgnore] = useState(false), [importId, setImportId] = useState('');
  const [inspection, setInspection] = useState<Record<string, unknown> | null>(null), [inspecting, setInspecting] = useState(false), [relocation, setRelocation] = useState(''), [mapping, setMapping] = useState('');
  const request = useRef(new LatestRequest());
  const invalidate = () => { request.current.invalidate(); setInspection(null); };
  const change = (setter: (s: string) => void) => (s: string) => { setter(s); invalidate(); };
  const options = () => mode === 'package' ? { source } : ({ ...(images ? { images } : {}), ...(model ? { model } : {}), ...(masks ? { masks } : {}), ...(groups ? { group_file: groups } : {}), independent_images: independent, white_ignore: whiteIgnore });
  const inspect = async () => {
    const token = request.current.next(); setInspection(null); setInspecting(true);
    try { const value = await studio.read('inspect_external_import', options()); if (request.current.current(token)) setInspection(value); }
    catch (e) { if (request.current.current(token)) studio.report(e); }
    finally { setInspecting(false); }
  };
  const errors = Array.isArray(inspection?.errors) ? inspection.errors : [], candidates = Array.isArray(inspection?.model_candidates) ? inspection.model_candidates : Array.isArray(inspection?.models) ? inspection.models : [];
  const canPrepare = !!inspection?.input_identity && !errors.length && inspection?.selection_required !== true;
  return <div className="gs-page"><span className="eyebrow">EXTERNAL COLMAP / SHARED PACKAGE</span><h1>检查并固定外部训练包</h1><p>共享包保留可信分组；普通 COLMAP 必须明确提供分组或确认每张图独立。此入口不代替 Run 遮罩 / 路线 QA。</p><section className="gs-card"><fieldset disabled={studio.locked}><Field label="输入方式"><select value={mode} onChange={e => { setMode(e.target.value); invalidate(); }}><option value="colmap">外部 COLMAP · 先检查，后固定</option><option value="package">已验证共享训练包 · 保留可信分组</option></select></Field>{mode === 'package' && <PathField label="共享训练包目录（dataset.json）" value={source} change={change(setSource)} />}{mode === 'colmap' && <><div className="gs-grid two"><PathField label="Images 目录" value={images} change={change(setImages)} /><PathField label="COLMAP 模型（BIN / TXT 三件套目录）" value={model} change={change(setModel)} /><PathField label="Masks 目录（可选）" value={masks} change={change(setMasks)} /><PathField label="显式分组 JSON（可选）" value={groups} change={change(setGroups)} directory={false} /></div><Check label="我明确确认普通 COLMAP 的每张图片是独立图像，不推断全景 / 时间分组" checked={independent} change={v => { setIndependent(v); invalidate(); }} /><Check label="外部遮罩白色表示忽略（默认白色保留）" checked={whiteIgnore} change={v => { setWhiteIgnore(v); invalidate(); }} /></>}<button className="button primary" disabled={inspecting || (mode === 'package' ? !source : !images)} onClick={() => void inspect()}>{inspecting ? '检查中…' : '检查相机、图像、遮罩、分组与哈希'}</button></fieldset></section>
    {inspection && <><section className="gs-card"><h2>检查结果</h2>{candidates.length > 1 && <Field label="存在多个模型，选择后重新检查"><select value={model} onChange={e => { setModel(e.target.value); invalidate(); }}><option value="">明确选择一个模型</option>{candidates.map((v, i) => <option value={typeof v === 'string' ? v : String(record(v).path)} key={i}>{typeof v === 'string' ? v : String(record(v).path)}</option>)}</select></Field>}<Json value={inspection} /><Field label="新外部导入 ID"><input pattern="[A-Za-z0-9][A-Za-z0-9_.-]*" value={importId} onChange={e => setImportId(e.target.value)} /></Field><button className="button primary" disabled={studio.locked || !canPrepare || !/^[a-z][a-z0-9-]{0,63}$/.test(importId)} onClick={() => void studio.operate('prepare_external_import', { ...options(), import_id: importId, expected_input_identity: inspection.input_identity }).then(value => { if (value) { setInspection(null); studio.setContext({ import_id: importId }); } })}>按已检查身份固定训练包</button></section>{Array.isArray(inspection.cameras) && <CameraInspection cameras={inspection.cameras.map(record)} />}</>}
    <section className="gs-card"><h2>已登记外部项目</h2><Field label="项目"><select value={studio.context.import_id ?? ''} onChange={e => studio.setContext({ import_id: e.target.value })}><option value="">选择外部项目</option>{studio.catalog.external_imports.map(v => <option key={v.id} value={v.id}>{v.id} · {String(v.status)}</option>)}</select></Field><div className="gs-actions"><button className="button" disabled={!studio.context.import_id} onClick={goTraining}>打开此项目实验</button><button className="button" onClick={() => void studio.refresh()}>刷新清单</button></div><PathField label="重新定位后的源根目录" value={relocation} change={setRelocation} /><PathField label="或逐文件路径映射 JSON" value={mapping} change={setMapping} directory={false} /><button className="button" disabled={studio.locked || !studio.context.import_id || !relocation && !mapping || !!relocation && !!mapping} onClick={() => void studio.operate('relink_external_import', { ...(mapping ? { mapping_file: mapping } : { new_root: relocation }) })}>逐文件校验身份并重新定位</button></section>
  </div>;
}
