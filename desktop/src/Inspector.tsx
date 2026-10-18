import { useEffect, useState } from 'react';
import { Box, ChevronDown, Crop, Info, RotateCcw, Settings2, SlidersHorizontal, Sparkles } from 'lucide-react';
import { formatCount, projectCounts, record } from './types';
import type { Model, Vec3 } from './types';
import type { Workbench } from './useWorkbench';

interface NumberFieldProps { value?: number; onCommit?: (value: number) => void; disabled?: boolean; label: string; min?: number; max?: number; step?: number; precision?: number; integer?: boolean }
export function NumberField({ value, onCommit, disabled, label, min, max, step = 0.1, precision = 3, integer }: NumberFieldProps) {
  const format = (v: number | undefined) => v === undefined ? '' : String(Number(v.toFixed(precision)));
  const [draft, setDraft] = useState(format(value)), [invalid, setInvalid] = useState(false);
  useEffect(() => { setDraft(format(value)); setInvalid(false); }, [value]); // Preserve a focused draft until its committed value changes.
  const commit = () => {
    const number = Number(draft);
    if (!draft.trim() || !Number.isFinite(number) || min !== undefined && number < min || max !== undefined && number > max || integer && !Number.isInteger(number)) {
      setInvalid(true); return;
    }
    setInvalid(false); if (number !== value) onCommit?.(number);
  };
  return <input className={`number-input mono ${invalid ? 'invalid' : ''}`} type="number" aria-label={label} aria-invalid={invalid} placeholder="—"
    title={disabled ? '选择模型或打开项目后可编辑' : invalid ? `请输入${integer ? '整数' : '数值'}${min !== undefined ? ` ≥ ${min}` : ''}${max !== undefined ? ` ≤ ${max}` : ''}` : label}
    value={draft} disabled={disabled} min={min} max={max} step={step} onChange={event => { setDraft(event.target.value); setInvalid(false); }} onBlur={commit}
    onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); if (event.key === 'Escape') { setDraft(format(value)); setInvalid(false); } }} />;
}
function VectorFields({ label, value, disabled, onCommit, min, max }: { label: string; value?: Vec3; disabled: boolean; onCommit: (value: Vec3) => void; min?: Vec3; max?: Vec3 }) {
  return <div className="vector-row"><span className="field-label">{label}</span><div className="vector-fields">{(['X', 'Y', 'Z'] as const).map((axis, index) => <label key={axis} className={`axis-field axis-${axis.toLowerCase()}`}><span>{axis}</span><NumberField label={`${label} ${axis}`} value={value?.[index]} disabled={disabled} min={min?.[index]} max={max?.[index]} onCommit={number => { if (value) { const next: Vec3 = [...value]; next[index] = number; onCommit(next); } }} /></label>)}</div></div>;
}
function SectionTitle({ children, icon }: { children: React.ReactNode; icon?: React.ReactNode }) { return <div className="section-title">{icon}<span>{children}</span><ChevronDown size={12} /></div>; }
export function Inspector({ work, settingsOnly = false }: { work: Workbench; settingsOnly?: boolean }) {
  const [tab, setTab] = useState<'model' | 'settings'>(settingsOnly ? 'settings' : 'model');
  const [cleanRadius, setCleanRadius] = useState(0.15);
  const selected = work.project?.models.find(m => m.id === work.selectedId);
  const model = work.selectedId === '__preview__' ? work.preview : selected;
  const editable = !!selected && !work.operation && !work.projectBusy;
  const modify = (change: (model: Model) => Model) => { if (selected) work.editModel(selected.id, change); };
  const transform = <K extends keyof Model['transform']>(key: K, value: Model['transform'][K]) => modify(m => ({ ...m, transform: { ...m.transform, [key]: value } }));
  const crop = <K extends keyof Model['crop']>(key: K, value: Model['crop'][K]) => modify(m => ({ ...m, crop: { ...m.crop, [key]: value } }));
  const settings = work.project?.settings;
  const hasRecipeOverrides = Object.keys(record(settings?.training_recipe_overrides)).length > 0;
  const disabledSettings = !work.project || !!work.operation || work.projectBusy;
  const canClean = editable && (projectCounts(work.project).cameras ?? 0) > 0;
  return <aside className="inspector panel">
    <div className="panel-heading"><SlidersHorizontal size={14} /><span>检查器</span><span className="small-label">INSPECTOR</span></div>
    <div className="tab-strip">{!settingsOnly && <button className={tab === 'model' ? 'selected' : ''} onClick={() => setTab('model')}>模型属性</button>}<button className={tab === 'settings' ? 'selected' : ''} onClick={() => setTab('settings')}>原生实验设置</button></div>
    <div className="inspector-scroll">
      {tab === 'model' ? <>
        <div className="selection-card"><div className="selection-icon"><Box size={19} /></div><div><strong>{model?.name ?? '未选择模型'}</strong><span>{model ? `${formatCount(model.count)} 个高斯点${model.previewStep !== undefined ? ' · 只读预览' : ''}` : '在场景列表中选择一个模型'}</span></div></div>
        <SectionTitle icon={<Settings2 size={13} />}>变换</SectionTitle>
        <div className="property-section">
          <VectorFields label="位置" value={model?.transform.position} disabled={!editable} onCommit={value => transform('position', value)} />
          <VectorFields label="旋转 °" value={model?.transform.rotation.map(v => v * 180 / Math.PI) as Vec3 | undefined} disabled={!editable} onCommit={value => transform('rotation', value.map(v => v * Math.PI / 180) as Vec3)} />
          <div className="single-row"><span className="field-label">统一缩放</span><NumberField label="统一缩放（必须大于零）" value={model?.transform.scale} min={0.0001} step={0.01} precision={4} disabled={!editable} onCommit={value => transform('scale', value)} /></div>
          <button className="text-button reset-button" disabled={!editable} title={editable ? '恢复位置、旋转和缩放默认值' : '选择可编辑模型后可用'} onClick={() => modify(m => ({ ...m, transform: { position: [0, 0, 0], rotation: [0, 0, 0], scale: 1 } }))}><RotateCcw size={11} />重置变换</button>
        </div>
        <SectionTitle icon={<Crop size={13} />}>空间裁剪</SectionTitle>
        <div className="property-section">
          <div className="single-row"><span>启用包围盒</span><button role="switch" aria-checked={model?.crop.enabled ?? false} aria-label="启用空间裁剪" className={`toggle ${model?.crop.enabled ? 'on' : ''}`} disabled={!editable} title={editable ? '实时裁剪模型坐标内的高斯点' : '选择可编辑模型后可用'} onClick={() => crop('enabled', !model?.crop.enabled)}><i /></button></div>
          <div className={!model?.crop.enabled ? 'muted-section' : ''}>
            <VectorFields label="最小值" value={model?.crop.min} max={model?.crop.max.map(v => v - 0.0001) as Vec3 | undefined} disabled={!editable || !model?.crop.enabled} onCommit={value => crop('min', value)} />
            <VectorFields label="最大值" value={model?.crop.max} min={model?.crop.min.map(v => v + 0.0001) as Vec3 | undefined} disabled={!editable || !model?.crop.enabled} onCommit={value => crop('max', value)} />
            <label className="checkbox-row"><input type="checkbox" disabled={!editable || !model?.crop.enabled} checked={model?.crop.invert ?? false} onChange={event => crop('invert', event.target.checked)} /><span>反向裁剪 · 移除盒内高斯点</span></label>
          </div>
          <div className="property-note"><Info size={12} /><span>包围盒使用模型局部坐标。编辑自动保存，导出时应用变换与裁剪。</span></div>
        </div>
        <SectionTitle icon={<Sparkles size={13} />}>投影几何清理</SectionTitle>
        <div className="property-section">
          <div className="single-row"><span className="field-label">边长占比上限</span><NumberField label="最大投影半径占图像边长的比例" value={cleanRadius} min={0.001} max={1} step={0.01} disabled={!canClean} precision={3} onCommit={setCleanRadius} /></div>
          <button className="button compact" disabled={!canClean} title={canClean ? '按训练相机筛选大投影并生成独立副本，原模型保留' : '选择模型并完成相机重建后可用'} onClick={() => { if (selected) void work.run('model.clean', { model_id: selected.id, max_screen_radius: cleanRadius }); }}><Sparkles size={13} />清理大投影</button>
          <div className="property-note"><Info size={12} /><span>按训练相机中的投影大小生成清理副本，原模型保留。{Math.round(cleanRadius * 1000) / 10}% 表示图像边长占比。</span></div>
        </div>
        <SectionTitle>文件信息</SectionTitle>
        <div className="property-section file-info"><span className="field-label">源文件</span><p title={model?.path}>{model?.path ?? '—'}</p><div><span>坐标约定</span><span className="mono">Y-UP / RIGHT-HANDED</span></div></div>
      </> : <>
        <div className="settings-intro"><Settings2 size={19} /><strong>训练与素材设置</strong><p>设置保存在当前项目中，下次导入或训练时生效。</p></div>
        <SectionTitle>高斯训练</SectionTitle>
        <div className="property-section settings-fields">
          <label><span>训练配方</span><select className="number-input" aria-label="训练配方" disabled={disabledSettings}
            value={String(settings?.training_recipe ?? 'quality_v2')}
            onChange={event => work.updateSettings({ training_recipe: event.target.value,
              training_recipe_overrides: {}, training_backend: 'optimized' })}>
            <option value="observed_v5">投影合成 v5 · 已发布{hasRecipeOverrides && settings?.training_recipe === 'observed_v5' ? '（自定义）' : ''}</option>
            <option value="observed_v6">椭圆裁剪 v6 · 实验{hasRecipeOverrides && settings?.training_recipe === 'observed_v6' ? '（自定义）' : ''}</option>
            <option value="quality_v2">公开基线 v2{hasRecipeOverrides && settings?.training_recipe === 'quality_v2' ? '（自定义）' : ''}</option>
          </select></label>
          {['observed_v5', 'observed_v6'].includes(String(settings?.training_recipe)) && <label className="checkbox-row"><input type="checkbox" disabled={disabledSettings}
            checked={record(settings?.training_recipe_overrides).scale_anchor_regularization !== false}
            onChange={event => work.updateSettings({ training_recipe_overrides: { ...record(settings?.training_recipe_overrides), scale_anchor_regularization: event.target.checked } })} />
            <span>尺度锚定正则 · 实验</span></label>}
          {['observed_v5', 'observed_v6'].includes(String(settings?.training_recipe)) && record(settings?.training_recipe_overrides).scale_anchor_regularization !== false &&
            <label><span>正则强度 <small>1 = 观测系数</small></span><NumberField label="尺度正则相对强度" disabled={disabledSettings}
              value={Number(record(settings?.training_recipe_overrides).scale_rel_l2reg ?? (['observed_v5', 'observed_v6'].includes(String(settings?.training_recipe)) ? 0.00001 : 0.05)) / 0.05} min={0} max={10} step={0.0001} precision={6}
              onCommit={value => work.updateSettings({ training_recipe_overrides: { ...record(settings?.training_recipe_overrides),
                scale_anchor_regularization: value > 0, scale_rel_l2reg: 0.05 * value, scale_lower_l2reg: 0.1 * value } })} /></label>}
          {settings?.training_recipe === 'observed_v5' && <p className="property-note">使用独立合成器及冻结的训练投影配置。旧检查点仍沿用保存的配方；固定数据对照与适用范围见随包验收报告。</p>}
          {settings?.training_recipe === 'observed_v6' && <p className="property-note">实验配置，当前尚未通过 R13 画质门槛。继续训练沿用检查点设置；详细结果见随包对照报告。</p>}
          <label><span>训练步数 <small>Steps</small></span><NumberField label="训练步数" value={settings?.steps} disabled={disabledSettings} min={1} max={1000000} step={100} integer precision={0} onCommit={steps => work.updateSettings({ steps })} /></label>
          <label><span>高斯点上限 <small>Max splats</small></span><NumberField label="高斯点上限" value={settings?.max_splats} disabled={disabledSettings} min={1} max={100000000} step={10000} integer precision={0} onCommit={max_splats => work.updateSettings({ max_splats })} /></label>
        </div>
        <SectionTitle>图像与视频</SectionTitle>
        <div className="property-section settings-fields">
          <label><span>图像最大边长 <small>px</small></span><NumberField label="图像最大边长" value={settings?.max_image_size} disabled={disabledSettings} min={64} max={16384} step={100} integer precision={0} onCommit={max_image_size => work.updateSettings({ max_image_size })} /></label>
          <label><span>视频采样帧率 <small>fps</small></span><NumberField label="视频采样帧率" value={settings?.video_fps} disabled={disabledSettings} min={0.01} max={120} step={0.5} onCommit={video_fps => work.updateSettings({ video_fps })} /></label>
          <label><span>视频提取帧上限 <small>frames</small></span><NumberField label="视频提取帧上限" value={settings?.max_frames} disabled={disabledSettings} min={1} max={100000} step={16} integer precision={0} onCommit={max_frames => work.updateSettings({ max_frames })} /></label>
        </div>
        <div className="property-note settings-note"><Info size={13} /><span>切换配方后需重新训练。暂停会保存完整检查点；「恢复」沿用检查点中的配方和参数。</span></div>
      </>}
    </div>
    <div className="inspector-footer"><span className={`save-dot ${work.saveState}`} />{!work.project ? '打开项目后可编辑' : ({ saved: '所有编辑已保存', pending: '编辑等待保存…', saving: '正在保存编辑…', error: '保存失败 · 点击保存重试' }[work.saveState])}</div>
  </aside>;
}
