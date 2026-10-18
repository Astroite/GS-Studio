import { useEffect, useState } from 'react';
import { Box, Camera, ChevronRight, FolderOpen, Layers3, MapPin, ScanLine } from 'lucide-react';
import type { StudioWork } from './useStudio';
import type { Workbench } from './useWorkbench';
import type { Row } from './studioState';
import { manifestStatus } from './studioState';
import { readable, record } from './types';
import { displayStatus, experimentLabel } from './Workflow';

function TreeText({ label, status }: { label: string; status?: unknown }) {
  return <span className="gs-tree-text"><span className="gs-tree-label" title={label}>{label}</span>{status != null && <small title={String(status)}>{displayStatus(status)}</small>}</span>;
}
function ExperimentButton({ experiment, choose }: { experiment: Record<string, unknown>; choose: () => void }) {
  const label = `实验 ${experimentLabel(experiment)}`, output = String(experiment.output ?? '');
  return <button title={output} aria-label={`${label} · ${displayStatus(experiment.status)} · ${output}`} onClick={choose}><TreeText label={label} status={experiment.status} /></button>;
}

export function Catalog({ studio, work, go, chooseRun, chooseTraining }: { studio: StudioWork; work: Workbench; go: (tab: string) => void; chooseRun: (row: Row) => void; chooseTraining: (row: Row, segment?: string, output?: string) => void }) {
  const x = studio.context;
  const runNode = (run: Row) => <RunBranch key={run.id} row={run} studio={studio} chooseRun={chooseRun} chooseTraining={chooseTraining} />;
  return <aside className="gs-catalog"><div className="panel-heading"><Layers3 size={14} />项目目录<span className="small-label">CATALOG</span></div><div className="gs-tree">
    {!studio.catalog.locations.length && <p className="gs-note">创建地点和场景，开始登记素材。</p>}
    {studio.catalog.locations.map(location => <details open key={location.id}>
      <summary onClick={() => studio.setContext({ location_id: location.id })}><MapPin size={13} /><TreeText label={String(location.display_name ?? location.id)} /></summary>
      {location.readable === false && <p className="gs-note">{readable(location.problems)}</p>}
      {studio.catalog.scenes.filter(s => s.location_id === location.id).map(scene => <details open key={scene.id}>
        <summary onClick={() => studio.setContext({ location_id: location.id, scene_id: scene.id })}><ChevronRight size={12} /><TreeText label={String(scene.display_name ?? scene.id)} /></summary>
        {scene.readable === false && <p className="gs-note">{readable(scene.problems)}</p>}
        {studio.catalog.captures.filter(c => c.location_id === location.id && c.scene_id === scene.id).map(capture => <div className="gs-tree-capture" key={capture.id}>
          <button className={x.capture_id === capture.id && x.scene_id === scene.id ? 'selected' : ''} onClick={() => { studio.setContext({ location_id: location.id, scene_id: scene.id, capture_id: capture.id }); go('capture'); }}><Camera size={12} /><TreeText label={capture.id} status={manifestStatus(capture)} /></button>
          {studio.catalog.runs.filter(r => r.location_id === location.id && r.scene_id === scene.id && r.capture_id === capture.id).map(runNode)}
        </div>)}
        {studio.catalog.runs.filter(r => r.location_id === location.id && r.scene_id === scene.id && !studio.catalog.captures.some(c => c.id === r.capture_id && c.location_id === location.id && c.scene_id === scene.id)).map(runNode)}
      </details>)}
    </details>)}
    <details open><summary onClick={() => go('external')}><FolderOpen size={13} /><TreeText label="外部导入" /></summary>{studio.catalog.external_imports.map(v => <details key={v.id}><summary onClick={() => { studio.setContext({ import_id: v.id }); go('training'); }}><TreeText label={v.id} status={v.status} /></summary>{(Array.isArray(v.experiments) ? v.experiments.map(record) : []).map((e, i) => <ExperimentButton key={i} experiment={e} choose={() => { studio.setContext({ import_id: v.id }); chooseTraining({ id: '', import_id: v.id }, undefined, String(e.output ?? '')); }} />)}</details>)}</details>
    <details open><summary onClick={() => go('models')}><Box size={13} /><TreeText label="独立模型 / 原生实验" /></summary>{work.projects.map(p => <button disabled={studio.locked} key={p.path} onClick={() => { void work.openProject(p.path); go('models'); }}><TreeText label={p.name} status={p.status} /></button>)}</details>
  </div></aside>;
}

function RunBranch({ row, studio, chooseRun, chooseTraining }: { row: Row; studio: StudioWork; chooseRun: (row: Row) => void; chooseTraining: (row: Row, segment?: string, output?: string) => void }) {
  const [expanded, setExpanded] = useState(false), [segments, setSegments] = useState<Record<string, unknown>[]>([]), [experiments, setExperiments] = useState<Record<string, unknown>[]>([]), [error, setError] = useState('');
  useEffect(() => {
    let live = true;
    if (expanded && row.readable !== false) {
      const ids = { location_id: row.location_id, scene_id: row.scene_id, run_id: row.id };
      void Promise.allSettled([studio.read('segment_choices', ids), studio.read('experiments', ids)]).then(values => {
        if (!live) return;
        const [choice, list] = values;
        if (choice.status === 'fulfilled') setSegments(Array.isArray(choice.value.segments) ? choice.value.segments.map(record) : []);
        else setError(readable(choice.reason));
        if (list.status === 'fulfilled') setExperiments(Array.isArray(list.value.experiments) ? list.value.experiments.map(record) : []);
        else setError(readable(list.reason));
      });
    }
    return () => { live = false; };
  }, [expanded, row.id, row.status, row.updated_at]);
  return <details onToggle={e => setExpanded(e.currentTarget.open)}><summary className={studio.context.run_id === row.id ? 'selected' : ''} onClick={() => chooseRun(row)}><ScanLine size={12} /><TreeText label={row.id} status={manifestStatus(row)} /></summary>
    {segments.map(segment => <details key={String(segment.id)}><summary onClick={() => chooseTraining(row, String(segment.id))}><TreeText label={`分段 ${String(segment.id)}`} /></summary>{experiments.filter(e => e.segment === segment.id).map((e, i) => <ExperimentButton key={i} experiment={e} choose={() => chooseTraining(row, String(segment.id), String(e.output))} />)}</details>)}
    {experiments.filter(e => !segments.some(s => s.id === e.segment)).map((e, i) => <ExperimentButton key={i} experiment={e} choose={() => chooseTraining(row, String(e.segment ?? ''), String(e.output))} />)}
    {error && <p className="gs-note">{error}</p>}
  </details>;
}
