import { record } from './types.ts';

// The worker supplies the portable/configured root in the first catalog response.
export const DEFAULT_ROOT = '';
export type Row = Record<string, unknown> & { id: string };
export interface Catalog { root: string; locations: Row[]; scenes: Row[]; captures: Row[]; runs: Row[]; external_imports: Row[] }
export interface Context { location_id?: string; scene_id?: string; capture_id?: string; run_id?: string; import_id?: string }
export const rows = (value: unknown): Row[] => Array.isArray(value) ? value.filter(v => typeof record(v).id === 'string') as Row[] : [];
export function catalogFrom(value: unknown): Catalog {
  const c = record(value);
  if (typeof c.root !== 'string' || !Array.isArray(c.locations) || !Array.isArray(c.runs)) throw new Error('studio.catalog 返回无效清单');
  return { root: c.root, locations: rows(c.locations), scenes: rows(c.scenes), captures: rows(c.captures), runs: rows(c.runs), external_imports: rows(c.external_imports) };
}
export function inContext(row: Row, context: Context): boolean {
  return (!context.location_id || row.location_id === context.location_id) && (!context.scene_id || row.scene_id === context.scene_id);
}
export function manifestStatus(row: Row): string { return row.readable === false ? 'unreadable' : typeof row.status === 'string' ? row.status : 'unknown'; }
export function eventMatches(data: Record<string, unknown>, context: Context): boolean {
  return Object.entries(context).every(([key, value]) => !value || data[key] === undefined || data[key] === value);
}
export function previewMatches(value: unknown, expected: { request_id: string; package_sha256: string; image: string }): boolean {
  const v = record(value);
  return v.request_id === expected.request_id && v.package_sha256 === expected.package_sha256 && v.image === expected.image
    && typeof v.source_path === 'string' && typeof v.render_path === 'string';
}
export class LatestRequest {
  private revision = 0;
  next() { return ++this.revision; }
  current(id: number) { return id === this.revision; }
  invalidate() { ++this.revision; }
}
export function canFinalizeReview(snapshot: Record<string, unknown>, draft: Record<string, string>, committed: boolean): boolean {
  const items = Array.isArray(snapshot.items) ? snapshot.items.map(record) : [];
  return committed && !!snapshot.identity && snapshot.can_finalize === true && items.length > 0
    && items.every(item => ['ok', 'exclude'].includes(draft[String(item.image_id)] ?? String(item.status)));
}
export function controlIdentity(value: unknown): { output: string; package_sha256: string; generation: string } | null {
  const v = record(value), c = record(v.control_state);
  const output = v.output, digest = v.package_sha256 ?? c.package_sha256, generation = v.generation ?? c.generation;
  return typeof output === 'string' && typeof digest === 'string' && typeof generation === 'string' ? { output, package_sha256: digest, generation } : null;
}
