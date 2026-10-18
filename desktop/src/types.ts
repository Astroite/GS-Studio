export type Vec3 = [number, number, number];
export interface Model {
  id: string;
  name: string;
  path: string;
  count: number;
  visible: boolean;
  transform: { position: Vec3; rotation: Vec3; scale: number };
  crop: { enabled: boolean; min: Vec3; max: Vec3; invert: boolean };
  previewStep?: number;
  [key: string]: unknown;
}
export interface Settings {
  steps: number;
  max_splats: number;
  max_image_size: number;
  video_fps: number;
  max_frames: number;
  [key: string]: unknown;
}
export interface Project {
  id: string;
  name: string;
  path: string;
  created_at: string;
  updated_at: string;
  status: string;
  settings: Settings;
  models: Model[];
  [key: string]: unknown;
}
export interface EngineEvent {
  type: 'event';
  id: string;
  event: 'progress' | 'log' | 'metric' | 'preview';
  data: Record<string, unknown>;
}
export interface LogEntry { id: number; time: string; level: string; message: string }
export interface Metric { step: number; loss: number; psnr?: number; splats?: number; elapsed_seconds?: number; gpu_memory_mb?: number }
export interface Operation { method: string; title: string; path: string; started: number; dispatched: boolean }
export interface Progress { stage: string; progress?: number; message: string }
export type Health = Record<string, unknown>;
export const defaultSettings: Settings = { steps: 10000, max_splats: 1000000, max_image_size: 1600, video_fps: 1, max_frames: 128 };

export function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
export function finite(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}
function vector(value: unknown, fallback: Vec3): Vec3 {
  return Array.isArray(value) && value.length === 3 ? value.map((v, i) => finite(v, fallback[i])) as Vec3 : [...fallback];
}
export function normalizeModel(value: unknown, index = 0): Model {
  const m = record(value), t = record(m.transform), c = record(m.crop);
  return { ...m, id: String(m.id ?? `model-${index}`), name: String(m.name ?? '未命名模型'), path: String(m.path ?? ''),
    count: finite(m.count), visible: m.visible !== false,
    transform: { position: vector(t.position, [0, 0, 0]), rotation: vector(t.rotation, [0, 0, 0]), scale: Math.max(0.0001, finite(t.scale, 1)) },
    crop: { enabled: c.enabled === true, min: vector(c.min, [-1, -1, -1]), max: vector(c.max, [1, 1, 1]), invert: c.invert === true } };
}
export function normalizeProject(value: unknown): Project {
  const p = record(value), settings = record(p.settings);
  if (typeof p.path !== 'string' || !p.path) throw new Error('引擎返回的项目缺少有效 path');
  return { ...p, id: String(p.id ?? p.path), name: String(p.name ?? '未命名项目'), path: p.path,
    created_at: String(p.created_at ?? ''), updated_at: String(p.updated_at ?? ''), status: String(p.status ?? 'unknown'),
    settings: { ...defaultSettings, ...settings } as Settings,
    models: Array.isArray(p.models) ? p.models.map(normalizeModel) : [] };
}
export function projectCounts(project: Project | null): { images?: number; cameras?: number } {
  if (!project) return {};
  const counts = record(project.counts), dataset = record(project.dataset), reconstruction = record(project.reconstruction), assets = record(project.assets);
  const count = (...values: unknown[]) => {
    for (const value of values) {
      if (typeof value === 'number' && Number.isFinite(value)) return value;
      if (Array.isArray(value)) return value.length;
    }
    return undefined;
  };
  return { images: count(project.image_count, counts.images, dataset.image_count, dataset.images, project.images, assets.images),
    cameras: count(project.camera_count, counts.cameras, reconstruction.camera_count, reconstruction.registered_images, reconstruction.cameras, project.cameras) };
}
export function healthMissing(health: Health | null): string[] {
  if (!health) return [];
  const missing = health.missing ?? health.missing_dependencies;
  if (Array.isArray(missing)) return missing.map(v => typeof v === 'string' ? v : JSON.stringify(v));
  if (missing && typeof missing === 'object') return Object.entries(record(missing)).map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`);
  const dependencies = record(health.dependencies);
  return Object.entries(dependencies).filter(([, v]) => v === false || record(v).available === false)
    .map(([k, v]) => `${k}${record(v).error ? `: ${record(v).error}` : ''}`);
}
export function readable(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value instanceof Error) return value.message;
  try { return JSON.stringify(value, null, 2) ?? String(value); } catch { return String(value); }
}
export const formatCount = (value: number | undefined) => value === undefined ? '—' : new Intl.NumberFormat('zh-CN').format(value);
