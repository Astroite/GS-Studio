import { invoke, isTauri, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open, save } from '@tauri-apps/plugin-dialog';
import type { EngineEvent } from './types';
import { readable } from './types';

export const desktop = isTauri();
export const desktopMessage = '请在桌面程序中打开以连接本地引擎';
export async function requestStageStop(job_id: string, generation_id: string) {
  if (!desktop) throw new Error(desktopMessage);
  return engineCall('studio.operation', { action: 'control_run', job_id, generation_id, control_action: 'stop_after_stage' });
}
export class EngineError extends Error {
  constructor(public method: string, cause: unknown) { super(readable(cause)); this.name = 'EngineError'; }
}
export async function engineCall<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  if (!desktop) throw new EngineError(method, desktopMessage);
  try { return await invoke<T>('engine_call', { method, params }); }
  catch (error) { throw new EngineError(method, error); }
}
export async function subscribeEngine(callback: (event: EngineEvent) => void): Promise<() => void> {
  if (!desktop) return () => {};
  return listen<EngineEvent>('engine-event', ({ payload }) => {
    if (payload?.type === 'event' && payload.data && typeof payload.data === 'object') callback(payload);
  });
}
export const modelUrl = (path: string, version?: number) => `${convertFileSrc(path)}${version !== undefined ? `?preview=${version}` : ''}`;
export async function pickDirectory(): Promise<string | null> {
  if (!desktop) throw new Error(desktopMessage);
  const result = await open({ directory: true, multiple: false, title: '选择项目文件夹' });
  return typeof result === 'string' ? result : null;
}
export async function pickImport(kind: 'images' | 'video' | 'model'): Promise<string[]> {
  if (!desktop) throw new Error(desktopMessage);
  const result = await open({ multiple: kind === 'images', title: { images: '导入照片', video: '导入视频', model: '导入 Gaussian Splat 模型' }[kind],
    filters: [{ name: { images: '图像', video: '视频', model: 'Gaussian Splat' }[kind],
      extensions: { images: ['jpg', 'jpeg', 'png', 'webp', 'tif', 'tiff'], video: ['mp4', 'mov', 'mkv', 'avi', 'webm'], model: ['ply', 'spz'] }[kind] }] });
  return typeof result === 'string' ? [result] : result ?? [];
}
export async function pickExport(name: string): Promise<string | null> {
  if (!desktop) throw new Error(desktopMessage);
  return save({ title: '导出模型（应用变换与裁剪）', defaultPath: `${name.replace(/\.(ply|spz)$/i, '')}.ply`,
    filters: [{ name: 'Gaussian PLY', extensions: ['ply'] }, { name: 'SPZ', extensions: ['spz'] }] });
}
