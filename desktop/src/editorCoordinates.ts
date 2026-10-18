import { Matrix4, PerspectiveCamera } from 'three';
import type { ReconstructionCamera } from './cameras';

export function viewportCamera(camera: PerspectiveCamera, width: number, height: number): ReconstructionCamera {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) throw new Error('视口尺寸无效');
  camera.updateMatrixWorld(true);
  // Projection is the actual viewport projection, including principal point and zoom.
  const p = camera.projectionMatrix.elements;
  const K = [[p[0] * width / 2, 0, (1 - p[8]) * width / 2], [0, p[5] * height / 2, (1 + p[9]) * height / 2], [0, 0, 1]];
  const rdf = new Matrix4().makeScale(1, -1, -1).multiply(camera.matrixWorld.clone().invert()).transpose().elements;
  return { name: '当前视口', width, height, K, w2c: Array.from({ length: 4 }, (_, i) => rdf.slice(i * 4, i * 4 + 4)) };
}
export function selectionRectangle(start: [number, number], end: [number, number], bounds: { left: number; top: number; width: number; height: number }, width: number, height: number): [number, number, number, number] {
  if (bounds.width <= 0 || bounds.height <= 0) throw new Error('选择区域无效');
  const map = (p: [number, number]) => [Math.max(0, Math.min(width, (p[0] - bounds.left) / bounds.width * width)), Math.max(0, Math.min(height, (p[1] - bounds.top) / bounds.height * height))];
  const a = map(start), b = map(end);
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
}
export type EditTransform = { translation: [number, number, number]; rotation_degrees: [number, number, number]; scale: number; pivot: [number, number, number] };
export type EditCrop = { min: [number, number, number]; max: [number, number, number]; keep_inside: boolean };
export const identityTransform = (): EditTransform => ({ translation: [0, 0, 0], rotation_degrees: [0, 0, 0], scale: 1, pivot: [0, 0, 0] });
export function validEdit(transform?: EditTransform, crop?: EditCrop): boolean {
  if (transform && (![...transform.translation, ...transform.rotation_degrees, transform.scale, ...transform.pivot].every(Number.isFinite) || transform.scale <= 0)) return false;
  if (crop && (![...crop.min, ...crop.max].every(Number.isFinite) || crop.min.some((v, i) => v >= crop.max[i]))) return false;
  return true;
}
export function editorPreviewArgs(transform: EditTransform, crop: EditCrop, enabled: boolean) {
  if (!validEdit(transform, enabled ? crop : undefined)) throw new Error('变换 / 裁剪参数无效');
  return { transform: { position: transform.translation, rotation: transform.rotation_degrees.map(v => v * Math.PI / 180), scale: transform.scale }, crop: { enabled, min: crop.min, max: crop.max, invert: !crop.keep_inside } };
}
