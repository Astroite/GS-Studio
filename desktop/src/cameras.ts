import { Matrix4 } from 'three';

export interface ReconstructionCamera { name: string; width: number; height: number; K: number[][]; w2c: number[][] }
export interface CameraView { camera: ReconstructionCamera; version: number }
export function reconstructionCamera(value: unknown): ReconstructionCamera | null {
  if (!value || typeof value !== 'object') return null;
  const source = value as Record<string, unknown>;
  const validMatrix = (matrix: unknown, size: number): matrix is number[][] => Array.isArray(matrix) && matrix.length === size
    && matrix.every(row => Array.isArray(row) && row.length === size && row.every(v => typeof v === 'number' && Number.isFinite(v)));
  if (!validMatrix(source.K, 3) || !validMatrix(source.w2c, 4) || source.K[1][1] <= 0
    || typeof source.width !== 'number' || source.width <= 0 || typeof source.height !== 'number' || source.height <= 0) return null;
  return { name: String(source.name ?? '重建相机'), width: source.width, height: source.height, K: source.K, w2c: source.w2c };
}
export function cameraWorldPose(camera: ReconstructionCamera): Matrix4 {
  // JSON matrices are row-major; Three stores column-major. RDF camera -> RUB
  // camera requires postmultiplication after inverting the supplied world-to-RDF.
  const w2c = new Matrix4().fromArray(camera.w2c.flat()).transpose();
  if (Math.abs(w2c.determinant()) < 1e-12) throw new Error('相机矩阵不可逆');
  return w2c.invert().multiply(new Matrix4().makeScale(1, -1, -1));
}
export function cameraVerticalFov(camera: ReconstructionCamera): number {
  return 2 * Math.atan(camera.height / (2 * camera.K[1][1])) * 180 / Math.PI;
}
