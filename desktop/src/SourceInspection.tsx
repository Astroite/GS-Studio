import { useState } from 'react';
import { Camera, Images, TriangleAlert } from 'lucide-react';
import { modelUrl } from './engine';
import { formatCount, record } from './types';
import type { Project } from './types';
import './source-inspection.css';
import { reconstructionCamera } from './cameras';
import type { ReconstructionCamera } from './cameras';

const basename = (path: unknown) => String(path ?? '').split(/[\\/]/).at(-1) ?? '';
const matrix = (value: unknown): string => Array.isArray(value)
  ? value.map(row => Array.isArray(row) ? row.map(v => typeof v === 'number' ? v.toFixed(5).padStart(11) : String(v)).join(' ') : String(row)).join('\n')
  : '无相机矩阵';

export function SourceInspection({ project, onViewCamera }: { project: Project; onViewCamera: (camera: ReconstructionCamera) => void }) {
  const assets = record(project.assets), reconstruction = record(project.reconstruction);
  const images = Array.isArray(assets.images) ? assets.images.map(record) : [];
  const cameras = Array.isArray(reconstruction.cameras) ? reconstruction.cameras.map(record) : [];
  const [tab, setTab] = useState<'images' | 'cameras'>('images');
  const [imageIndex, setImageIndex] = useState(0), [cameraIndex, setCameraIndex] = useState(0);
  const [imageError, setImageError] = useState(false);
  const image = images[imageIndex], camera = cameras[cameraIndex];
  const validCamera = reconstructionCamera(camera);
  const cameraNames = new Set(cameras.map(item => basename(item.name)));
  return <div className="source-inspection">
    <div className="source-summary"><span><Images size={15} />{formatCount(images.length)} 张图像</span><span><Camera size={15} />{formatCount(cameras.length)} 台已注册相机</span><span>{formatCount(typeof reconstruction.point_count === 'number' ? reconstruction.point_count : undefined)} 稀疏点</span></div>
    <div className="tab-strip"><button className={tab === 'images' ? 'selected' : ''} onClick={() => setTab('images')}>素材检查</button><button className={tab === 'cameras' ? 'selected' : ''} onClick={() => setTab('cameras')}>相机参数</button></div>
    <div className="source-inspection-body">
      <div className="source-file-list">{tab === 'images' ? images.map((item, index) => <button key={String(item.path ?? index)} className={index === imageIndex ? 'selected' : ''} onClick={() => { setImageIndex(index); setImageError(false); }} title={String(item.source ?? item.path)}><Images size={14} /><span><strong>{basename(item.source ?? item.path)}</strong><small>{String(item.width ?? '—')} × {String(item.height ?? '—')}{cameraNames.has(basename(item.path)) ? ' · 已注册' : ''}</small></span></button>) : cameras.map((item, index) => <button key={String(item.name ?? index)} className={index === cameraIndex ? 'selected' : ''} onClick={() => setCameraIndex(index)}><Camera size={14} /><span><strong>{basename(item.name) || `相机 ${index + 1}`}</strong><small>{String(item.width ?? '—')} × {String(item.height ?? '—')}</small></span></button>)}
        {!(tab === 'images' ? images.length : cameras.length) && <div className="source-empty">{tab === 'images' ? '当前项目尚未导入图像' : '重建相机后可检查其参数'}</div>}
      </div>
      <div className="source-detail">{tab === 'images' && image ? <>
        <div className="source-image">{imageError ? <div className="source-empty"><TriangleAlert size={22} />无法读取图像文件</div> : <img src={modelUrl(String(image.path))} alt={basename(image.source ?? image.path)} onError={() => setImageError(true)} />}</div>
        <strong>{basename(image.source ?? image.path)}</strong><p className="quiet">{String(image.width ?? '—')} × {String(image.height ?? '—')} 像素</p><span className="field-label">项目图像</span><p className="source-path">{String(image.path ?? '—')}</p><span className="field-label">原始素材</span><p className="source-path">{String(image.source ?? '—')}</p>
      </> : tab === 'cameras' && camera ? <><h3>{basename(camera.name) || `相机 ${cameraIndex + 1}`}</h3><p className="quiet">去畸变图像 · {String(camera.width ?? '—')} × {String(camera.height ?? '—')} 像素</p><button className="button primary compact" disabled={!validCamera} onClick={() => { if (validCamera) onViewCamera(validCamera); }}><Camera size={14} />从此相机查看</button><h4>相机内参 K</h4><pre className="camera-matrix mono">{matrix(camera.K)}</pre><h4>世界到相机矩阵</h4><pre className="camera-matrix mono">{matrix(camera.w2c)}</pre><p className="property-note">场景使用右手坐标系、Y 轴向上；以上矩阵对应已求解的实际相机。</p></> : <div className="source-empty">从左侧选择一项查看详情</div>}</div>
    </div>
  </div>;
}
