import { record } from './types';

export const newTrainingSettings = () => ({ training_preset: 'auto', training_recipe: 'quality_v2', sh_degree: 3 });

// Only explicit overrides travel to the pipeline. Resolved defaults describe the
// package; copying them into a new request would override source-aware defaults.
export function checkedTrainingOptions(value: unknown, request: Record<string, unknown>) {
  const v = record(value);
  if (typeof v.package !== 'string' || !v.package || typeof v.package_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(v.package_sha256)
    || !['photo', 'gs'].includes(String(v.training_preset)) || v.requested_training_preset !== request.training_preset
    || v.training_recipe !== request.training_recipe || v.sh_degree !== 3
    || !['steps', 'max_splats', 'seed', 'max_image_size', 'sparse_depth_observations'].every(k => Number.isInteger(v[k]) && Number(v[k]) >= 0)
    || Number(v.steps) < 1 || Number(v.max_splats) < 1
    || !['photo_comp', 'use_bilateral_grid', 'use_sparse_depth', 'sparse_depth_requested', 'sparse_depth_available'].every(k => typeof v[k] === 'boolean')
    || v.sparse_depth_available !== (Number(v.sparse_depth_observations) > 0)
    || v.use_sparse_depth !== (v.sparse_depth_requested && v.sparse_depth_available)
    || v.sparse_depth_status !== (!v.sparse_depth_requested ? 'disabled' : v.sparse_depth_available ? 'enabled' : 'unavailable')
    || !(typeof v.sparse_depth_reason === 'string' && (v.sparse_depth_available || v.sparse_depth_reason.length > 0)
      || v.sparse_depth_available && v.sparse_depth_reason === null)) throw new Error('training_options 返回的预算 / 训练包 / 深度状态无效');
  return v;
}

export function trainingOptionsIdentity(v: Record<string, unknown>) {
  return JSON.stringify(['package', 'package_sha256', 'source_kind', 'requested_training_preset', 'training_preset',
    'steps', 'training_recipe', 'sh_degree', 'max_splats', 'seed', 'max_image_size', 'photo_comp', 'use_bilateral_grid',
    'sparse_depth_requested', 'use_sparse_depth', 'sparse_depth_available', 'sparse_depth_observations', 'sparse_depth_status'].map(k => v[k]));
}

export function savedTrainingSettings(dispatch: Record<string, unknown>, config: Record<string, unknown>) {
  return { training_preset: dispatch.requested_training_preset ?? 'auto',
    steps: dispatch.requested_steps ?? config.steps, photo_comp: dispatch.photo_comp ?? config.photo_comp,
    use_bilateral_grid: dispatch.use_bilateral_grid ?? config.use_bilateral_grid,
    use_sparse_depth: dispatch.sparse_depth_requested ?? config.sparse_depth_requested,
    training_recipe: dispatch.training_recipe ?? config.training_recipe, sh_degree: 3 };
}
