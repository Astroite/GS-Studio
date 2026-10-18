// Match the frozen engine evaluation policy. Spark remains an interactive
// preview rasterizer, with a finite far proxy for unbounded perspective.
export interface ViewportPolicy { near: number; far: number; mode: 'classic' | 'antialiased'; preBlurAmount: number; blurAmount: number }
function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
export function projectRenderPolicy(project: unknown): ViewportPolicy {
  const recipe = object(object(object(project).training).recipe), policy = object(recipe.projection);
  const near = typeof policy.eval_near_plane === 'number' && Number.isFinite(policy.eval_near_plane) && policy.eval_near_plane > 0 ? policy.eval_near_plane : 0.01;
  const far = policy.far_mode === 'fixed' && typeof policy.far_plane === 'number' && Number.isFinite(policy.far_plane) && policy.far_plane > near ? policy.far_plane : 1e8;
  const mode = (policy.eval_rasterize_mode ?? policy.rasterize_mode) === 'antialiased' ? 'antialiased' : 'classic';
  return { near, far, mode, preBlurAmount: mode === 'classic' ? 0.3 : 0, blurAmount: mode === 'classic' ? 0 : 0.3 };
}
