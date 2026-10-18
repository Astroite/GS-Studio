import assert from 'node:assert/strict';
import test from 'node:test';
import { PerspectiveCamera, Vector3 } from 'three';
import { projectRenderPolicy } from './renderPolicy.ts';

test('classic and AA preserve the same blurred covariance, with only AA compensating opacity', () => {
  const input = [0.02, 0.07];
  for (const mode of ['classic','antialiased']) {
    const p=projectRenderPolicy({training:{recipe:{projection:{eval_rasterize_mode:mode}}}});
    const before=input.map(x=>x+p.preBlurAmount), after=before.map(x=>x+p.blurAmount);
    assert.deepEqual(after,[0.32,0.37]);
    const opacity=Math.sqrt(before[0]*before[1]/(after[0]*after[1]));
    if(mode==='classic')assert.equal(opacity,1);
    else assert(Math.abs(opacity-Math.sqrt(.02*.07/(.32*.37)))<1e-15);
  }
});
test('eval policy wins over training AA and settings edited after checkpoint creation', () => {
  const p=projectRenderPolicy({settings:{training_recipe_overrides:{projection:{eval_near_plane:5}}},
    training:{recipe:{projection:{eval_near_plane:.1,train_near_plane:.2,train_rasterize_mode:'antialiased',eval_rasterize_mode:'classic'}}}});
  assert.equal(p.near,.1);assert.equal(p.mode,'classic');assert.equal(p.far,1e8);
  const camera=new PerspectiveCamera(50,1,p.near,p.far);
  assert(new Vector3(0,0,-.05).project(camera).z < -1);
  assert(new Vector3(0,0,-.2).project(camera).z > -1);
});
test('legacy/no-project policy matches engine defaults and invalid metadata cannot corrupt camera matrix', () => {
  for(const input of [null,{}, {training:{recipe:{projection:{eval_near_plane:NaN,far_mode:'fixed',far_plane:0}}}}]) {
    const p=projectRenderPolicy(input);assert.equal(p.near,.01);assert.equal(p.mode,'classic');
    assert(new PerspectiveCamera(50,1,p.near,p.far).projectionMatrix.elements.every(Number.isFinite));
  }
});
