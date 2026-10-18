import assert from 'node:assert/strict';
import test from 'node:test';
import { PerspectiveCamera, Vector3, Matrix4, Euler } from 'three';
import { catalogFrom, manifestStatus, previewMatches, eventMatches, LatestRequest } from './studioState.ts';
import { viewportCamera, selectionRectangle, identityTransform, validEdit, editorPreviewArgs } from './editorCoordinates.ts';
import { verifiedPreviewBytes } from './previewFiles.ts';

test('manifest failures remain failures even with PLY paths', () => {
  const c = catalogFrom({root:'E:/AI/GSStudio/projects',locations:[],scenes:[],captures:[],runs:[{id:'run',status:'failed',path:'existing.ply'}],external_imports:[]});
  assert.equal(manifestStatus(c.runs[0]),'failed');
  assert.equal(manifestStatus({id:'broken',status:'succeeded',readable:false}),'unreadable');
  assert.throws(()=>catalogFrom({locations:[],runs:[]}),/无效清单/);
});
test('request correlation rejects camera/package/stale ID and stale editor completions', () => {
  const expected = {request_id:'a',package_sha256:'hash',image:'camera-a'};
  const pair = {...expected,source_path:'source.png',render_path:'render.png'};
  assert(previewMatches(pair,expected));
  for (const field of ['request_id','package_sha256','image']) assert(!previewMatches({...pair,[field]:'stale'},expected));
  const requests=new LatestRequest(), old=requests.next(), current=requests.next();
  assert(!requests.current(old)); assert(requests.current(current)); requests.invalidate(); assert(!requests.current(current));
  assert(!eventMatches({run_id:'old'},{run_id:'new'}));
});
test('viewport K and RDF w2c match projected pixels after roll, pan, zoom and off-center projection', () => {
  const c=new PerspectiveCamera(63,1200/700,.01,10000); c.position.set(3,-2,5); c.rotation.copy(new Euler(.2,-.4,.3)); c.zoom=1.3; c.setViewOffset(2400,1400,230,170,1200,700); c.updateProjectionMatrix(); c.updateMatrixWorld(true);
  const native=viewportCamera(c,1200,700), w2c=new Matrix4().fromArray(native.w2c.flat()).transpose();
  for (const p of [new Vector3(1,.2,-5),new Vector3(-.1,.1,-10)]) {
    p.applyMatrix4(c.matrixWorld); const ndc=p.clone().project(c), rdf=p.clone().applyMatrix4(w2c);
    assert(Math.abs(native.K[0][0]*rdf.x/rdf.z+native.K[0][2]-(ndc.x+1)*600)<1e-8);
    assert(Math.abs(native.K[1][1]*rdf.y/rdf.z+native.K[1][2]-(1-ndc.y)*350)<1e-8);
  }
});
test('rectangle helpers account for CSS scaling, reverse drags and clamp viewport bounds', () => {
  assert.deepEqual(selectionRectangle([250,120],[50,20],{left:100,top:50,width:200,height:100},1000,500),[0,0,750,350]);
  assert.throws(()=>selectionRectangle([0,0],[1,1],{left:0,top:0,width:0,height:1},1,1));
});
test('numeric/gizmo arguments use identical core units and reject invalid crops/scales', () => {
  const transform={...identityTransform(),rotation_degrees:[90,0,180]}, crop={min:[-1,-2,-3],max:[1,2,3],keep_inside:false};
  const wire=editorPreviewArgs(transform,crop,true); assert.deepEqual(wire.transform.position,[0,0,0]); assert.equal(wire.transform.rotation[0],Math.PI/2); assert.equal(wire.crop.invert,true);
  assert(!validEdit({...transform,scale:0})); assert(!validEdit(transform,{...crop,min:[5,0,0]}));
});
test('paired preview bytes are pinned to both hashes even if an atomic slot is reused during fetch', async () => {
  const bytes=new TextEncoder().encode('original preview').buffer;
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
  assert.equal((await verifiedPreviewBytes(['source','render'],[hash,hash],async()=>bytes)).length,2);
  await assert.rejects(verifiedPreviewBytes(['source','render'],[hash,hash],async p=>p==='source'?bytes:new TextEncoder().encode('reused slot').buffer),/已更新/);
  await assert.rejects(verifiedPreviewBytes(['source','render'],['',hash],async()=>bytes),/SHA256/);
});
