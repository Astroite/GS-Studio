import assert from 'node:assert/strict';
import test from 'node:test';
import { Euler, Matrix4, PerspectiveCamera, Quaternion, Vector3 } from 'three';
import { reconstructionCamera, cameraWorldPose, cameraVerticalFov } from './cameras.ts';

const source = {name:'identity RDF',width:1000,height:800,K:[[500,0,500],[0,500,400],[0,0,1]],w2c:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]};

test('RDF positive-Z camera converts to Three negative-Z with Y up', () => {
  const pose = cameraWorldPose(reconstructionCamera(source));
  const point = new Vector3(0.5,0.25,2).applyMatrix4(pose.clone().invert());
  assert(point.distanceTo(new Vector3(0.5,-0.25,-2)) < 1e-12);
});

test('reference-camera pixel projection agrees with centered pinhole K', () => {
  const camera = reconstructionCamera(source), pose = cameraWorldPose(camera);
  const view = new PerspectiveCamera(cameraVerticalFov(camera),camera.width/camera.height,0.001,1000);
  view.matrixWorld.copy(pose); view.matrixWorldInverse.copy(pose).invert();
  const ndc = new Vector3(0.5,0.25,2).project(view);
  assert(Math.abs((ndc.x+1)*camera.width/2-625)<1e-9);
  assert(Math.abs((1-ndc.y)*camera.height/2-462.5)<1e-9);
});

test('translated and rolled camera preserves inverse/multiplication order', () => {
  const original = new Matrix4().compose(new Vector3(2,-3,4),new Quaternion().setFromEuler(new Euler(0.4,-0.7,0.2)),new Vector3(1,1,1));
  const w2c = new Matrix4().makeScale(1,-1,-1).multiply(original.clone().invert()).transpose().elements;
  const converted = reconstructionCamera({...source,w2c:Array.from({length:4},(_,i)=>w2c.slice(i*4,i*4+4))});
  const recovered = cameraWorldPose(converted);
  assert(Math.max(...original.elements.map((v,i)=>Math.abs(v-recovered.elements[i])))<1e-12);
});

test('non-finite intrinsics and invalid image dimensions are rejected', () => {
  assert.equal(reconstructionCamera({...source,K:[[500,0,500],[0,NaN,400],[0,0,1]]}),null);
  assert.equal(reconstructionCamera({...source,width:0}),null);
});

test('singular camera matrix cannot silently create an invalid view', () => {
  assert.throws(()=>cameraWorldPose({...source,w2c:[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]}),/不可逆/);
});
