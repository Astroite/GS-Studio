import assert from 'node:assert/strict';
import test from 'node:test';
import { JSDOM } from 'jsdom';
import React, { act } from 'react';
import { createServer } from 'vite';

const dom=new JSDOM('<!doctype html><div id="test-root"></div>',{url:'http://localhost/'});
globalThis.window=dom.window; globalThis.document=dom.window.document;
globalThis.HTMLElement=dom.window.HTMLElement; globalThis.IS_REACT_ACT_ENVIRONMENT=true;
const {createRoot}=await import('react-dom/client');
window.__TAURI_INTERNALS__={convertFileSrc:path=>`asset://localhost/${path}`};
globalThis.isTauri=true;
const vite=await createServer({configFile:false,root:process.cwd(),server:{middlewareMode:true},appType:'custom'});
const {useStudio}=await vite.ssrLoadModule('/src/useStudio.ts');
const {MaskReview}=await vite.ssrLoadModule('/src/MaskReview.tsx');
const {TrainingPreview,Training}=await vite.ssrLoadModule('/src/Training.tsx');
const {savedTrainingSettings}=await vite.ssrLoadModule('/src/trainingOptions.ts');
const {Workflow,SelectSharp,RunReview}=await vite.ssrLoadModule('/src/Workflow.tsx');
const {Catalog}=await vite.ssrLoadModule('/src/Catalog.tsx');
const {Inspector}=await vite.ssrLoadModule('/src/Inspector.tsx');
const {NativeEditor}=await vite.ssrLoadModule('/src/NativeEditor.tsx');
const flush=()=>new Promise(resolve=>setTimeout(resolve,0));
let root;
async function mount(element){root=createRoot(document.getElementById('test-root'));await act(async()=>{root.render(element);await flush();});}
async function click(element){assert(element,'control missing');await act(async()=>{element.dispatchEvent(new window.MouseEvent('click',{bubbles:true}));await flush();});}
async function change(element,value){await act(async()=>{const setter=Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype,'value').set;setter.call(element,value);element.dispatchEvent(new window.Event('change',{bubbles:true}));await flush();});}
async function cleanup(){await act(async()=>root?.unmount());root=undefined;document.getElementById('test-root').innerHTML='';}
test.afterEach(cleanup);
test.after(async()=>{await vite.close();dom.window.close();});

test('actual useStudio serializes writes but serves reads/control and preserves terminal errors',async()=>{
  let api,complete;const calls=[];let listener;
  const transport={available:true,subscribe:async cb=>{listener=cb;return()=>{};},call:async(method,params)=>{
    calls.push([method,params]);if(method==='studio.catalog')return{root:'E:/AI/GSStudio/projects',locations:[],scenes:[],captures:[],runs:[],external_imports:[]};
    if(params.action==='create_run')return await new Promise(resolve=>{complete=resolve;});
    if(params.action==='ingest')throw Error('hash identity mismatch');return{request_id:'control-id'};
  }};
  function Harness(){api=useStudio(false,transport);return React.createElement('span',null,api.busy??api.error);}
  await mount(React.createElement(Harness));let running;
  await act(async()=>{running=api.operate('create_run');await flush();});assert.equal(api.busy,'create_run');
  await act(async()=>{assert.equal(await api.operate('ingest'),null);await api.read('training_state',{output:'experiment'});await api.control({output:'experiment',control_action:'pause',package_sha256:'hash',generation_id:'generation'});});
  assert(calls.some(([,p])=>p?.action==='training_state'));assert(calls.some(([,p])=>p?.control_action==='pause'));
  await act(async()=>{listener({type:'event',event:'progress',data:{kind:'started',detail:{output:'exp',package_sha256:'hash'}}});complete({run_id:'run',status:'needs_mask_review'});await running;});
  assert.equal(api.busy,null);assert.equal(api.result.status,'needs_mask_review');
  await act(async()=>{await api.operate('ingest');});assert.match(api.error,/hash identity mismatch/);assert.equal(api.result.status,'needs_mask_review');await cleanup();
});

test('actual mask review requires user status and commits exact image/mask/dataset identities',async()=>{
  const requests=[];let committed=false;
  const studio={context:{location_id:'l',scene_id:'s',run_id:'r'},locked:false,report:e=>{throw e;},read:async(action,params)=>{
    requests.push([action,params]);if(action==='masks.review')return{dataset_identity:'dataset-hash',items:[{image_id:1,source_image:'one.jpg',review:{status:committed?'ok':'unreviewed'}}],total:1,summary:committed?{ok:1}:{unreviewed:1}};
    return{image_path:'image.png',mask_path:'mask.png',width:100,height:50,image_sha256:'image-hash',mask_sha256:'mask-hash',dataset_identity:'dataset-hash',review:{status:'unreviewed'},mask_semantics:'white_is_ignored'};
  },operate:async(action,params)=>{requests.push([action,params]);if(action==='masks.commit')committed=true;return{updated:[1]};}};
  await mount(React.createElement(MaskReview,{studio}));
  assert(!requests.some(([a])=>a==='masks.commit'||a==='finalize_masks'));
  const buttons=()=>[...document.querySelectorAll('button')];
  assert(buttons().find(b=>b.textContent.includes('固定 primary')).disabled);
  await click(buttons().find(b=>b.textContent.includes('one.jpg')));
  await change([...document.querySelectorAll('select')].find(s=>[...s.options].some(o=>o.value==='false_negative')),'ok');
  await click(buttons().find(b=>b.textContent.includes('提交当前审核')));
  const commit=requests.find(([a])=>a==='masks.commit')[1];assert.equal(commit.dataset_identity,'dataset-hash');assert.deepEqual(commit.edits,[{image_id:1,image_sha256:'image-hash',mask_sha256:'mask-hash',status:'ok',note:''}]);
  assert(buttons().find(b=>b.textContent.includes('固定 primary')).disabled);
  await click([...document.querySelectorAll('input[type=checkbox]')].at(-1));
  assert(!buttons().find(b=>b.textContent.includes('固定 primary')).disabled);assert(!requests.some(([a])=>a==='finalize_masks'));await cleanup();
});

test('actual paired preview discards stale camera response instead of labeling it current',async()=>{
  let lastRequest;let stale=true;
  const studio={read:async(action,params)=>{
    if(action==='sources.list')return{cameras:[{image:'a',source_image:'A'},{image:'b',source_image:'B'}]};
    if(action==='preview.request'){lastRequest=params;return params;}
    return{preview:{...lastRequest,image:stale?'wrong-camera':lastRequest.image,source_path:'source.png',render_path:'render.png',step:500}};
  }};
  await mount(React.createElement(TrainingPreview,{studio,output:'experiment',digest:'package-hash',loadFiles:async p=>[p.source_path,p.render_path]}));
  await change(document.querySelector('select'),'a');assert.equal(document.querySelectorAll('.gs-paired img').length,0);
  stale=false;await change(document.querySelector('select'),'b');assert.equal(document.querySelectorAll('.gs-paired img').length,2);assert.match(document.body.textContent,/step 500/);await cleanup();
});

test('actual native editor previews numeric/gizmo edits, commits once and cancels next preview with Escape',async()=>{
  let viewProps,lastPreview,revision=0,holdPreview=false,releasePreview;const actions=[],errors=[];
  let project={id:'p',name:'fixture',path:'E:/fixture',status:'model imported',created_at:'',updated_at:'',settings:{steps:30,max_splats:100,max_image_size:800,video_fps:1,max_frames:10},models:[{id:'m',name:'model',path:'E:/fixture/model.ply',visible:true,count:100,transform:{position:[0,0,0],rotation:[0,0,0],scale:1},crop:{enabled:false,min:[-1,-1,-1],max:[1,1,1],invert:false}}]};
  const camera={name:'viewport',width:800,height:600,K:[[400,0,400],[0,400,300],[0,0,1]],w2c:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]};
  project.reconstruction={cameras:[{...camera,name:'train-a'},{...camera,name:'train-b',w2c:[[1,0,0,2],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]};
  function Scene(props){viewProps=props;React.useEffect(()=>{props.apiRef.current={camera:()=>camera};},[]);return React.createElement('div',{'data-testid':'scene'});}
  const rpc=async(method,params)=>{
    assert.equal(method,'editor.operation');actions.push(params);
    if(params.action==='preview'){lastPreview=params.args;if(holdPreview){holdPreview=false;await new Promise(resolve=>{releasePreview=resolve;});}}
    if(params.action==='commit')project={...project,models:[{...project.models[0],transform:lastPreview.transform,crop:lastPreview.crop}]};
    if(params.action!=='render')++revision;
    return{project,selection_count:0,native_render:'E:/fixture/native.png',revision,can_undo:actions.some(v=>v.action==='commit'),can_redo:false};
  };
  function Harness(){const [current,setCurrent]=React.useState(project);const work={project:current,selectedId:'m',operation:null,report:e=>errors.push(e),acceptProject:setCurrent,save:async()=>{},setNotice:()=>{},setSelectedId:()=>{},exportModel:async()=>{}};return React.createElement(NativeEditor,{work,blocked:false,rpc,Scene});}
  await mount(React.createElement(Harness));
  assert.equal(viewProps.reference.camera.name,'train-a');
  await change(selectWith('1'),'1');
  assert.equal(viewProps.reference.camera.name,'train-b');assert.equal(viewProps.reference.camera.w2c[0][3],2);
  const chosenCameraVersion=viewProps.reference.version;
  const priorFrame={...viewProps.selection},priorEditRevision=revision;
  await act(async()=>{viewProps.onCameraChange();await flush();});
  assert.equal(viewProps.selection.nativeRender,'','a changed camera must hide old native pixels');
  await act(async()=>{await new Promise(resolve=>setTimeout(resolve,390));await flush();});
  assert.equal(revision,priorEditRevision,'camera-only render must not create an edit');
  assert.equal(viewProps.selection.nativeRender,priorFrame.nativeRender,'core may reuse the same render path');
  assert(viewProps.selection.revision>priorFrame.revision,'new camera pixels require a fresh image URL even when edit revision is unchanged');
  await change(document.querySelector('select[aria-label="编辑工具"]'),'translate');
  await act(async()=>{viewProps.gizmo.change({...viewProps.gizmo.transform,translation:[1,2,3]},viewProps.gizmo.crop);await flush();});
  await act(async()=>{await new Promise(resolve=>setTimeout(resolve,360));await flush();});
  assert.equal(actions.filter(v=>v.action==='preview').length,1);assert.equal(actions.filter(v=>v.action==='commit').length,0);assert.deepEqual(lastPreview.transform.position,[1,2,3]);
  await click([...document.querySelectorAll('button')].find(v=>v.textContent==='确认一次编辑'));
  assert.equal(actions.filter(v=>v.action==='commit').length,1);
  assert.equal(viewProps.reference.version,chosenCameraVersion,'model commits must not reset the chosen training view');
  holdPreview=true;
  await act(async()=>{viewProps.gizmo.change({...viewProps.gizmo.transform,scale:2},viewProps.gizmo.crop);await flush();});
  await act(async()=>{await new Promise(resolve=>setTimeout(resolve,360));await flush();});
  await act(async()=>{viewProps.gizmo.change({...viewProps.gizmo.transform,scale:3},viewProps.gizmo.crop);await flush();});
  await act(async()=>{releasePreview();await flush();});
  assert([...document.querySelectorAll('button')].find(v=>v.textContent==='确认一次编辑').disabled,'superseded preview must never enable commit for the newer draft');
  await act(async()=>{window.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true}));await flush();});
  assert.equal(actions.filter(v=>v.action==='cancel_preview').length,2);assert.equal(actions.filter(v=>v.action==='commit').length,1);assert.equal(errors.length,0);await cleanup();
});

function optionFixture(params, source='colmap') {
  const preset=params.training_preset==='auto'?(source==='colmap'||source==='perspective_images'?'photo':'gs'):params.training_preset;
  const requested=params.use_sparse_depth??preset==='gs';
  return {package:`E:/fixture/${params.import_id??params.segment}/${preset}`,package_sha256:(preset==='photo'?'a':'b').repeat(64),source_kind:source,
    requested_training_preset:params.training_preset,training_preset:preset,training_recipe:params.training_recipe,steps:params.steps??(preset==='photo'?10000:30000),
    sh_degree:3,max_splats:1000000,seed:42,max_image_size:preset==='photo'?1600:0,
    photo_comp:params.photo_comp??preset==='gs',use_bilateral_grid:params.use_bilateral_grid??preset==='gs',
    use_sparse_depth:false,sparse_depth_requested:requested,sparse_depth_available:false,sparse_depth_observations:0,
    sparse_depth_status:requested?'unavailable':'disabled',sparse_depth_reason:'No validated sparse-depth anchors in this package'};
}
function trainingStudio(readOptions) {
  const calls=[],errors=[];
  const studio={context:{import_id:'photos'},catalog:{root:'E:/configured-projects',external_imports:[{id:'photos'},{id:'video'}],captures:[],runs:[]},locked:false,result:{},
    report:e=>errors.push(e),read:async(action,params={})=>{calls.push([action,params]);
      if(action==='training_options')return readOptions?readOptions(params):optionFixture(params);
      if(action==='segment_choices')return{segments:[{id:'segment-001'},{id:'segment-002'}],training_status:'passed',coverage_status:'incomplete'};
      if(action==='list_external_imports')return{imports:[]};return{};
    },operate:async(action,params)=>{calls.push([action,params]);return{};}};
  return{studio,calls,errors};
}
const button=text=>[...document.querySelectorAll('button')].find(b=>b.textContent===text);
const selectWith=value=>[...document.querySelectorAll('select')].find(s=>[...s.options].some(o=>o.value===value));

test('photo training resolves before submit, omits untouched aux overrides and labels recipes truthfully',async()=>{
  const{studio,calls}=trainingStudio();
  await mount(React.createElement(Training,{studio,openModel:async()=>{}}));
  await click(button('新建实验'));
  assert.match(document.body.textContent,/已解析 photo.*10000 步/s);assert.match(document.body.textContent,/最大边 1600 px/);
  const depth=[...document.querySelectorAll('input[type=checkbox]')][0];assert(depth.disabled);assert(!depth.checked);assert.match(document.body.textContent,/No validated sparse-depth anchors/);
  assert.equal(selectWith('quality_v2').value,'quality_v2');assert.match(selectWith('observed_v5').options[1].textContent,/已发布/);
  assert(![...selectWith('quality_v2').options].some(o=>['gs_studio','observed_v3','observed_v4','legacy_adc'].includes(o.value)));
  await click(button('创建 GS 训练实验'));
  const submit=calls.find(([a])=>a==='train_external_import')[1];assert.equal(calls.filter(([a])=>a==='training_options').length,2);
  for(const k of ['steps','photo_comp','use_bilateral_grid','use_sparse_depth'])assert.equal(submit[k],undefined,k);
  assert.equal(submit.root,'E:/configured-projects');assert.equal(submit.training_preset,'auto');assert.equal(submit.training_recipe,'quality_v2');assert.equal(submit.experiment_id,undefined);
  await change(selectWith('gs'),'gs');assert.match(document.body.textContent,/已解析 gs.*30000 步/s);assert.match(document.body.textContent,/保留源图分辨率/);
  await change(selectWith('bilateral'),'camera');await change(selectWith('observed_v5'),'observed_v5');await click(button('创建 GS 训练实验'));
  const explicit=calls.filter(([a])=>a==='train_external_import').at(-1)[1];assert.equal(explicit.photo_comp,true);assert.equal(explicit.use_bilateral_grid,false);assert.equal(explicit.use_sparse_depth,undefined);assert.equal(explicit.training_recipe,'observed_v5');
  await change(selectWith('bilateral'),'bilateral');await click(button('创建 GS 训练实验'));
  const bilateral=calls.filter(([a])=>a==='train_external_import').at(-1)[1];assert.equal(bilateral.photo_comp,true,'Bilateral Grid requires the core photometric master switch');assert.equal(bilateral.use_bilateral_grid,true);
  assert.match(document.body.textContent,/双边网格（相机补偿不叠加）/);
  await change(selectWith('off'),'off');await click(button('创建 GS 训练实验'));
  const off=calls.filter(([a])=>a==='train_external_import').at(-1)[1];assert.equal(off.photo_comp,false);assert.equal(off.use_bilateral_grid,false);assert.match(document.body.textContent,/光度补偿关闭/);await cleanup();
});

test('training options discard late imports and presets and cannot submit after source changes during preflight',async()=>{
  const pending=[];const{studio,calls}=trainingStudio(params=>new Promise(resolve=>pending.push({params,resolve})));
  await mount(React.createElement(Training,{studio,openModel:async()=>{}}));await click(button('新建实验'));assert(button('创建 GS 训练实验').disabled);
  studio.context={import_id:'video'};
  await act(async()=>{root.render(React.createElement(Training,{studio,openModel:async()=>{}}));await flush();});
  await click(button('新建实验'));
  assert.equal(pending.length,1,'package preparation is serialized');
  await act(async()=>{pending[0].resolve(optionFixture(pending[0].params));await flush();});assert(button('创建 GS 训练实验').disabled);assert.doesNotMatch(document.body.textContent,/已解析 photo/);
  await act(async()=>{pending[1].resolve(optionFixture(pending[1].params,'perspective_video'));await flush();});assert.match(document.body.textContent,/已解析 gs/);
  await change(selectWith('photo'),'photo');assert(button('创建 GS 训练实验').disabled);await change(selectWith('gs'),'gs');
  await act(async()=>{pending[2].resolve(optionFixture(pending[2].params));await flush();});assert(button('创建 GS 训练实验').disabled);
  await act(async()=>{pending[3].resolve(optionFixture(pending[3].params,'perspective_video'));await flush();});assert.match(document.body.textContent,/已解析 gs/);
  await click(button('创建 GS 训练实验'));assert.equal(pending.length,5);
  studio.context={import_id:'photos'};
  await act(async()=>{root.render(React.createElement(Training,{studio,openModel:async()=>{}}));await flush();});
  await act(async()=>{pending[4].resolve(optionFixture(pending[4].params,'perspective_video'));await flush();});
  assert(!calls.some(([a])=>a==='train_external_import'));assert.equal(button('创建 GS 训练实验'),undefined);await cleanup();
});

test('available depth remains off for untouched photos and only explicit user enable sends true',async()=>{
  const{studio,calls}=trainingStudio(params=>{const base=optionFixture(params);return{...base,sparse_depth_available:true,sparse_depth_observations:17,sparse_depth_reason:null,use_sparse_depth:base.sparse_depth_requested,sparse_depth_status:base.sparse_depth_requested?'enabled':'disabled'};});
  await mount(React.createElement(Training,{studio,openModel:async()=>{}}));
  await click(button('新建实验'));
  const depth=()=>[...document.querySelectorAll('label')].find(v=>v.textContent.includes('启用已验证的稀疏深度锚')).querySelector('input');assert(!depth().disabled);assert(!depth().checked);
  await click(depth());assert(depth().checked);await click(button('创建 GS 训练实验'));
  assert.equal(calls.find(([a])=>a==='train_external_import')[1].use_sparse_depth,true);await click(button('深度恢复预设默认值'));assert(!depth().checked);await cleanup();
});

test('segment changes invalidate options and changed package preflight requires another reviewed submit',async()=>{
  let late,changed=false;const{studio,calls,errors}=trainingStudio(params=>params.segment==='segment-001'?new Promise(resolve=>{late=()=>resolve(optionFixture(params));}):{...optionFixture(params),package_sha256:(changed?'c':'a').repeat(64)});
  studio.context={location_id:'l',scene_id:'s',run_id:'r',capture_id:'c'};
  await mount(React.createElement(Training,{studio,openModel:async()=>{}}));
  await change(selectWith('segment-001'),'segment-001');await click(button('新建实验'));assert(button('创建 GS 训练实验').disabled);
  await change(selectWith('segment-002'),'segment-002');await click(button('新建实验'));assert(button('创建 GS 训练实验').disabled);
  await act(async()=>{late();await flush();});assert.doesNotMatch(document.body.textContent,/fixture\/segment-001/);assert.match(document.body.textContent,/fixture\/segment-002/);
  changed=true;await click(button('创建 GS 训练实验'));assert(!calls.some(([a])=>a==='train_segment'));assert.match(errors.at(-1).message,/训练包或预算已变化/);
  await click(button('创建 GS 训练实验'));assert.equal(calls.filter(([a])=>a==='train_segment').length,1);assert.equal(calls.find(([a])=>a==='train_segment')[1].segment,'segment-002');await cleanup();
});

test('resume restores requested preset/depth instead of substituting effective unavailable flag',()=>{
  const settings=savedTrainingSettings({requested_training_preset:'auto',training_preset:'gs',requested_steps:30000,sparse_depth_requested:true,use_sparse_depth:false,training_recipe:'observed_v5'},{});
  assert.equal(settings.training_preset,'auto');assert.equal(settings.use_sparse_depth,true);assert.equal(settings.training_recipe,'observed_v5');
});

const existingOutput='E:\\fixture\\projects\\l\\s\\r\\experiments\\gs-representative-001';
const secondOutput='E:\\fixture\\projects\\l\\s\\r\\experiments\\second';
function existingTrainingState(output=existingOutput,status='running',generation='generation-one') {
  return {state:{status,generation_id:generation,step:125},
    dispatch:{output,package:'E:/fixture/immutable-package',package_sha256:'b'.repeat(64),requested_training_preset:'auto',training_preset:'gs',
      requested_steps:30000,training_recipe:'observed_v5',photo_comp:true,use_bilateral_grid:true,sparse_depth_requested:true,use_sparse_depth:false,max_image_size:0,max_splats:1000000,sh_degree:3,seed:42},
    config:{steps:30000,training_recipe:'observed_v5'},training:{status,checkpoint_sha256:'checkpoint-hash',completed_steps:100},
    log_tail:JSON.stringify({type:'event',event:'metric',data:{step:125,loss:0.12345,splats:45678,elapsed_seconds:73.9}}),metrics:{}};
}
function existingStudio() {
  const {studio,calls,errors}=trainingStudio(params=>optionFixture(params,'perspective_video'));
  studio.context={location_id:'l',scene_id:'s',run_id:'r',capture_id:'c'};
  const records=[{output:existingOutput,segment:'segment-001',status:'preparing'},{output:secondOutput,segment:'segment-002',status:'running'}];
  const defaultRead=studio.read;
  studio.read=async(action,params={})=>{
    if(action==='experiments'){calls.push([action,params]);return{experiments:records};}
    if(action==='training_state'){calls.push([action,params]);return existingTrainingState(params.output);}
    return defaultRead(action,params);
  };
  studio.control=async params=>{calls.push(['control_training',params]);return{request_id:'control-request'};};
  return{studio,calls,errors,records};
}

test('existing running experiment stays read-only through mount, polls and selection; controls retain package and generation',async()=>{
  const {studio,calls,errors}=existingStudio();studio.locked=true;
  let poll;const originalInterval=window.setInterval;
  window.setInterval=(fn,delay,...args)=>{if(delay===2000)poll=fn;return originalInterval.call(window,fn,delay,...args);};
  const render=()=>React.createElement(Training,{studio,focus:{segment:'segment-001',output:existingOutput},openModel:async()=>{}});
  try {
    await mount(render());
    assert.equal(calls.filter(([a])=>a==='training_options').length,0);
    assert.equal(button('创建 GS 训练实验'),undefined);assert(!button('保存检查点').disabled);
    const current=document.querySelector('section[aria-label="当前实验"]');assert.match(current.textContent,/运行中/);
    assert.match(current.textContent,/125 \/ 30,000/);assert.match(current.textContent,/1 分 13 秒/);assert.match(current.textContent,/45,678/);
    assert.match(current.textContent,/已保存的解析配置.*gs \/ observed_v5/s);assert.match(current.textContent,/双边网格/);
    for(const pre of current.querySelectorAll('pre'))assert.equal(pre.closest('details').open,false,'raw diagnostics/logs are collapsed');
    await click(button('保存检查点'));
    assert.deepEqual(calls.find(([a])=>a==='control_training')[1],{output:existingOutput,control_action:'checkpoint',package_sha256:'b'.repeat(64),generation_id:'generation-one'});
    studio.read=async(action,params={})=>{calls.push([action,params]);return action==='training_state'?existingTrainingState(params.output,'running','generation-two'):{};};
    await act(async()=>{await poll();await flush();});
    assert.equal(calls.filter(([a])=>a==='training_options').length,0);
    // A pending old request must not be carried into a different experiment.
    let resolveSecond;
    studio.read=async(action,params={})=>{calls.push([action,params]);return action==='training_state'?new Promise(resolve=>{resolveSecond=resolve;}):{};};
    await change(selectWith(secondOutput),secondOutput);assert(button('保存检查点').disabled);
    await act(async()=>{resolveSecond({...existingTrainingState(secondOutput,'running','second-generation'),dispatch:{package_sha256:'c'.repeat(64)}});await flush();});
    await click(button('保存检查点'));
    assert.deepEqual(calls.filter(([a])=>a==='control_training').at(-1)[1],{output:secondOutput,control_action:'checkpoint',package_sha256:'c'.repeat(64),generation_id:'second-generation'});
    assert.equal(calls.filter(([a])=>a==='training_options').length,0);assert.equal(errors.length,0);
  } finally {window.setInterval=originalInterval;await cleanup();}
});

test('selecting a segment with an active experiment waits for explicit New, which prepares and revalidates exact new submission',async()=>{
  const {studio,calls,errors}=existingStudio();
  await mount(React.createElement(Training,{studio,focus:{segment:'segment-001'},openModel:async()=>{}}));
  assert(calls.some(([a])=>a==='experiments'));assert.equal(calls.filter(([a])=>a==='training_options').length,0);
  await change(selectWith(existingOutput),existingOutput);
  assert.equal(calls.filter(([a])=>a==='training_options').length,0);
  await click(button('新建实验'));
  const options=calls.filter(([a])=>a==='training_options');assert.equal(options.length,1);
  assert.deepEqual(options[0][1],{root:'E:/configured-projects',location_id:'l',scene_id:'s',run_id:'r',capture_id:'c',import_id:undefined,segment:'segment-001',training_preset:'auto',training_recipe:'quality_v2',sh_degree:3});
  await click(button('创建 GS 训练实验'));
  assert.equal(calls.filter(([a])=>a==='training_options').length,2,'new submission repeats the serialized full preflight');
  assert.deepEqual(calls.find(([a])=>a==='train_segment')[1],{...options[0][1],output:undefined,resume:false,restart_from_step_zero:false});
  assert.equal(errors.length,0);await cleanup();
});

test('resume performs no eager options request and preflights the selected saved segment, overrides and output',async()=>{
  const {studio,calls}=existingStudio();const defaultRead=studio.read;
  studio.read=async(action,params={})=>action==='training_state'?(calls.push([action,params]),existingTrainingState(params.output,'paused')):defaultRead(action,params);
  await mount(React.createElement(Training,{studio,focus:{segment:'segment-002',output:existingOutput},openModel:async()=>{}}));
  assert.equal(calls.filter(([a])=>a==='training_options').length,0);
  await click(button('检查并恢复已保存实验'));
  const preflight=calls.find(([a])=>a==='training_options')[1],submit=calls.find(([a])=>a==='train_segment')[1];
  assert.equal(preflight.segment,'segment-001','resume binds the recorded segment rather than the new-form selection');
  assert.equal(preflight.training_recipe,'observed_v5');assert.equal(preflight.steps,30000);
  assert.equal(preflight.use_sparse_depth,true,'restore the requested setting even when effective depth is unavailable');
  assert.equal(preflight.photo_comp,true);assert.equal(preflight.use_bilateral_grid,true);
  assert.deepEqual(submit,{...preflight,output:existingOutput,resume:true,restart_from_step_zero:false});await cleanup();
});

test('external existing experiment selection and catalog refresh never prepare a new package',async()=>{
  const {studio,calls}=existingStudio();studio.context={import_id:'photos'};
  const defaultRead=studio.read;
  studio.read=async(action,params={})=>action==='list_external_imports'?(calls.push([action,params]),{imports:[{id:'photos',experiments:[{id:'saved',output:existingOutput,status:'running'}]}]}):defaultRead(action,params);
  await mount(React.createElement(Training,{studio,focus:{output:existingOutput},openModel:async()=>{}}));
  await click(button('读取实验清单'));
  assert.equal(calls.filter(([a])=>a==='training_options').length,0);assert(!button('保存检查点').disabled);
  await change(selectWith(existingOutput),'');assert.equal(calls.filter(([a])=>a==='training_options').length,0);
  await click(button('新建实验'));assert.equal(calls.filter(([a])=>a==='training_options').length,1);await cleanup();
});

test('missing metric evidence shows unknown values without invented progress or saved defaults',async()=>{
  const {studio}=existingStudio();const read=studio.read;
  studio.read=async(action,params={})=>action==='training_state'?{}:read(action,params);
  await mount(React.createElement(Training,{studio,focus:{output:existingOutput},openModel:async()=>{}}));
  const measured=document.querySelector('[aria-label="实测训练指标"]');assert.match(measured.textContent,/— \/ —/);assert.equal(measured.querySelector('progress'),null);
  assert(!document.querySelector('section[aria-label="当前实验"]').textContent.includes('30,000 步'));
  assert(button('保存检查点').disabled);await cleanup();
});

test('catalog experiment without id displays a concise Windows basename, separate status and unchanged output selection',async()=>{
  const {studio,records}=existingStudio();let selection;
  const row={id:'r',location_id:'l',scene_id:'s',capture_id:'c'};
  studio.catalog={...studio.catalog,locations:[{id:'l',display_name:'地点'}],scenes:[{id:'s',location_id:'l'}],captures:[{id:'c',location_id:'l',scene_id:'s'}],runs:[row]};
  await mount(React.createElement(Catalog,{studio,work:{projects:[]},go:()=>{},chooseRun:()=>{},chooseTraining:(...args)=>{selection=args;}}));
  const runSummary=[...document.querySelectorAll('summary')].find(v=>v.querySelector('.gs-tree-label')?.textContent==='r');
  await act(async()=>{runSummary.parentElement.open=true;runSummary.parentElement.dispatchEvent(new window.Event('toggle'));await flush();});
  const experiment=document.querySelector(`button[title="${existingOutput.replaceAll('\\','\\\\')}"]`);
  assert(experiment);assert.equal(experiment.querySelector('.gs-tree-label').textContent,'实验 gs-representative-001');
  assert.equal(experiment.querySelector('small').textContent,'准备中');assert(!experiment.textContent.includes('E:'));
  assert(experiment.getAttribute('aria-label').includes(existingOutput));
  await click(experiment);assert.deepEqual(selection,[row,'segment-001',existingOutput]);assert.equal(records[0].output,existingOutput);await cleanup();
});

test('actual Run QA card summarizes recorded stages/attempt/problems and keeps report and approval semantics',async()=>{
  const calls=[],run={id:'r',location_id:'l',scene_id:'s',status:'accepted',stages:{preprocess:'succeeded',mask:'succeeded',reconstruct:'succeeded',qa:'succeeded'},
    selected_dataset:'r/reconstruction-repair',manifest_path:'E:/fixture/manifest.yaml',config_hash:'hash-evidence',problems:['覆盖缺口仍需检查']};
  const studio={context:{location_id:'l',scene_id:'s',run_id:'r'},catalog:{locations:[{id:'l'}],scenes:[{id:'s',location_id:'l'}],runs:[run]},locked:false,result:{},setContext:()=>{},report:e=>{throw e;},
    read:async(action,params)=>{calls.push([action,params]);return action==='qa_report'?{path:'E:/fixture/qa.md',manifest_status:'accepted',text:'实际报告内容'}:{training_status:'passed',coverage_status:'incomplete',segments:[]};},
    operate:async(action,params)=>{calls.push([action,params]);return{};}};
  await mount(React.createElement(RunReview,{studio,goMasks:()=>{},goTraining:()=>{}}));
  const summary=document.querySelector('.gs-card');assert.match(summary.textContent,/预处理.*已完成.*遮罩.*已完成.*重建.*已完成.*质量检查（QA）.*已完成/s);
  assert.match(summary.textContent,/repair · 修复重建/);assert.match(summary.textContent,/覆盖缺口仍需检查/);
  const diagnostics=summary.querySelector('details');assert(!diagnostics.open);assert(diagnostics.textContent.includes('hash-evidence'));assert(diagnostics.textContent.includes('E:/fixture/manifest.yaml'));
  assert(button('提交不接受决定').disabled);assert.equal(calls.length,0);
  await click(button('读取分段 QA 与覆盖证据'));assert.match(document.body.textContent,/全路线覆盖不完整/);
  await click(button('读取真实 QA 报告'));assert(document.body.textContent.includes('实际报告内容'));assert(!calls.some(([a])=>a==='review_qa'));
  const textarea=document.querySelector('form textarea');
  await act(async()=>{Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set.call(textarea,'人工测试说明');textarea.dispatchEvent(new window.Event('input',{bubbles:true}));textarea.dispatchEvent(new window.Event('change',{bubbles:true}));await flush();});
  await click(document.querySelector('form input[type=checkbox]'));
  await act(async()=>{document.querySelector('form').dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));await flush();});
  assert.deepEqual(calls.find(([a])=>a==='review_qa'),['review_qa',{accepted:true,notes:'人工测试说明'}]);await cleanup();
});

test('initial catalog uses configured data root, then pins operations to returned root',async()=>{
  let api;const calls=[];const transport={available:true,subscribe:async()=>()=>{},call:async(method,params)=>{calls.push([method,params]);return method==='studio.catalog'?{root:'F:/relocated/projects',locations:[],runs:[]}:{};}};
  function Harness(){api=useStudio(false,transport);return null;}
  await mount(React.createElement(Harness));assert.deepEqual(calls[0],['studio.catalog',{}]);await act(async()=>{await api.read('capture_config');});assert.equal(calls.at(-1)[1].root,'F:/relocated/projects');await cleanup();
});

test('intake exposes every pipeline source and photo directory controls; select-sharp controls remain',async()=>{
  const studio={context:{},catalog:{locations:[],scenes:[],captures:[]},locked:false,result:{}};
  await mount(React.createElement(Workflow,{studio}));
  for(const source of ['insta360_insv','equirect_video','equirect_sequence','perspective_video','perspective_images'])assert(selectWith(source));
  await change(selectWith('perspective_images'),'perspective_images');assert.match(document.body.textContent,/照片序列目录/);assert.match(document.body.textContent,/每行一个/);await cleanup();
  await mount(React.createElement(SelectSharp,{studio}));for(const label of ['源文件','输出目录','候选 fps','投影视图数','保留候选'])assert(document.body.textContent.includes(label),label);await cleanup();
});

test('standalone recipe entry defaults quality_v2 and exposes released v5 and experimental v6 only',async()=>{
  await mount(React.createElement(Inspector,{work:{project:{settings:{},models:[]},operation:null,projectBusy:false},settingsOnly:true}));
  const recipes=document.querySelector('select[aria-label="训练配方"]');assert.equal(recipes.value,'quality_v2');assert.deepEqual([...recipes.options].map(o=>o.value).sort(),['observed_v5','observed_v6','quality_v2']);assert.match(recipes.options[0].textContent,/已发布/);await cleanup();
});

test('native clean calls authoritative committed model and retains derivative history without baked reimport',async()=>{
  const actions=[];let accepted,selected;
  const source={id:'m',name:'edited',path:'E:/fixture/raw.ply',visible:true,count:4,editor:{revision:4},transform:{position:[1,2,3],rotation:[0,0,0],scale:2},crop:{enabled:true,min:[-1,-1,-1],max:[1,1,1],invert:false}};
  const project={id:'p',name:'fixture',path:'E:/fixture',settings:{},models:[source],reconstruction:{cameras:[{}]}};
  function Scene(){return null;}
  const rpc=async(method,params)=>{actions.push([method,params]);if(method==='model.clean')return{...project,models:[{...source,visible:false},{...source,id:'clean',editor:{revision:5}}]};return{project,selection_count:0,revision:4};};
  await mount(React.createElement(NativeEditor,{work:{project,selectedId:'m',report:e=>{throw e;},save:async()=>{},acceptProject:p=>{accepted=p;},setSelectedId:id=>{selected=id;},setNotice:()=>{}},blocked:false,rpc,Scene}));
  await click(button('按训练相机生成清理副本'));assert.equal(actions.find(([m])=>m==='model.clean')[1].model_id,'m');assert(!actions.some(([m])=>['model.export','model.import','project.update'].includes(m)));assert.equal(selected,'clean');assert.equal(accepted.models[1].editor.revision,5);assert.equal(accepted.models[1].path,source.path);await cleanup();
});
