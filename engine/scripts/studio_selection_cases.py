"""Parent-scheduled CUDA ROI parity against actual frozen pixel compositors."""
from pathlib import Path


def run_cases(output):
    from local_postshot.studio_bridge import gpu_session
    with gpu_session():
        return _run_cases(output)


def _run_cases(output):
    import numpy as np
    import torch
    from PIL import Image
    from local_postshot.training import tensors_from_cloud,render_parameters
    from local_postshot.studio_edit import native_selection,project_points
    from local_postshot.studio_selection import roi_max
    from local_postshot.projection_policy import validated_projection
    from local_postshot.splat3_renderer import rasterize_projected
    from gsplat.cuda._wrapper import rasterize_to_pixels
    output=Path(output);output.mkdir(exist_ok=False)
    camera={'width':128,'height':96,'K':[[80.,0,64.],[0,80.,48.],[0,0,1]],'w2c':np.eye(4).tolist()}
    cloud={'means':np.array([[1.3,.45,2.],[-.8,-.4,2.],[.8,.2,3.],[.8,.2,3.4],[1.3,.45,2.5]],np.float32),
           'scales':np.log(np.array([[.75,.025,.3],[.3,.04,.07],[.11,.11,.11],[.08,.08,.08],[.05,.05,.05]],np.float32)),
           'quats':np.tile(np.array([1.,0,0,0],np.float32),(5,1)),
           'opacities':np.array([2.,.5,8.,3.,3.],np.float32),'sh':np.zeros((5,1,3),np.float32)}
    # Tilt the long axis into camera z to exercise FOV covariance clipping.
    cloud['quats'][0]=[.9238795,0,.3826834,0]
    reports=[]
    for native in (False,True):
        policy=validated_projection({'compositor':'native_v1'}) if native else None
        with torch.no_grad():
            image,_,meta=render_parameters(tensors_from_cloud(cloud),camera,0,projection_policy=policy)
            def pixel_weights(m,bounds):
                left,top,right,bottom=bounds
                n=m['means2d'].shape[1];values=[]
                for g in range(n):
                    colors=torch.zeros((1,n,3),device='cuda');colors[0,g]=1
                    fn=rasterize_projected if native else rasterize_to_pixels
                    rendered,_=fn(means2d=m['means2d'],conics=m['conics'],colors=colors,
                                  opacities=m['opacities'],isect_offsets=m['isect_offsets'],flatten_ids=m['flatten_ids'],
                                  **({'width':m['width'],'height':m['height']} if native else
                                     {'image_width':m['width'],'image_height':m['height']}),tile_size=m['tile_size'])
                    values.append(rendered[0,top:bottom,left:right,0].max())
                return torch.stack(values)
            # Irregular rectangles cut through tiles, away from optical center.
            for bounds in ((95,51,127,83),(7,12,44,47),(0,0,128,96),(84,49,88,56)):
                expected=pixel_weights(meta,bounds)
                actual=roi_max(meta,bounds)
                np.savez(output/('offaxis-'+str(native)+'-'+'-'.join(map(str,bounds))+'.npz'),
                         expected=expected.cpu().numpy(),actual=actual.cpu().numpy())
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                selected,display=native_selection(cloud,camera,list(bounds),policy,tile_size=13)
                xy,z=project_points(cloud['means'],camera)
                left,top,right,bottom=bounds
                centers=(z>0)&(xy[:,0]>=left)&(xy[:,0]<=right)&(xy[:,1]>=top)&(xy[:,1]<=bottom)
                np.testing.assert_array_equal(selected,centers & (expected.cpu().numpy()>=np.float32(1/255)))
                np.testing.assert_array_equal(display,image[0].clamp(0,1).cpu().numpy())
                reports.append({'native':native,'bounds':bounds,'max_abs':0.,'selected':int(selected.sum())})
            Image.fromarray((image[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(output/f'offaxis-{native}.png')
            # Direct metadata keeps opacities exactly on either side of cutoff;
            # sigmoids/AA would move the threshold being tested.
            threshold=np.float32(1/255)
            opacity=torch.tensor([[np.nextafter(threshold,np.float32(0)),threshold,
                                   np.nextafter(threshold,np.float32(1)),.99,.999,.9]],device='cuda')
            m={'means2d':torch.tensor([[[8.5,8.5],[24.5,8.5],[40.5,8.5],[56.5,8.5],[56.5,8.5],[56.5,8.5]]],device='cuda'),
               'conics':torch.tensor([[[1.,0,1.]]*6],device='cuda'),'opacities':opacity,
               'isect_offsets':torch.tensor([[[0,1,2,3]]],device='cuda',dtype=torch.int32),
               'flatten_ids':torch.arange(6,device='cuda',dtype=torch.int32),'width':64,'height':16,'tile_size':16,
               'compositor':'native_renderer_v1' if native else 'gsplat'}
            for bounds in ((8,8,9,9),(24,8,25,9),(40,8,41,9),(56,8,57,9)):
                expected=pixel_weights(m,bounds);actual=roi_max(m,bounds)
                np.savez(output/('threshold-'+str(native)+'-'+'-'.join(map(str,bounds))+'.npz'),
                         expected=expected.cpu().numpy(),actual=actual.cpu().numpy())
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            weights=roi_max(m,(0,0,64,16))
            assert (weights[:3]>=threshold).tolist()==[False,True,True]
            # Occlusion termination is a center-pixel condition; Gaussian tails
            # in a full-frame maximum legitimately contribute in both renderers.
            weights=roi_max(m,(56,8,57,9))
            assert bool(weights[4]>=threshold)==native
            assert weights[5]==0
            reports.append({'native':native,'threshold_and_occlusion':'passed'})
            class Cancel:
                def is_set(self):return True
            try:native_selection(cloud,camera,[0,0,128,96],policy,Cancel())
            except InterruptedError:pass
            else:raise AssertionError('Selection did not honor cancellation')
            class CancelAfterChunk:
                calls=0
                def is_set(self):
                    self.calls+=1
                    return self.calls>2
            try:native_selection(cloud,camera,[0,0,128,96],policy,CancelAfterChunk(),tile_size=8)
            except InterruptedError:pass
            else:raise AssertionError('Selection did not cancel between bounded chunks')
            assert torch.cuda.current_stream().query(),'Cancellation left queued work after GPU-lock release'
    return reports
