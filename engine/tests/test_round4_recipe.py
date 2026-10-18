"""Version boundary, explicit policy and immutable projection identity checks."""
import copy
import pytest
from local_postshot.training_recipe import recipe_config,recipe_from_checkpoint
from local_postshot.projection_policy import validated_projection,projection_artifact_identity


def test_v6_changes_only_tile_support_and_immutable_camera_cache_from_r13_defaults():
    old=recipe_config('observed_v5',10000,train_image_count=112)
    new=recipe_config('observed_v6',10000,train_image_count=112)
    assert new['projection']['tile_culling']=='conservative_v1'
    assert new['projection'].get('aa_compensation','public')=='public'
    assert new['projection']['train_rasterize_mode']=='classic'
    compare=copy.deepcopy(new)
    compare['name']=old['name']
    compare['unrecovered_original_behavior']=old['unrecovered_original_behavior']
    assert compare.pop('camera_center_cache') is True
    del compare['projection']['tile_culling']
    assert compare==old


def test_old_projection_and_checkpoint_are_not_enriched():
    old=recipe_config('observed_v5',10000,train_image_count=112)
    assert 'tile_culling' not in old['projection']
    assert 'aa_compensation' not in old['projection']
    assert projection_artifact_identity(old['projection'])=={}
    assert recipe_from_checkpoint({'training_recipe':old})==old


@pytest.mark.parametrize('changes',[{'tile_culling':'unknown'},{'aa_compensation':'unknown'},
    {'tile_culling':'conservative_v1','compositor':'gsplat'},
    {'aa_compensation':'native_floor_v1','compositor':'gsplat'}])
def test_invalid_or_unpaired_kernel_policies_fail(changes):
    with pytest.raises(ValueError):validated_projection(changes)


def test_native_support_and_aa_are_explicit_separate_opt_ins():
    p=validated_projection({'compositor':'native_v1','tile_culling':'native_exact_v1',
        'aa_compensation':'native_floor_v1','train_rasterize_mode':'antialiased'})
    assert p['tile_culling']=='native_exact_v1'
    assert p['train_rasterize_mode']=='antialiased'
    # The shader option itself does not secretly enable antialiasing at eval.
    assert p['eval_rasterize_mode'] is None and p['rasterize_mode']=='classic'
