"""Integration checks for the pinned optional baseline dependencies."""
import numpy as np
import pytest
import torch
from experiments.data import Patches
from experiments.external import make_method,preprocess
from experiments.run import parse_args


def options(name):
    return parse_args(['--model',name,'--source','Houston13','--target','Houston18',
                       '--device','cpu','--batch-size','4'])


@pytest.mark.parametrize('model',['CLDA','SCLUDA'])
@pytest.mark.parametrize('collapsed',[False,True])
def test_cleanlab_refresh_and_late_step(model,collapsed):
    pytest.importorskip('cleanlab')
    rng=np.random.default_rng(4)
    image=rng.random((14,14,48),dtype=np.float32)
    labels=np.tile(np.arange(7),28).reshape(14,14)
    image[...,0]=labels/6
    coords=np.argwhere(labels>=0);a=options(model);m=make_method(a)
    src=Patches(image,coords,a.patch_size,labels);tar=Patches(image,coords,a.patch_size)
    old=m.predict
    # Synthetic feature-encoded class predictions exercise both Cleanlab paths.
    def predict(x):
        y=torch.zeros(len(x),dtype=torch.long) if collapsed else (x[:,0,a.patch_size//2,a.patch_size//2]*6).round().long()
        return torch.nn.functional.one_hot(y,7).float()*10
    m.predict=predict
    m.prepare_epoch(40,src,tar,torch.device('cpu'),0)
    m.predict=old
    assert len(m.clean_mask)==len(tar)
    if collapsed:assert not m.clean_mask.any()
    else:assert m.clean_mask.any()
    v=m.step(torch.rand(4,48,a.patch_size,a.patch_size),torch.arange(4),
             torch.rand(4,48,a.patch_size,a.patch_size),torch.arange(4),40)
    assert np.isfinite(v['loss'])


def test_mlu_image_preprocessing():
    cv2=pytest.importorskip('cv2')
    if not hasattr(cv2,'ximgproc'):pytest.skip('contrib OpenCV required')
    x=np.random.default_rng(9).random((12,13,48),dtype=np.float32)
    s,t=preprocess(options('MLUDA'),x,x.copy())
    assert s.shape==x.shape and t.shape==x.shape
    assert np.isfinite(s).all() and np.isfinite(t).all()
