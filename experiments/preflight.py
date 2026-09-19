"""Check imports/ABI, optional-baseline APIs, dataset counts and chosen accelerator."""
import argparse
import importlib
import sys
import numpy as np
import torch
from .data import load_scene


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu',action='store_true');p.add_argument('--data-dir',default='Houston')
    a=p.parse_args()
    for name in ('numpy','scipy','sklearn','h5py','pandas','cv2','skimage','cleanlab','einops','pytest'):
        m=importlib.import_module(name)
        print(name,getattr(m,'__version__','available'))
    import cv2
    from cleanlab.filter import find_label_issues
    if not hasattr(cv2,'ximgproc'):raise RuntimeError('Install the contrib OpenCV wheel in requirements-fair.txt')
    # Exercise compiled sklearn/scipy/NumPy and the cleanlab API, not just imports.
    truth=np.tile(np.arange(7),4);prob=np.full((28,7),.01);prob[np.arange(28),truth]=.94
    assert not find_label_issues(truth,prob,n_jobs=1).any()
    print('Python',sys.version.split()[0],'Torch',torch.__version__,'CUDA',torch.version.cuda)
    if not a.cpu:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; install the wheel matching your GPU/driver, or use --cpu only for local checks')
        print('GPU',torch.cuda.get_device_name(0))
        x=torch.randn(4,4,device='cuda');assert torch.isfinite(x@x).all()
    for name in ('Houston13','Houston18'):
        _,gt,_=load_scene(a.data_dir,name)
        print(name,'evaluation pixels',int((gt>=0).sum()))
    print('Preflight passed')


if __name__=='__main__':main()
