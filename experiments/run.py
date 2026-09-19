"""Fixed-budget, final-checkpoint, transductive Houston UDA benchmark."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import CLASS_NAMES, Patches, load_scene, source_split, seed_worker, digest

MODELS = ['GAHT', 'BiDA', 'AgentBiDA', 'SelfAttentionAgentBiDA',
          'TSTnet', 'CLDA', 'SCLUDA', 'MLUDA', 'SSWADA', 'CACL']
PROTOCOL = 'houston-uda-final-v1'
ABLATIONS = ['full', 'source_only', 'no_mmd', 'no_distill', 'no_consistency', 'no_pairing']
DEFAULTS = {'GAHT': (13,128,.001), 'BiDA': (13,128,.01),
 'AgentBiDA': (13,128,.01), 'SelfAttentionAgentBiDA': (13,128,.01),
 'TSTnet': (12,100,.01), 'CLDA': (5,36,.01), 'SCLUDA': (7,32,.01),
 'MLUDA': (7,32,.01), 'SSWADA': (7,128,.01), 'CACL': (5,36,.0005)}


def metrics(pred, true):
    cm = np.bincount(7*true + pred, minlength=49).reshape(7,7)
    support = cm.sum(1)
    if (support == 0).any():
        raise ValueError('All seven classes must appear in target evaluation')
    recall = cm.diagonal()/support
    oa = cm.trace()/cm.sum()
    pe = (cm.sum(0)*support).sum()/float(cm.sum()**2)
    return {'OA': float(100*oa), 'AA': float(100*recall.mean()),
            'Kappa': float(100*(oa-pe)/(1-pe)),
            'classes': {str(i+1): float(100*v) for i,v in enumerate(recall)},
            'support': support.tolist(), 'confusion_matrix': cm.tolist()}


def loader(dataset, batch_size, workers, seed, train=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
        persistent_workers=workers > 0)


def repeat(dl):
    while True:
        yield from dl


@torch.no_grad()
def evaluate(method, dl, device):
    method.eval()
    predictions, labels = [], []
    for x,y in dl:
        predictions.append(method.predict(x.to(device)).argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(predictions), np.concatenate(labels)


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    files = sorted(p for folder in ('experiments','models','loss','external_methods')
                   for p in (root/folder).rglob('*.py'))
    return hashlib.sha256(''.join(str(p.relative_to(root))+digest(p) for p in files).encode()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--source', choices=['Houston13','Houston18'], required=True)
    p.add_argument('--target', choices=['Houston13','Houston18'], required=True)
    p.add_argument('--data-dir', default='Houston')
    p.add_argument('--output', default='fair_results')
    p.add_argument('--seed', type=int, default=2100)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--patch-size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--source-ratio', type=float, default=.95)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--num-heads', type=int, default=8)
    p.add_argument('--num-agents', type=int, default=4)
    p.add_argument('--num-tokens', type=int, default=4)
    p.add_argument('--lambda1', type=float, default=.1)
    p.add_argument('--lambda2', type=float, default=1.)
    p.add_argument('--ema-decay', type=float, default=.999)
    p.add_argument('--noise-std', type=float, default=.05)
    p.add_argument('--entropy-threshold', type=float, default=.5)
    p.add_argument('--mmd-start', type=int, default=1)
    p.add_argument('--ablation', choices=ABLATIONS, default='full')
    p.add_argument('--smoke', action='store_true', help='Two optimization steps; never emit publishable metrics')
    p.add_argument('--skip-complete', action='store_true')
    a = p.parse_args(argv)
    for k,v in zip(('patch_size','batch_size','lr'), DEFAULTS[a.model]):
        if getattr(a,k) is None: setattr(a,k,v)
    if a.source == a.target or a.epochs < 1 or a.batch_size < 2:
        p.error('Require different domains, positive epochs, and batch_size >= 2')
    if not 0 < a.source_ratio < 1 or a.num_agents < 1 or a.num_heads < 1 or a.dim < 1 or a.depth < 1 or a.dim % a.num_heads:
        p.error('Invalid source ratio, agent count, or head dimension')
    if not a.smoke and a.mmd_start > a.epochs:
        p.error('MMD start exceeds training budget')
    if a.model not in MODELS[1:4] and a.ablation != 'full':
        p.error('Loss ablations are defined for the BiDA family only')
    if a.model == 'TSTnet' and math.isqrt(a.batch_size)**2 != a.batch_size:
        p.error('TSTnet graph construction requires a square batch size')
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Fix the server environment or explicitly use --device cpu.')
    if a.device == 'cpu': torch.set_num_threads(min(4, os.cpu_count() or 1))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(a.device)
    source, sy, sh = load_scene(a.data_dir, a.source)
    target, ty, th = load_scene(a.data_dir, a.target)
    tr, va = source_split(sy, a.seed, a.source_ratio)
    tc = np.argwhere(ty >= 0)
    packages = {}
    for name in ('torch','numpy','scipy','scikit-learn','h5py','cleanlab','opencv-contrib-python-headless','scikit-image'):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    config = vars(a).copy()
    for key in ('output','skip_complete','workers','device','data_dir'): config.pop(key)
    config.update(protocol=PROTOCOL, package_versions=packages, data_sha256=sh|th, code_sha256=code_fingerprint(),
                  source_split_sha256=hashlib.sha256(tr.tobytes()+va.tobytes()).hexdigest())
    run_id = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(a.output)/f'{a.source}_to_{a.target}'/a.model/a.ablation/f'seed_{a.seed}_{run_id}'
    if (run_dir/'result.json').exists():
        if a.skip_complete:
            print(f'Already complete: {run_dir}', flush=True); return
        raise FileExistsError(f'Completed run already exists: {run_dir}. Use --skip-complete.')
    run_dir.mkdir(parents=True,exist_ok=True)
    lock = (run_dir/'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'Another process is running {run_dir}')
    # No target class IDs enter method construction or optimization.
    if a.model in MODELS[:4]:
        from .core import CoreMethod
        method = CoreMethod(a)
    else:
        from .external import make_method, preprocess
        source,target = preprocess(a,source,target)
        method = make_method(a)
    method.to(device)
    train = Patches(source,tr,a.patch_size,sy)
    unlabeled = Patches(target,tc,a.patch_size)
    train_dl = loader(train,a.batch_size,a.workers,a.seed,True)
    target_dl = loader(unlabeled,a.batch_size,a.workers,a.seed+1,True)
    if not len(train_dl) or not len(target_dl): raise ValueError('Batch size exceeds available samples')
    eval_dl = loader(Patches(target,tc,a.patch_size,ty),a.batch_size,a.workers,a.seed)
    manifest = {'config':config,'run_id':run_id,'selection':'fixed_final_epoch',
        'source_train_count':len(tr),'source_validation_count':len(va),
        'target_count':len(tc),'target_labels_used_for_training':False,
        'target_population':'original labeled foreground mask; class IDs withheld',
        'python':platform.python_version(),'packages':packages,'device':str(device),
        'recipe':getattr(method,'recipe','corrected BiDA family / source-only GAHT'),
        'parameter_count':sum(p.numel() for p in method.parameters() if p.requires_grad),
        'training_steps_per_epoch': 2 if a.smoke else max(len(train_dl),len(target_dl))}
    (run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    np.savez_compressed(run_dir/'coordinates.npz',source_train=tr,source_validation=va,target=tc)
    start=time.time()
    epochs = 1 if a.smoke else a.epochs
    # Common image-exposure budget; method-specific batch sizes are disclosed.
    steps = 2 if a.smoke else max(len(train_dl),len(target_dl))
    with (run_dir/'training.jsonl').open('w') as log:
        for epoch in range(1,epochs+1):
            method.train()
            if hasattr(method,'prepare_epoch'):
                method.prepare_epoch(epoch,train,unlabeled,device,a.workers,smoke=a.smoke)
            src,tgt=repeat(train_dl),repeat(target_dl)
            totals={}
            for step in range(steps):
                xs,ys=next(src); xt,ids=next(tgt)
                values=method.step(xs.to(device),ys.to(device),xt.to(device),ids.to(device),epoch)
                for k,v in values.items():
                    if not np.isfinite(v): raise FloatingPointError(f'{a.model}: nonfinite {k}')
                    totals[k]=totals.get(k,0.)+float(v)
                if step % 100 == 0: print(f'{a.model} seed={a.seed} epoch={epoch}/{epochs} step={step}/{steps} loss={values["loss"]:.5f}',flush=True)
            row={'epoch':epoch,'steps':steps,**{k:v/steps for k,v in totals.items()}}
            log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps(row),flush=True)
    if a.smoke:
        method.eval()
        with torch.no_grad():
            out=method.predict(next(iter(eval_dl))[0][:2].to(device))
        if out.shape!=(2,7) or not torch.isfinite(out).all(): raise AssertionError('Invalid inference output')
        (run_dir/'smoke_passed.json').write_text(json.dumps({'passed':True,'model':a.model}))
        print(f'SMOKE PASS (no accuracy report): {run_dir}'); return
    # Target evaluation happens once, after all optimization and checkpoint selection.
    torch.save({'state_dict':method.state_dict(),'manifest':manifest},run_dir/'final.pt')
    pred,true=evaluate(method,eval_dl,device)
    scores=metrics(pred,true)
    np.savez_compressed(run_dir/'predictions.npz',prediction=pred,label=true,coordinates=tc)
    result=manifest|{'status':'complete','metrics':scores,'elapsed_seconds':time.time()-start}
    tmp=run_dir/'result.json.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(run_dir/'result.json')
    print(json.dumps(scores,indent=2));print(f'Saved {run_dir}',flush=True)


if __name__=='__main__': main()
