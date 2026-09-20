"""Atomic epoch-boundary recovery for trusted, locally generated checkpoints."""
import os
import random
from pathlib import Path
import numpy as np
import torch


class EpochCheckpoint:
    def __init__(self, objects, loaders=()):
        self.objects = objects
        self.loaders = loaders
        folder = os.environ.get('BIDA_RUN_DIR')
        self.path = Path(folder) / 'last_epoch.pt' if folder else None
        self.fingerprint = os.environ.get('BIDA_RUN_FINGERPRINT')

    def load(self):
        if self.path is None or not self.path.exists():
            return None, {}
        # Only load our own run artifacts, never downloaded checkpoints.
        state = torch.load(self.path, weights_only=False)
        if state['fingerprint'] != self.fingerprint:
            raise ValueError('Checkpoint does not match this run configuration/code')
        if set(state['objects']) != set(self.objects):
            raise ValueError('Checkpoint object mismatch')
        for key, obj in self.objects.items():
            obj.load_state_dict(state['objects'][key])
            if isinstance(obj, torch.nn.Module):
                for name, module in obj.named_modules():
                    module.training = state['modes'][key][name]
                    # GRL counters are Python attributes, absent from state_dict.
                    counter = state['counters'].get(key, {}).get(name)
                    if counter is not None:module.iter_num = counter
        for loader, generator in zip(self.loaders, state['generators']):
            if generator is not None:loader.generator.set_state(generator)
        random.setstate(state['random'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if torch.cuda.is_available() and state['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['cuda'])
        print(f'Resuming from completed epoch boundary: next loop index {state["next_epoch"]}', flush=True)
        for relative, content in state.get('artifacts', {}).items():
            target = self.path.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return state['next_epoch'], state['extra']

    def save(self, next_epoch, extra):
        if self.path is None:return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for pattern in ('CACL_best.pt', 'checkpoints/model_ts_best*.pth'):
            for file in self.path.parent.glob(pattern):
                artifacts[str(file.relative_to(self.path.parent))] = file.read_bytes()
        state = dict(artifacts=artifacts, fingerprint=self.fingerprint, next_epoch=next_epoch, extra=extra,
                     objects={k:v.state_dict() for k,v in self.objects.items()},
                     modes={k:{n:m.training for n,m in v.named_modules()}
                            for k,v in self.objects.items() if isinstance(v,torch.nn.Module)},
                     counters={k:{n:m.iter_num for n,m in v.named_modules() if hasattr(m,'iter_num')}
                               for k,v in self.objects.items() if isinstance(v,torch.nn.Module)},
                     generators=[l.generator.get_state() if l.generator is not None else None for l in self.loaders],
                     random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        temp=self.path.with_suffix('.tmp')
        with temp.open('wb') as f:
            torch.save(state,f);f.flush();os.fsync(f.fileno())
        temp.replace(self.path)


def collect(namespace, names):
    return {name:namespace[name] for name in names.split() if name in namespace}
