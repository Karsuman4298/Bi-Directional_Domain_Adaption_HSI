"""Launch restored recipes; no model/loss changes or claims of equal protocols.

Uses the active Python environment. Legacy dependencies must already be installed.
Failures are recorded and other runs continue. --resume skips completed runs and
recovers interrupted jobs from their last completed epoch. One launcher per checkout.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MODELS = ['GAHT', 'BiDA', 'AgentBiDA', 'SelfAttentionAgentBiDA',
          'TSTnet', 'CLDA', 'SCLUDA', 'MLUDA', 'SSWADA', 'CACL']
ROOT = Path(__file__).resolve().parent


def command(model, source, target, seed, epochs, agents):
    reverse = source == 'Houston18'
    domain = ['--source_name', source, '--target_name', target]
    common = ['--seed', str(seed)]
    if model in MODELS[:4]:
        script = {'AgentBiDA': 'train_agent_bida_fix.py',
                  'SelfAttentionAgentBiDA': 'train_self_attn_agent_bida_fix.py'}.get(model, 'main.py')
        args = ['--model', model] + domain + common + ['--epoch', str(epochs), '--num_workers', '2']
        if model in MODELS[2:4]:
            args += ['--num_agents', str(agents)]
        return ROOT, [sys.executable, '-u', script] + args
    cwd = ROOT / 'external_methods' / model
    scripts = {
        'TSTnet': 'train_tstnet.py',
        'CLDA': 'CLDA_HOUSTON18_2_13.py' if reverse else 'CLDA_HOUSTON13_2_18.py',
        'SCLUDA': 'SCLUDA_Houston_18_2_13.py' if reverse else 'SCLUDA_Houston.py',
        'MLUDA': 'MLUDA_hu_18_2_13.py' if reverse else 'MLUDA_hu.py',
        'SSWADA': 'main_18_2_13.py' if reverse else 'main.py',
        'CACL': 'demo_singleDA.py',
    }
    flag = '--num_epoch' if model in ('TSTnet', 'SSWADA') else '--epochs'
    args = common + [flag, str(epochs)]
    if model in ('TSTnet', 'CACL'):
        args += domain
    if model == 'CACL':
        args += ['--dataset', 'M_Houston', '--in_channel', '48']
    return cwd, [sys.executable, '-u', scripts[model]] + args


def fingerprint():
    import hashlib
    h = hashlib.sha256()
    paths = [ROOT / x for x in ('main.py', 'train_pipeline.py', 'run_restored_all.py',
             'restored_checkpoint.py', 'restored_reporting.py')]
    paths += list(ROOT.glob('train_*bida*.py'))
    for folder in ('models', 'utils', 'loss', 'external_methods'):
        paths += list((ROOT / folder).rglob('*.py'))
    for path in sorted(set(paths)):
        h.update(str(path.relative_to(ROOT)).encode()); h.update(path.read_bytes())
    for path in sorted((ROOT / 'Houston').glob('*.mat')):
        h.update(path.name.encode())
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def main():
    from restored_reporting import atomic_json, validate, tables
    import fcntl
    import importlib.metadata
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction', choices=['both', '13to18', '18to13'], default='both')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--seeds', nargs='+', type=int, default=[2100, 2101, 2102])
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--num-agents', type=int, default=4)
    p.add_argument('--output', default='restored_runs/agents4_120')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.epochs < 1 or a.num_agents < 1 or len(set(a.seeds)) != len(a.seeds):
        p.error('Require positive epochs/agents and distinct seeds')
    out = Path(a.output).resolve()
    config = {k:v for k,v in vars(a).items() if k not in ('resume','dry_run','output')}
    legacy = False
    if not a.dry_run:
        # Legacy exporters share repository-relative paths. Lock the whole checkout.
        lock = (ROOT / '.restored_launcher.lock').open('a')
        try:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:p.error('Another restored launcher is using this checkout')
        if out.exists() and not a.resume:p.error('Output exists; use --resume or a fresh --output')
        if a.resume and not (out/'configuration.json').exists():p.error('Resume configuration missing')
        config['run_fingerprint'] = fingerprint()
        config['python'] = sys.version
        config['packages'] = {}
        for package in ('torch','numpy','scikit-learn','cleanlab','scipy','h5py','hdf5storage',
                        'scikit-image','opencv-contrib-python-headless','einops','torch-geometric'):
            try:config['packages'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:config['packages'][package] = None
        if a.resume:
            old=json.loads((out/'configuration.json').read_text())
            for key in ('direction','epochs','seeds','models','num_agents'):
                if old.get(key) != config[key]:p.error('Resume configuration mismatch: '+key)
            legacy='run_fingerprint' not in old
            if not legacy:
                for key in ('run_fingerprint','python','packages'):
                    if old.get(key)!=config[key]:p.error('Code, data or environment changed: '+key+'; use a new output folder')
            else:
                print('Importing validated legacy completed scores; old interrupted runs have no epoch checkpoints and restart.')
                atomic_json(out/'legacy_configuration.json', old)
        out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'configuration.json',config)
    if not a.dry_run:
        tables(out)
        import signal
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
    failures=[]
    for source,target,direction in [('Houston13','Houston18','13to18'),('Houston18','Houston13','18to13')]:
        if a.direction not in ('both',direction):continue
        for model in a.models:
            for seed in a.seeds:
                cwd,cmd=command(model,source,target,seed,a.epochs,a.num_agents)
                print(f'{direction} {model} seed={seed}: {cmd}',flush=True)
                if a.dry_run:continue
                dest=out/direction/model/str(seed)
                dest.mkdir(parents=True,exist_ok=True)
                status_path=dest/'status.json'
                try:
                    status=json.loads(status_path.read_text())
                    validate(json.loads((dest/'result.json').read_text()))
                    if status.get('exit_code')==0 and status.get('scores_exported'):
                        if legacy:
                            status['legacy_import']=True
                            status['run_fingerprint']=config['run_fingerprint']
                            atomic_json(status_path,status)
                        if status.get('run_fingerprint')==config['run_fingerprint']:
                            print('SKIP completed:',dest,flush=True);tables(out);continue
                except (OSError,ValueError,KeyError,TypeError):pass
                result=ROOT/'ablation_results'/f'{model}_results_seed_{seed}.json'
                if result.exists():
                    shutil.move(str(result),str(dest/f'previous_result_{time.time_ns()}.json'))
                env=os.environ.copy()
                env['PYTHONPATH']=os.pathsep.join([str(cwd),str(ROOT)])
                env.setdefault('CUDA_VISIBLE_DEVICES','0')
                env['BIDA_RUN_DIR']=str(dest)
                env['BIDA_RUN_FINGERPRINT']=config['run_fingerprint']+f'/{direction}/{model}/{seed}/{a.epochs}/{a.num_agents}'
                status=dict(exit_code=None,scores_exported=False,command=cmd,cwd=str(cwd),
                            run_fingerprint=config['run_fingerprint'],state='running')
                atomic_json(status_path,status)
                code=-1;error=None
                with (dest/'training.log').open('a') as log:
                    log.write('\n=== Launch '+time.strftime('%Y-%m-%d %H:%M:%S')+' ===\n');log.flush()
                    proc=None
                    try:
                        proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT,text=True,start_new_session=True)
                        for line in proc.stdout:
                            print(line,end='',flush=True);log.write(line);log.flush()
                        code=proc.wait()
                    except KeyboardInterrupt:
                        import signal
                        if proc is not None:
                            os.killpg(proc.pid,signal.SIGTERM);proc.wait()
                        status.update(state='interrupted',exit_code=-15)
                        atomic_json(status_path,status);tables(out)
                        raise
                    except OSError as e:error=str(e)
                exported=False
                if result.exists():
                    shutil.copy2(result,dest/'result.json')
                    try:validate(json.loads(result.read_text()));exported=True
                    except (ValueError,KeyError,TypeError) as e:error='Invalid scores: '+str(e)
                status.update(exit_code=code,scores_exported=exported,error=error,
                              state='complete' if code==0 and exported else 'failed')
                atomic_json(status_path,status)
                if code or not exported:failures.append(f'{direction}/{model}/{seed}')
                atomic_json(out/'failures.json',failures)
                tables(out)
    if not a.dry_run:
        atomic_json(out/'failures.json',failures)
        print(f'Logs, checkpoints, scores and tables: {out}')
        if failures:
            print(f'{len(failures)} runs failed or did not export valid scores; inspect training.log.')
            sys.exit(1)


if __name__ == '__main__':
    main()
