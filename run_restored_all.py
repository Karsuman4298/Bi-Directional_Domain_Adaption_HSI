"""Launch restored recipes; no model/loss changes or claims of equal protocols.

Uses the active Python environment. Legacy dependencies must already be installed.
Failures are recorded and other runs continue. Run only one launcher per checkout.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction', choices=['both', '13to18', '18to13'], default='both')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--seeds', nargs='+', type=int, default=[2100, 2101, 2102])
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--num-agents', type=int, default=2)
    p.add_argument('--output', default='restored_runs/' + time.strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.epochs < 1:
        p.error('epochs must be positive')
    out = Path(a.output).resolve()
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'configuration.json').write_text(json.dumps(vars(a), indent=2))
    failures = []
    for source, target, direction in [('Houston13', 'Houston18', '13to18'),
                                      ('Houston18', 'Houston13', '18to13')]:
        if a.direction not in ('both', direction):
            continue
        for model in a.models:
            for seed in a.seeds:
                cwd, cmd = command(model, source, target, seed, a.epochs, a.num_agents)
                print(f'{direction} {model} seed={seed}: cwd={cwd} command={cmd}', flush=True)
                if a.dry_run:
                    continue
                dest = out / direction / model / str(seed)
                dest.mkdir(parents=True)
                result = ROOT / 'ablation_results' / f'{model}_results_seed_{seed}.json'
                if result.exists():
                    shutil.move(str(result), str(dest / 'previous_result.json'))
                env = os.environ.copy()
                env['PYTHONPATH'] = str(cwd)
                env.setdefault('CUDA_VISIBLE_DEVICES', '0')
                with (dest / 'training.log').open('w') as log:
                    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True)
                    try:
                        for line in proc.stdout:
                            print(line, end='', flush=True)
                            log.write(line)
                            log.flush()
                        code = proc.wait()
                    except KeyboardInterrupt:
                        proc.terminate()
                        proc.wait()
                        raise
                if result.exists():
                    shutil.copy2(result, dest / 'result.json')
                status = {'exit_code': code, 'scores_exported': result.exists(),
                          'command': cmd, 'cwd': str(cwd)}
                (dest / 'status.json').write_text(json.dumps(status, indent=2))
                if code or not result.exists():
                    failures.append(f'{direction}/{model}/{seed}')
    if not a.dry_run:
        (out / 'failures.json').write_text(json.dumps(failures, indent=2))
        print(f'Logs, scores and failure list: {out}')
        if failures:
            print(f'{len(failures)} runs failed or did not export scores; inspect training.log.')
            sys.exit(1)


if __name__ == '__main__':
    main()
