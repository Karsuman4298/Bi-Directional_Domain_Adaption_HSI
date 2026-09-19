"""Strict complete-run aggregation: no best-seed selection, no legacy JSON inputs."""
import argparse
import hashlib
import csv
import json
from pathlib import Path
import numpy as np
from scipy import stats
from .data import CLASS_NAMES, EXPECTED
from .run import MODELS, PROTOCOL, metrics


def read_results(root,source,target,models,seeds,ablation='full',agents=4):
    selected={}
    for path in (Path(root)/f'{source}_to_{target}').glob('*/*/*/result.json'):
        r=json.loads(path.read_text());c=r['config']
        if c['model'] not in models or c['seed'] not in seeds or c['ablation']!=ablation:continue
        if c['model'] in ('AgentBiDA','SelfAttentionAgentBiDA') and c['num_agents']!=agents:continue
        if (r.get('status')!='complete' or c.get('protocol')!=PROTOCOL or c.get('smoke')
            or r.get('selection')!='fixed_final_epoch' or r.get('target_labels_used_for_training') is not False
            or c['source']!=source or c['target']!=target):
            raise ValueError(f'Invalid protocol: {path}')
        pred=np.load(path.parent/'predictions.npz',allow_pickle=False)
        if not np.array_equal(pred['coordinates'],np.unique(pred['coordinates'],axis=0)):
            raise ValueError(f'Duplicate or unordered evaluation coordinates: {path}')
        actual=metrics(pred['prediction'],pred['label'])
        if actual['support']!=EXPECTED[target]:raise ValueError(f'Wrong target population: {path}')
        for key in ('OA','AA','Kappa'):
            if not np.isfinite(r['metrics'][key]) or not np.isclose(actual[key],r['metrics'][key]):
                raise ValueError(f'Corrupt metric {key}: {path}')
        if actual['confusion_matrix']!=r['metrics']['confusion_matrix']:
            raise ValueError(f'Confusion matrix mismatch: {path}')
        for i in range(1,8):
            if not np.isclose(actual['classes'][str(i)],r['metrics']['classes'][str(i)]):raise ValueError(f'Class metric mismatch: {path}')
        key=(c['model'],c['seed'])
        if key in selected:raise ValueError(f'Ambiguous repeated configuration for {key}; use a separate --output directory')
        selected[key]=r
    missing=[f'{m}/seed={s}' for m in models for s in seeds if (m,s) not in selected]
    if missing:raise ValueError('Missing complete runs (no partial averaging): '+', '.join(missing))
    # Architecture-specific settings differ; these protocol fields must not.
    signatures=set()
    for r in selected.values():
        c=r['config']
        signatures.add(json.dumps({k:c.get(k) for k in ('epochs','source_ratio','code_sha256','data_sha256','package_versions')},sort_keys=True))
    if len(signatures)!=1:raise ValueError('Mixed epoch budgets, splits, code revisions, or datasets')
    for seed in seeds:
        if len({selected[m,seed]['config']['source_split_sha256'] for m in models})!=1:
            raise ValueError(f'Source split mismatch at seed {seed}')
    family=[m for m in models if m in ('BiDA','AgentBiDA','SelfAttentionAgentBiDA')]
    settings=('patch_size','batch_size','lr','depth','dim','num_heads','num_tokens',
              'lambda1','lambda2','ema_decay','noise_std','entropy_threshold','mmd_start')
    for seed in seeds:
        signatures={json.dumps({k:selected[m,seed]['config'].get(k) for k in settings},sort_keys=True) for m in family}
        if len(signatures)>1:raise ValueError('BiDA-family architecture comparison has mismatched training settings')
    if len(seeds)<2 or len(set(seeds))!=len(seeds):raise ValueError('Use at least two distinct seeds for mean ± sample SD')
    return selected


def summarize(values):return f'{np.mean(values):.2f} ± {np.std(values,ddof=1):.2f}'


def write_table(path,models,seeds,runs,title):
    rows=[]
    for i,name in enumerate(CLASS_NAMES,1):
        rows.append([name]+[summarize([runs[m,s]['metrics']['classes'][str(i)] for s in seeds]) for m in models])
    for metric in ('OA','AA','Kappa'):
        rows.append([metric if metric!='Kappa' else 'Kappa × 100']+[summarize([runs[m,s]['metrics'][metric] for s in seeds]) for m in models])
    columns=['Class / metric']+models
    with path.with_suffix('.csv').open('w',newline='') as f:csv.writer(f).writerows([columns]+rows)
    md=f'# {title}\n\nMean ± sample SD over exactly {len(seeds)} paired seeds: {seeds}.\n\n'
    md+='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    md+='\n'.join('| '+' | '.join(row)+' |' for row in rows)
    md+='\n\nCorrected repository recipes; fixed final checkpoint; identical source split and original target pixel population. No target-label model selection. These are reruns, not copied paper scores.\n'
    md+='\n| Method | Patch | Batch | LR | Epochs |\n| --- | --- | --- | --- | --- |\n'
    for m in models:
        c=runs[m,seeds[0]]['config'];md+=f'| {m} | {c["patch_size"]} | {c["batch_size"]} | {c["lr"]} | {c["epochs"]} |\n'
    path.with_suffix('.md').write_text(md)
    def tex(s):return str(s).replace('_',r'\_').replace('±',r'$\pm$').replace('×',r'$\times$')
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{'+tex(title)+f'; mean and sample standard deviation over {len(seeds)} seeds.'+'}',r'\resizebox{\textwidth}{!}{',r'\begin{tabular}{l'+'c'*len(models)+'}',r'\toprule']
    lines+=[' & '.join(map(tex,row))+r' \\' for row in [columns]+rows]
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table*}']
    path.with_suffix('.tex').write_text('\n'.join(lines)+'\n')


def comparisons(runs,seeds,models):
    rows=[]
    if 'BiDA' not in models:return rows
    for model in ('AgentBiDA','SelfAttentionAgentBiDA'):
        if model not in models:continue
        for key in ('OA','AA','Kappa'):
            d=np.array([runs[model,s]['metrics'][key]-runs['BiDA',s]['metrics'][key] for s in seeds])
            sem=stats.sem(d);margin=stats.t.ppf(.975,len(d)-1)*sem
            rows.append({'model':model,'metric':key,'paired_deltas':d.tolist(),'mean_delta':float(d.mean()),
                'ci95_low':float(d.mean()-margin),'ci95_high':float(d.mean()+margin),
                'note':'Descriptive paired t interval over training seeds; not adjusted for multiple comparisons or spatial dependence.'})
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='fair_results');p.add_argument('--out',default='fair_tables')
    p.add_argument('--source',choices=['Houston13','Houston18'],required=True)
    p.add_argument('--target',choices=['Houston13','Houston18'],required=True)
    p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    p.add_argument('--seeds',nargs='+',type=int,default=[2100,2101,2102,2103,2104])
    p.add_argument('--ablation',default='full');p.add_argument('--num-agents',type=int,default=4)
    a=p.parse_args(argv)
    r=read_results(a.root,a.source,a.target,a.models,a.seeds,a.ablation,a.num_agents)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    suffix='' if a.models==MODELS else '_subset_'+hashlib.sha256(','.join(a.models).encode()).hexdigest()[:8]
    path=out/f'{a.source}_to_{a.target}_{a.ablation}_agents{a.num_agents}{suffix}'
    write_table(path,a.models,a.seeds,r,f'{a.source} to {a.target}, {a.ablation}')
    path.with_suffix('.paired.json').write_text(json.dumps(comparisons(r,a.seeds,a.models),indent=2))
    print(path.with_suffix('.md'))


if __name__=='__main__': main()
