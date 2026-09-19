"""One combined component-ablation table per transfer direction."""
import argparse
import csv
from pathlib import Path
from .table import read_results,summarize
from .run import ABLATIONS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='fair_results');p.add_argument('--source',required=True)
    p.add_argument('--target',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[2100,2101,2102])
    p.add_argument('--num-agents',type=int,default=4)
    a=p.parse_args();models=['BiDA','AgentBiDA','SelfAttentionAgentBiDA']
    rows=[];fingerprints=set()
    for ab in ABLATIONS:
        runs=read_results(a.root,a.source,a.target,models,a.seeds,ab,a.num_agents)
        for m in models:
            c=runs[m,a.seeds[0]]['config'];fingerprints.add((c['code_sha256'],c['epochs'],c['source_ratio']))
            rows.append([m,ab]+[summarize([runs[m,s]['metrics'][key] for s in a.seeds]) for key in ('OA','AA','Kappa')])
    if len(fingerprints)!=1:raise ValueError('Ablations use different code revisions or budgets')
    columns=['Model','Ablation','OA (%)','AA (%)','Kappa x 100']
    out=Path(a.root)/'tables';out.mkdir(exist_ok=True,parents=True)
    base=out/f'{a.source}_to_{a.target}_component_ablation'
    with base.with_suffix('.csv').open('w',newline='') as f:csv.writer(f).writerows([columns]+rows)
    md='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    md+='\n'.join('| '+' | '.join(row)+' |' for row in rows)+'\n'
    base.with_suffix('.md').write_text(md)
    def tex(x):return x.replace('_',r'\_').replace('%',r'\%').replace('±',r'$\pm$')
    lines=[r'\begin{tabular}{llccc}',r'\toprule']+[' & '.join(map(tex,row))+r' \\' for row in [columns]+rows]+[r'\bottomrule',r'\end{tabular}']
    base.with_suffix('.tex').write_text('\n'.join(lines)+'\n')
    print(base.with_suffix('.md'))


if __name__=='__main__':main()
