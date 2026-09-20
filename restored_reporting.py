"""Metrics and tables for restored recipes; deliberately no claim of a common protocol."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

CLASSES = ['Grass healthy', 'Grass stressed', 'Trees', 'Water',
           'Residential buildings', 'Non-residential buildings', 'Road']


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def from_confusion(cm, selection):
    import numpy as np
    c = np.asarray(cm, dtype=float)
    if c.shape != (7, 7) or not np.isfinite(c).all() or (c < 0).any() or (c != np.floor(c)).any():
        raise ValueError('Expected a finite nonnegative 7x7 confusion matrix')
    support = c.sum(1)
    if (support == 0).any():
        raise ValueError('Cannot report seven-class AA: a class is absent')
    n = c.sum()
    recall = c.diagonal() / support
    oa = c.trace() / n
    pe = (support * c.sum(0)).sum() / n**2
    return dict(OA=float(100*oa), AA=float(100*recall.mean()),
                Kappa=float(100*(oa-pe)/(1-pe)),
                classes={str(i+1): float(100*v) for i,v in enumerate(recall)},
                support=support.astype(int).tolist(), confusion_matrix=c.astype(int).tolist(),
                selection=selection)


def from_predictions(labels, predictions, selection):
    import numpy as np
    y = np.asarray(labels).reshape(-1)
    pred = np.asarray(predictions).reshape(-1)
    if y.shape != pred.shape or not len(y):
        raise ValueError('Prediction/label length mismatch')
    if any(not np.isfinite(v).all() or (v != v.astype(int)).any() or
           (v < 0).any() or (v > 6).any() for v in (y, pred)):
        raise ValueError('Expected zero-based class IDs 0..6')
    cm = np.bincount(7*y.astype(int)+pred.astype(int), minlength=49).reshape(7,7)
    return from_confusion(cm, selection)


def validate(scores):
    values = [scores[k] for k in ('OA','AA','Kappa')]
    values += [scores['classes'][str(i)] for i in range(1,8)]
    if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in values):
        raise ValueError('Nonfinite or missing scores')
    if not (0 <= values[0] <= 100 and 0 <= values[1] <= 100 and -100 <= values[2] <= 100):
        raise ValueError('Metric scale must be percentage; Kappa x100')
    if not all(0 <= v <= 100 for v in values[3:]):
        raise ValueError('Invalid class recalls')
    if abs(scores['AA'] - statistics.mean(values[3:])) > 1e-5:
        raise ValueError('AA does not match the seven class recalls')
    if 'confusion_matrix' in scores:
        check = from_confusion(scores['confusion_matrix'], scores.get('selection','unknown'))
        for key in ('OA','AA','Kappa'):
            if abs(check[key]-scores[key]) > 1e-5:
                raise ValueError('Metric does not match confusion matrix: '+key)
        for key in check['classes']:
            if abs(check['classes'][key]-scores['classes'][key]) > 1e-5:
                raise ValueError('Class metric mismatch')
        if scores.get('support') != check['support']:
            raise ValueError('Support mismatch')
    return scores


def summarize(values):
    mean = statistics.mean(values)
    return f'{mean:.2f} ± {statistics.stdev(values):.2f}' if len(values)>1 else f'{mean:.2f} (n=1)'


def tables(root):
    root=Path(root)
    cfg=json.loads((root/'configuration.json').read_text())
    out=root/'tables';out.mkdir(exist_ok=True)
    for direction in ('13to18','18to13'):
        if cfg['direction'] not in ('both',direction):continue
        models=cfg['models']; seeds=cfg['seeds']; runs={}; statuses=[]; protocols=[]
        for model in models:
            valid=[]
            for seed in seeds:
                base=root/direction/model/str(seed)
                try:
                    status=json.loads((base/'status.json').read_text())
                    if status.get('exit_code') != 0 or not status.get('scores_exported'):continue
                    scores=validate(json.loads((base/'result.json').read_text()))
                    if status.get('legacy_import'):scores['selection']=scores.get('selection','unknown')+' (legacy import; provenance unverified)'
                    if status.get('run_fingerprint') != cfg.get('run_fingerprint'):
                        raise ValueError('Run provenance mismatch')
                    valid.append(scores)
                except (OSError,ValueError,KeyError,TypeError):continue
            runs[model]=valid
            statuses.append(f'{len(valid)}/{len(seeds)}')
            supports={tuple(s.get('support',[])) for s in valid}
            selections=sorted({s.get('selection','unknown') + (f"; {s['completed_epochs']}/{s['requested_epochs']} epochs" if 'completed_epochs' in s else '') for s in valid})
            protocols.append([model, ', '.join(str(sum(s)) for s in sorted(supports)) or 'pending',
                              ', '.join(selections) or 'pending'])
        rows=[['Completed seeds']+statuses]
        for i,label in enumerate(CLASSES+['OA (%)','AA (%)','Kappa × 100']):
            row=[label]
            for model in models:
                ss=runs[model]
                if len(ss)!=len(seeds):row.append('pending');continue
                vals=[s['classes'][str(i+1)] if i<7 else s[('OA','AA','Kappa')[i-7]] for s in ss]
                row.append(summarize(vals))
            rows.append(row)
        header=['Class / metric']+models
        name=out/(direction+'_comparison')
        with name.with_suffix('.csv').open('w',newline='') as f:
            csv.writer(f).writerows([header]+rows)
        note=('Restored-recipe comparison; mean ± sample SD over exactly '+str(len(seeds))+
              ' seeds. Pending cells are not averages over a subset. Class scores are recall (%). '
              'Training/evaluation populations and checkpoint rules differ across methods; '
              'these tables do not establish a controlled fair comparison or reproduce published scores.')
        md='# '+direction+'\n\n'+note+'\n\n'
        md+='| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'
        md+='\n'.join('| '+' | '.join(row)+' |' for row in rows)+'\n\n'
        md+='| Model | Evaluated pixels | Checkpoint rule |\n| --- | --- | --- |\n'
        md+='\n'.join('| '+' | '.join(row)+' |' for row in protocols)+'\n'
        name.with_suffix('.md').write_text(md)
        def tex(s):
            return str(s).replace('_',r'\_').replace('%',r'\%').replace('±',r'$\pm$').replace('×',r'$\times$')
        lines=[r'\begin{table*}',r'\centering',r'\caption{'+tex(note)+'}',
               r'\resizebox{\textwidth}{!}{',r'\begin{tabular}{l'+'c'*len(models)+'}',r'\toprule']
        lines+=[' & '.join(map(tex,row))+r' \\' for row in [header]+rows]
        lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table*}']
        name.with_suffix('.tex').write_text('\n'.join(lines)+'\n')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    print(tables(p.parse_args().root))
