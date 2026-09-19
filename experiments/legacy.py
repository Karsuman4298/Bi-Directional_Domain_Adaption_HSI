"""Translate historical core CLI flags to the corrected, isolated experiment runner."""
import sys
from .run import main


def legacy(default_model=None):
    aliases={'--source_name':'--source','--target_name':'--target','--dataset_dir':'--data-dir',
             '--epoch':'--epochs','--bs':'--batch-size','--ratio':'--source-ratio',
             '--num_workers':'--workers'}
    args=sys.argv[1:];converted=[];i=0
    while i<len(args):
        key=args[i]
        if key in ('--loss_type','--labelsmooth','--re_ratio','--log_interval'):
            value=args[i+1]
            valid={'--loss_type':'softmax','--labelsmooth':'off','--re_ratio':'1'}
            if key in valid and value!=valid[key]:raise SystemExit(f'{key}={value} is unsupported by the fair protocol')
            i+=2;continue
        if key=='--debug_shapes':converted.append('--smoke');i+=1;continue
        key=aliases.get(key,key.replace('_','-'))
        if key=='--device' and i+1<len(args) and args[i+1].isdigit():
            converted.extend([key,'cuda:'+args[i+1]]);i+=2;continue
        converted.append(key);i+=1
    if '--model' not in converted:converted.extend(['--model',default_model or 'BiDA'])
    if '--source' not in converted:converted.extend(['--source','Houston13'])
    if '--target' not in converted:converted.extend(['--target','Houston18'])
    print('Using corrected final-checkpoint protocol. Results go to fair_results; legacy checkpoints are not reused.',flush=True)
    main(converted)
