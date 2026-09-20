import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import restored_reporting as report
from restored_checkpoint import EpochCheckpoint
import run_restored_all as launch


class RestoredTests(unittest.TestCase):
    def scores(self):
        c=np.eye(7,dtype=int)*10;c[0,0]=5;c[0,1]=5
        return report.from_confusion(c,'final_epoch')

    def test_metrics_scales_and_validation(self):
        s=self.scores()
        self.assertAlmostEqual(s['OA'],100*65/70)
        self.assertEqual(s['support'],[10]*7)
        report.validate(s)
        s['OA']*=100
        with self.assertRaises(ValueError):report.validate(s)
        with self.assertRaises(ValueError):report.from_predictions([7],[0],'final')
        with self.assertRaises(ValueError):report.from_confusion(np.zeros((7,7)),'final')

    def test_partial_and_complete_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cfg=dict(direction='13to18',models=['BiDA'],seeds=[2100,2101,2102],run_fingerprint='x')
            report.atomic_json(root/'configuration.json',cfg)
            for seed in cfg['seeds']:
                dest=root/'13to18'/'BiDA'/str(seed)
                report.atomic_json(dest/'result.json',self.scores())
                report.atomic_json(dest/'status.json',dict(exit_code=0,scores_exported=True,run_fingerprint='x'))
                report.tables(root)
                table=(root/'tables/13to18_comparison.md').read_text()
                self.assertIn(f'{seed-2099}/3',table)
                if seed<2102:self.assertIn('pending',table)
                else:self.assertIn('92.86 ± 0.00',table)
            self.assertTrue((root/'tables/13to18_comparison.tex').exists())
            report.atomic_json(root/'13to18/BiDA/2102/status.json',dict(exit_code=1,scores_exported=True,run_fingerprint='x'))
            report.tables(root)
            self.assertIn('pending',(root/'tables/13to18_comparison.md').read_text())

    def test_epoch_state_roundtrip_and_rng(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',BIDA_RUN_DIR=folder,BIDA_RUN_FINGERPRINT='x'):
            model=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Dropout(.1))
            optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
            loader=DataLoader(TensorDataset(torch.randn(8,3)),generator=torch.Generator().manual_seed(3))
            loss=model(torch.ones(2,3)).sum();loss.backward();optimizer.step()
            model.eval()
            recovery=EpochCheckpoint(dict(model=model,optimizer=optimizer),[loader])
            expected={k:v.clone() for k,v in model.state_dict().items()}
            recovery.save(3,dict(clean_data=np.arange(5),step=20))
            wanted=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=loader.generator))
            with torch.no_grad():
                for p in model.parameters():p.zero_()
            model.train();optimizer.param_groups[0]['lr']=9
            epoch,extra=recovery.load()
            self.assertEqual(epoch,3);self.assertEqual(extra['step'],20)
            self.assertFalse(model.training)
            self.assertEqual(optimizer.param_groups[0]['lr'],.1)
            self.assertTrue(optimizer.state)
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,expected[k]))
            self.assertEqual(random.random(),wanted[0]);self.assertEqual(np.random.rand(),wanted[1])
            self.assertTrue(torch.equal(torch.rand(3),wanted[2]))
            self.assertTrue(torch.equal(torch.rand(3,generator=loader.generator),wanted[3]))
            with patch.dict('os.environ',BIDA_RUN_FINGERPRINT='wrong'):
                with self.assertRaises(ValueError):EpochCheckpoint(dict(model=model,optimizer=optimizer),[loader]).load()

    def test_launcher_skip_and_configuration_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'runs'
            fake=root/'fake.py'
            fake.write_text('import json, pathlib\np=pathlib.Path("ablation_results/BiDA_results_seed_2100.json")\np.parent.mkdir(exist_ok=True)\np.write_text('+repr(json.dumps(self.scores()))+')\nwith open("calls.txt","a") as f:f.write("run\\n")\n')
            args=['launcher','--direction','13to18','--models','BiDA','--seeds','2100','--output',str(output)]
            with patch.object(launch,'ROOT',root),patch.object(launch,'fingerprint',return_value='fixture'),patch.object(launch,'command',return_value=(root,[sys.executable,str(fake)])):
                with patch.object(sys,'argv',args):launch.main()
                with patch.object(sys,'argv',args+['--resume']):launch.main()
                self.assertEqual((root/'calls.txt').read_text(),'run\n')
                with patch.object(sys,'argv',args+['--resume','--num-agents','2']):
                    with self.assertRaises(SystemExit):launch.main()
                self.assertEqual(json.loads((output/'failures.json').read_text()),[])

    def test_all_commands_have_both_directions_and_requested_budget(self):
        count=0
        for source,target in [('Houston13','Houston18'),('Houston18','Houston13')]:
            for model in launch.MODELS:
                for seed in (2100,2101,2102):
                    cwd,cmd=launch.command(model,source,target,seed,120,4)
                    self.assertTrue((cwd/cmd[2]).is_file())
                    self.assertEqual(cmd[cmd.index('--seed')+1],str(seed))
                    self.assertIn('120',cmd)
                    if model in launch.MODELS[2:4]:self.assertEqual(cmd[cmd.index('--num_agents')+1],'4')
                    count+=1
        self.assertEqual(count,60)



class CoreResumeIntegration(unittest.TestCase):
    def test_standard_training_matches_after_epoch_interruption(self):
        import ast
        import os
        import restored_checkpoint
        source=Path('train_pipeline.py').read_text()
        tree=ast.parse(source)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_standard')
        code=compile(ast.Module(body=[node],type_ignores=[]),'train_pipeline.py','exec')
        class StopAfterEpoch(Exception):pass
        class InterruptingCheckpoint(EpochCheckpoint):
            def save(self,epoch,extra):
                super().save(epoch,extra)
                if epoch==2:raise StopAfterEpoch()
        def execute(folder,interrupt=False):
            from types import SimpleNamespace
            random.seed(19);np.random.seed(19);torch.manual_seed(19)
            model=torch.nn.Linear(3,7)
            opt=torch.optim.SGD(model.parameters(),lr=.03,momentum=.9)
            data=TensorDataset(torch.randn(12,3),torch.arange(12)%7)
            loader=DataLoader(data,batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(19))
            args=SimpleNamespace(epoch=3,log_interval=1,model='GAHT',seed=2100,lr=.03)
            cm=np.eye(7,dtype=int)
            def evaluate(*args,**kwargs):
                return 1.,{'Confusion_matrix':cm,'Accuracy':100.,'TPR':np.ones(7),'Kappa':1.}
            def save_best(network,best,path,**kwargs):
                Path(path).mkdir(parents=True,exist_ok=True)
                torch.save(network.state_dict(),Path(path)/'model_ts_best1.0000_2100.pth')
            class Progress:
                def __call__(self,seq,**kwargs):return seq
                @staticmethod
                def write(message):pass
            scope=dict(os=os,np=np,torch=torch,tqdm=Progress(),
                       EpochCheckpoint=InterruptingCheckpoint if interrupt else EpochCheckpoint,
                       atomic_json=report.atomic_json,from_confusion=report.from_confusion,
                       validation_standard=evaluate,save_ts_checkpoint=save_best,
                       io=SimpleNamespace(savemat=lambda *a,**k:None))
            exec(code,scope)
            with patch.dict(os.environ,BIDA_RUN_DIR=str(folder),BIDA_RUN_FINGERPRINT='test'):
                scope['train_standard'](model,opt,torch.nn.CrossEntropyLoss(),7,loader,loader,args,str(folder),'cpu',None)
            return torch.load(Path(folder)/'last_epoch.pt',weights_only=False)['objects']['student']
        original=Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            os.chdir(temp)
            try:
                expected=execute(Path(temp)/'continuous')
                with self.assertRaises(StopAfterEpoch):execute(Path(temp)/'resumed',True)
                actual=execute(Path(temp)/'resumed')
                for key in expected:self.assertTrue(torch.equal(expected[key],actual[key]),key)
            finally:os.chdir(original)

if __name__=='__main__':unittest.main()
