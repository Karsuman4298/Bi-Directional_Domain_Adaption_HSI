import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from experiments.data import Patches, source_split, EXPECTED
from experiments.core import CoreMethod, pair_indices, update_teacher
from experiments.run import parse_args, metrics, PROTOCOL
from experiments.table import read_results
from loss.mmd_loss import MMD_loss

torch.set_num_threads(2)


def args(model, **changes):
    a=parse_args(['--model',model,'--source','Houston13','--target','Houston18',
                  '--device','cpu','--smoke','--batch-size','4'])
    for k,v in changes.items():setattr(a,k,v)
    return a


@pytest.mark.parametrize('name',['agent_bida','self_attention_agent_bida'])
def test_agent_attention_matches_reference(name):
    mod=importlib.import_module('models.'+name)
    m=mod.AgentAttention(8,num_heads=2,num_agents=4).eval()
    x,y=torch.randn(2,5,8),torch.randn(2,5,8)
    out=m(x,y)[2]
    q=m.qkv(x).reshape(2,5,3,2,4).permute(2,0,3,1,4)[0]
    kv=m.qkv(y).reshape(2,5,3,2,4).permute(2,0,3,1,4)
    pooled=F.adaptive_avg_pool1d(q[:,:,1:].transpose(2,3).reshape(4,4,4),3)
    a=torch.cat((q[:,:,:1],pooled.reshape(2,2,4,3).transpose(2,3)),2)
    expected=(q@a.transpose(-2,-1)*m.scale).softmax(-1)@((a@kv[1].transpose(-2,-1)*m.scale).softmax(-1)@kv[2])
    expected=m.proj(expected.transpose(1,2).reshape(2,5,8))
    torch.testing.assert_close(out,expected)
    fixed=importlib.import_module('models.'+name+'_fix')
    assert fixed.AgentBiDAnet is mod.AgentBiDAnet


@pytest.mark.parametrize('model',['BiDA','AgentBiDA','SelfAttentionAgentBiDA'])
def test_core_gradients_ema_and_pairs(model):
    a=args(model,entropy_threshold=1.)
    method=CoreMethod(a)
    assert all(torch.equal(x,y) for x,y in zip(method.model.parameters(),method.teacher.parameters()))
    x,y=torch.rand(4,48,13,13),torch.rand(4,48,13,13)
    with torch.no_grad(): labels=method.teacher(None,y[:,None],inference_target_only=True)[1].argmax(1)
    values=method.step(x,labels,y,torch.arange(4),1)
    assert values['pairs']==4
    assert values['distill']>0 and values['mmd']>=-1e-5
    assert all(p.grad is None for p in method.teacher.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in method.model.parameters())
    method.eval()
    with torch.no_grad():assert method.predict(x[:1]).shape==(1,7)


@pytest.mark.parametrize('ablation',['source_only','no_mmd','no_distill','no_consistency','no_pairing'])
def test_ablation_switches(ablation):
    m=CoreMethod(args('AgentBiDA',ablation=ablation,entropy_threshold=1.))
    x=torch.rand(4,48,13,13)
    with torch.no_grad():y=m.teacher(None,x[:,None],inference_target_only=True)[1].argmax(1)
    v=m.step(x,y,x,torch.arange(4),1)
    if ablation=='source_only':assert set(v)=={'loss','cls'}
    if ablation=='no_mmd':assert v['mmd']==0
    if ablation=='no_distill':assert v['distill']==0
    if ablation=='no_consistency':assert v['consistency']==0
    if ablation=='no_pairing':assert v['pairs']==4


def test_pairing_uses_predictions_and_filters_entropy():
    y=torch.tensor([0,1,2,0]);logits=torch.zeros(4,7)
    logits[0,0]=20;logits[1,2]=20
    s,t=pair_indices(y,logits)
    assert s.tolist()==[0,2,3]
    assert torch.equal(y[s],logits.argmax(1)[t])
    assert len(pair_indices(y,torch.zeros(4,7))[0])==0


def test_no_reflected_labels_or_padding_coordinates():
    image=np.arange(4*5*2).reshape(4,5,2).astype('float32')
    label=np.full((4,5),-1);label[0,0]=0;label[3,4]=1
    coords=np.argwhere(label>=0)
    ds=Patches(image,coords,3,label);unlabeled=Patches(image,coords,3)
    assert len(ds)==2 and unlabeled.labels is None
    np.testing.assert_equal(ds[0][0][:,1,1],image[0,0])
    np.testing.assert_equal(ds[1][0][:,1,1],image[3,4])


def test_mmd_constant_and_unequal_batches():
    loss=MMD_loss()
    assert loss(torch.ones(3,7),torch.ones(4,7)).item()==0
    x=torch.randn(3,7,requires_grad=True);y=torch.randn(5,7,requires_grad=True)
    torch.testing.assert_close(loss(x,y),loss(y,x))
    loss(x,y).backward();assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('model',['TSTnet','CLDA','SCLUDA','MLUDA','SSWADA','CACL'])
def test_external_steps(model):
    from experiments.external import make_method
    a=args(model);m=make_method(a)
    x=torch.rand(4,48,a.patch_size,a.patch_size);y=torch.arange(4)
    values=m.step(x,y,torch.rand_like(x),torch.arange(4),1)
    assert np.isfinite(values['loss'])
    m.eval()
    with torch.no_grad(): assert m.predict(x[:1]).shape==(1,7)


def test_cacl_late_prototype_objective():
    from experiments.external import CACL
    a=args('CACL');m=CACL(a)
    image=np.random.default_rng(1).random((7,4,48)).astype('float32')
    labels=np.repeat(np.arange(7),4).reshape(7,4);coords=np.argwhere(labels>=0)
    src=Patches(image,coords,5,labels);tar=Patches(image,coords,5)
    m.prepare_epoch(21,src,tar,torch.device('cpu'),0)
    v=m.step(torch.rand(4,48,5,5),torch.arange(4),torch.rand(4,48,5,5),torch.arange(4),21)
    assert np.isfinite(v['target']) and v['target']>0


def make_record(root,seed,model='BiDA'):
    true=np.repeat(np.arange(7),EXPECTED['Houston18'])
    coords=np.column_stack((np.arange(len(true)),np.zeros(len(true),dtype=int)))
    path=root/'Houston13_to_Houston18'/model/'full'/f'seed_{seed}_test'
    path.mkdir(parents=True)
    np.savez(path/'predictions.npz',prediction=true,label=true,coordinates=coords)
    c=dict(model=model,seed=seed,ablation='full',source='Houston13',target='Houston18',
           protocol=PROTOCOL,smoke=False,epochs=200,patch_size=13,batch_size=128,lr=.01,source_ratio=.95,code_sha256='a',data_sha256={},source_split_sha256=str(seed))
    r=dict(config=c,status='complete',selection='fixed_final_epoch',target_labels_used_for_training=False,metrics=metrics(true,true))
    (path/'result.json').write_text(json.dumps(r));return path


def test_table_refuses_missing_seeds_and_mixed_revisions(tmp_path):
    p=make_record(tmp_path,1)
    with pytest.raises(ValueError,match='Missing'):read_results(tmp_path,'Houston13','Houston18',['BiDA'],[1,2])
    q=make_record(tmp_path,2)
    assert len(read_results(tmp_path,'Houston13','Houston18',['BiDA'],[1,2]))==2
    r=json.loads((q/'result.json').read_text());r['config']['code_sha256']='b'
    (q/'result.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Mixed'):read_results(tmp_path,'Houston13','Houston18',['BiDA'],[1,2])


def test_table_detects_tampered_oa(tmp_path):
    p=make_record(tmp_path,1);make_record(tmp_path,2)
    r=json.loads((p/'result.json').read_text());r['metrics']['OA']=10000
    (p/'result.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Corrupt'):read_results(tmp_path,'Houston13','Houston18',['BiDA'],[1,2])


def test_final_checkpoint_and_metrics_end_to_end(tmp_path,monkeypatch):
    import experiments.run as run
    image=np.random.default_rng(0).random((7,8,48),dtype=np.float32)
    labels=np.repeat(np.arange(7),8).reshape(7,8)
    monkeypatch.setattr(run,'load_scene',lambda root,name:(image.copy(),labels.copy(),{name:'fixture'}))
    run.main(['--model','BiDA','--source','Houston13','--target','Houston18',
              '--device','cpu','--workers','0','--batch-size','4','--epochs','1',
              '--source-ratio','.5','--output',str(tmp_path)])
    files=list(tmp_path.rglob('result.json'));assert len(files)==1
    r=json.loads(files[0].read_text())
    assert r['status']=='complete' and r['selection']=='fixed_final_epoch'
    assert r['metrics']['support']==[8]*7
    assert (files[0].parent/'final.pt').exists()
    assert (files[0].parent/'predictions.npz').exists()


def test_table_writes_csv_markdown_latex_and_paired_deltas(tmp_path):
    from experiments.table import main
    root=tmp_path/'runs'
    for seed in (1,2):
        make_record(root,seed)
        p=make_record(root,seed,'AgentBiDA')
        r=json.loads((p/'result.json').read_text());r['config']['num_agents']=4
        (p/'result.json').write_text(json.dumps(r))
    output=tmp_path/'tables'
    main(['--root',str(root),'--out',str(output),'--source','Houston13','--target','Houston18',
          '--models','BiDA','AgentBiDA','--seeds','1','2'])
    assert len(list(output.glob('*.csv')))==1
    assert '100.00 ± 0.00' in next(output.glob('*.md')).read_text()
    assert '\\begin{table*}' in next(output.glob('*.tex')).read_text()
    stats=json.loads(next(output.glob('*.paired.json')).read_text())
    assert all(row['mean_delta']==0 for row in stats)
