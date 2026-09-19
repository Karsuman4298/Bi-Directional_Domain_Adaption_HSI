"""Controlled adapters around the supplied external architectures.

These are corrected repository recipes, NOT claims of exact reproduction of
published numbers. See docs/FAIR_EXPERIMENTS.md for settings and deviations.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import augment
from .losses import supcon, lmmd, cdd, entropy, marginal_entropy


def optimize(loss, *optimizers):
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite external-method objective')
    loss.backward()
    for opt in optimizers: opt.step()


def reset(*optimizers):
    for opt in optimizers: opt.zero_grad(set_to_none=True)


class External(nn.Module):
    def __init__(self,args):
        super().__init__(); self.args=args

    def set_lr(self,epoch):
        lr=self.args.lr/(1+10*(epoch-1)/self.args.epochs)**.75
        for opt in self.optimizers:
            for group in opt.param_groups: group['lr']=lr

    def prepare_epoch(self,epoch,source,target,device,workers,smoke=False):
        self.set_lr(epoch)


class TST(External):
    recipe='TSTnet supplied CNN/GraphSAGE + linear MMD + Wasserstein/GW + CNN/GCN soft supervision'
    def __init__(self,args):
        super().__init__(args)
        from external_methods.TSTnet.model.TSTnet import Feature_Extractor
        self.net=Feature_Extractor(48,7,args.patch_size)
        self.opt=torch.optim.SGD(self.parameters(),lr=args.lr,momentum=.9,weight_decay=1e-4)
        self.optimizers=[self.opt]

    def predict(self,x): return self.net(x)[0]

    def step(self,xs,ys,xt,ids,epoch):
        self.train();reset(self.opt)
        logits,mmd,wd,gw,_,graph=self.net(xs,xt)
        ce=F.cross_entropy(logits,ys)+F.cross_entropy(graph,ys)
        consistency=-(logits.softmax(1)*graph.log_softmax(1)).sum(1).mean()
        loss=ce+mmd+.1*(wd+gw)+consistency
        optimize(loss,self.opt)
        return {'loss':loss.item(),'cls':ce.item(),'mmd':mmd.item(),'transport':(wd+gw).item()}


class CleanMixin:
    """Source-trained SVM checks network pseudo-labels; no target labels are read."""
    @torch.no_grad()
    def clean_targets(self,source,target,device):
        from sklearn.svm import SVC
        from cleanlab.filter import find_label_issues
        self.eval()
        sf,sl,tf,tp=[],[],[],[]
        for x,y in DataLoader(source,batch_size=self.args.batch_size):
            sf.append(self.embedding(x.to(device)).cpu().numpy());sl.append(y.numpy())
        for x,_ in DataLoader(target,batch_size=self.args.batch_size):
            x=x.to(device);tf.append(self.embedding(x).cpu().numpy())
            tp.append(self.predict(x).argmax(1).cpu().numpy())
        sf,sl,tf,tp=map(np.concatenate,(sf,sl,tf,tp))
        # Bound the quadratic SVM cost in the 50k-source reverse direction.
        # This caps only SVM refinement fitting, not CNN source supervision.
        rng=np.random.default_rng(self.args.seed)
        fit_ids=np.concatenate([rng.choice(np.flatnonzero(sl==c),
            size=min(180,int((sl==c).sum())),replace=False) for c in range(7)])
        svm=SVC(probability=True,random_state=self.args.seed)
        svm.fit(sf[fit_ids],sl[fit_ids])
        probs=svm.predict_proba(tf)
        # Small/absent pseudo classes cannot support pruning; cleanlab retains them.
        if len(np.unique(tp)) < 2:
            # No class-separation evidence: do not self-train a collapsed label pool.
            issue=np.ones(len(tp),dtype=bool)
            print('Pseudo-label pool contains one class; skipping clean target CE until next refresh.',flush=True)
        else:
            issue=find_label_issues(tp,probs,filter_by='prune_by_class',min_examples_per_class=1,n_jobs=1)
        self.clean_mask=torch.as_tensor(~issue,device=device)
        self.pseudo=torch.as_tensor(tp,device=device,dtype=torch.long)
        original=np.bincount(tp,minlength=7)
        kept=np.bincount(tp[~issue],minlength=7)
        self.clean_weights=torch.as_tensor(original/np.maximum(kept,1),device=device,dtype=torch.float32)
        print(f'Confident learning kept {int((~issue).sum())}/{len(tp)} pseudo-labels',flush=True)
        self.train()


class CLDA(CleanMixin,External):
    recipe='CLDA supplied embedding/two classifiers + discrepancy min-max + SVM/cleanlab pseudo-label refinement'
    def __init__(self,args):
        super().__init__(args)
        from external_methods.CLDA.basenet import EmbeddingNetHyperX,ResClassifier
        self.g=EmbeddingNetHyperX(48,128,args.patch_size)
        self.f1=ResClassifier();self.f2=ResClassifier()
        self.og=torch.optim.SGD(self.g.parameters(),lr=args.lr,weight_decay=.0005)
        self.of=torch.optim.SGD(list(self.f1.parameters())+list(self.f2.parameters()),lr=args.lr,momentum=.9,weight_decay=.0005)
        self.optimizers=[self.og,self.of];self.clean_mask=None

    def embedding(self,x): return self.g(x)
    def predict(self,x):
        f=self.g(x);return self.f1(f)+self.f2(f)

    def prepare_epoch(self,epoch,source,target,device,workers,smoke=False):
        self.set_lr(epoch)
        if not smoke and epoch>=40 and (epoch-40)%20==0: self.clean_targets(source,target,device)

    def step(self,xs,ys,xt,ids,epoch):
        self.train();n=len(xs);x=torch.cat((xs,xt))
        reset(*self.optimizers)
        f=self.g(x);a,b=self.f1(f),self.f2(f)
        loss=F.cross_entropy(a[:n],ys)+F.cross_entropy(b[:n],ys)+.01*(marginal_entropy(a[n:])+marginal_entropy(b[n:]))
        if self.clean_mask is not None:
            mask=self.clean_mask[ids]
            if mask.any():
                y=self.pseudo[ids][mask]
                loss=loss+.1*(F.cross_entropy(a[n:][mask],y,weight=self.clean_weights)+F.cross_entropy(b[n:][mask],y,weight=self.clean_weights))+.01*(entropy(a[n:][mask])+entropy(b[n:][mask]))
        optimize(loss,*self.optimizers)
        reset(*self.optimizers)
        # Classifier discrepancy maximization, then feature minimization.
        f=self.g(x);a,b=self.f1(f),self.f2(f)
        disc=cdd(a[n:].softmax(1),b[n:].softmax(1))
        obj=F.cross_entropy(a[:n],ys)+F.cross_entropy(b[:n],ys)-.01*disc+.01*(marginal_entropy(a[n:])+marginal_entropy(b[n:]))
        optimize(obj,self.of)
        for _ in range(4):
            reset(*self.optimizers)
            f=self.g(x);a,b=self.f1(f)[n:],self.f2(f)[n:]
            obj=.01*cdd(a.softmax(1),b.softmax(1))+.01*(marginal_entropy(a)+marginal_entropy(b))
            optimize(obj,self.og)
        return {'loss':loss.item(),'discrepancy':disc.item()}


class Contrastive(CleanMixin,External):
    recipe='SCLUDA/MLUDA supplied dual-head network + LMMD + two-view supervised/pseudo contrastive + occupancy'
    def __init__(self,args):
        super().__init__(args);self.multi=args.model=='MLUDA'
        if self.multi:
            from external_methods.MLUDA.net2 import DSANSS
            self.net=DSANSS(48,args.patch_size,7)
        else:
            from external_methods.SCLUDA.net import DSAN
            self.net=DSAN(48,args.patch_size,7)
        self.opt=torch.optim.SGD(self.parameters(),lr=args.lr,momentum=.9,weight_decay=.0005)
        self.optimizers=[self.opt];self.clean_mask=None

    def outputs(self,x): return self.net(x,x)[:5] if self.multi else self.net(x)
    def embedding(self,x): return self.outputs(x)[0]
    def predict(self,x): return self.outputs(x)[3]
    def pair(self,x,y):
        if self.multi:
            o=self.net(x,y);return o[:5],o[5:]
        return self.net(x),self.net(y)

    def prepare_epoch(self,epoch,source,target,device,workers,smoke=False):
        self.set_lr(epoch)
        if not self.multi and not smoke and epoch>=40 and (epoch-40)%20==0:
            self.clean_targets(source,target,device)

    def step(self,xs,ys,xt,ids,epoch):
        self.train();reset(self.opt)
        s,t=self.pair(xs,xt)
        s1,t1=self.pair(augment(xs,.05),augment(xt,.05))
        s2,t2=self.pair(augment(xs,.05),augment(xt,.05))
        pseudo=t[3].detach().argmax(1)
        ce=F.cross_entropy(s[3],ys)
        alignment=lmmd(s[0],t[0],ys,t[3].softmax(1))
        contrast=supcon(torch.stack((s1[1],s2[1]),1),ys)+supcon(torch.stack((t1[2],t2[2]),1),pseudo)
        occupancy=-(s[4].clamp_min(1e-6).log().mean()+t[4].clamp_min(1e-6).log().mean())
        ramp=2/(1+math.exp(-10*(epoch-1)/self.args.epochs))-1
        loss=ce+.01*ramp*alignment+contrast+occupancy
        if self.clean_mask is not None:
            mask=self.clean_mask[ids]
            if mask.any(): loss=loss+.03*F.cross_entropy(t[3][mask],self.pseudo[ids][mask])
        optimize(loss,self.opt)
        return {'loss':loss.item(),'cls':ce.item(),'lmmd':alignment.item(),'contrast':contrast.item()}


class SSWADA(External):
    recipe='SSWADA supplied spatial/spectral extractor + 3 classifiers + weighted domain adversaries; repaired discriminator optimization'
    def __init__(self,args):
        super().__init__(args)
        from external_methods.SSWADA.FeatureNet import Features,ResClassifier
        from external_methods.SSWADA.domain_discriminator import DomainDiscriminator
        from external_methods.SSWADA.dann import DomainAdversarialLoss
        self.g=Features(48,7,args.patch_size)
        self.c=ResClassifier(7);self.f1=ResClassifier(7);self.f2=ResClassifier(7)
        self.d=DomainDiscriminator(2048,1024,False);self.d0=DomainDiscriminator(2048,1024,False)
        self.adv=DomainAdversarialLoss(self.d0)
        self.og=torch.optim.SGD(self.g.parameters(),lr=args.lr,weight_decay=.0005)
        self.oc=torch.optim.SGD(list(self.c.parameters())+list(self.f1.parameters())+list(self.f2.parameters()),lr=args.lr,momentum=.9,weight_decay=.0005)
        self.od=torch.optim.SGD(self.d.parameters(),lr=args.lr,weight_decay=.0005)
        self.od0=torch.optim.SGD(self.d0.parameters(),lr=args.lr,weight_decay=.0005)
        self.optimizers=[self.og,self.oc,self.od,self.od0]

    def predict(self,x): return self.c(self.g(x))

    def step(self,xs,ys,xt,ids,epoch):
        self.train();reset(*self.optimizers)
        fs,ft=self.g(xs),self.g(xt)
        cls=F.cross_entropy(self.c(fs),ys)+.01*marginal_entropy(self.c(ft))
        optimize(cls,self.og,self.oc)
        reset(*self.optimizers)
        fs,ft=self.g(xs),self.g(xt)
        ds,dt=self.d(fs.detach()),self.d(ft.detach())
        dloss=.5*(F.binary_cross_entropy(ds,torch.ones_like(ds))+F.binary_cross_entropy(dt,torch.zeros_like(dt)))
        optimize(dloss,self.od)
        reset(*self.optimizers)
        with torch.no_grad():
            w=1-self.d(fs);w=w/w.mean().clamp_min(1e-8)
        # Correct adversarial sign: GRL already reverses the feature gradient.
        adv=self.adv(fs,ft,len(xs),w_s=w)
        p=[net(ft).softmax(1) for net in (self.c,self.f1,self.f2)]
        disc=sum((p[i]-p[j]).abs().mean() for i,j in ((0,1),(0,2),(1,2)))
        classifier_loss=F.cross_entropy(self.c(fs),ys)+.1*disc+.5*adv
        optimize(classifier_loss,self.oc,self.od0)
        reset(*self.optimizers)
        # Third stage updates the feature extractor, with fixed classifiers/adversary.
        frozen=(self.c,self.f1,self.f2,self.d0)
        for module in frozen:
            for param in module.parameters(): param.requires_grad_(False)
        fs,ft=self.g(xs),self.g(xt)
        with torch.no_grad():
            w=1-self.d(fs);w=w/w.mean().clamp_min(1e-8)
        adv=self.adv(fs,ft,len(xs),w_s=w)
        p=[net(ft).softmax(1) for net in (self.c,self.f1,self.f2)]
        disc=sum((p[i]-p[j]).abs().mean() for i,j in ((0,1),(0,2),(1,2)))
        loss=.1*disc+.5*adv
        optimize(loss,self.og)
        for module in frozen:
            for param in module.parameters(): param.requires_grad_(True)
        return {'loss':loss.item(),'cls':cls.item(),'domain':adv.item(),'discrepancy':disc.item()}


class CACL(External):
    recipe='CACL supplied CNN/mapper + adversarial domain loss + spectral/spatial prototype consistency and focal/contrastive target losses'
    def __init__(self,args):
        super().__init__(args)
        from external_methods.CACL.Net_singleDA import Network,Discriminator
        self.net=Network(48,7);self.dis=Discriminator()
        self.og=torch.optim.Adam(self.net.parameters(),lr=args.lr)
        self.od=torch.optim.Adam(self.dis.parameters(),lr=args.lr)
        self.optimizers=[self.og,self.od];self.prototypes=None

    def predict(self,x): return self.net(x)[1]

    @torch.no_grad()
    def prepare_epoch(self,epoch,source,target,device,workers,smoke=False):
        # CACL uses a constant learning rate in its supplied recipe.
        if epoch < 21 or (epoch-21)%20: return
        self.eval();spa=torch.zeros(7,128,device=device);spe=torch.zeros(7,self.args.patch_size**2,device=device)
        count=torch.zeros(7,device=device)
        for x,y in DataLoader(source,batch_size=self.args.batch_size):
            y=y.to(device);f=self.net(x.to(device))[0].flatten(2)
            spa.index_add_(0,y,f.mean(-1));spe.index_add_(0,y,f.mean(1));count.index_add_(0,y,torch.ones_like(y,dtype=torch.float))
        self.prototypes=(F.normalize(spa/count[:,None].clamp_min(1),dim=1),F.normalize(spe/count[:,None].clamp_min(1),dim=1))
        self.train()

    def step(self,xs,ys,xt,ids,epoch):
        self.train();reset(*self.optimizers)
        es,ps,ms=self.net(xs);et,pt,mt=self.net(xt)
        ones=torch.ones(len(xs),device=xs.device,dtype=torch.long);zeros=torch.zeros_like(ones)
        dl=F.cross_entropy(self.dis(es.detach()),ones)+F.cross_entropy(self.dis(et.detach()),zeros)
        optimize(dl,self.od);reset(*self.optimizers)
        for p in self.dis.parameters():p.requires_grad_(False)
        adv=-(F.cross_entropy(self.dis(es),ones)+F.cross_entropy(self.dis(et),zeros))
        ce=F.cross_entropy(ps,ys);tl=ce.new_zeros(())
        if self.prototypes is not None:
            f=et.detach().flatten(2)
            spatial=(F.normalize(f.mean(-1),dim=1)@self.prototypes[0].T).argmax(1)
            spectral=(F.normalize(f.mean(1),dim=1)@self.prototypes[1].T).argmax(1)
            y=pt.detach().argmax(1);consistent=(y==spatial)&(y==spectral)
            for mask,is_consistent in ((consistent,True),(~consistent,False)):
                if not mask.any():continue
                fraction=mask.float().mean()
                features=torch.cat((ms,mt[mask])) if is_consistent else mt[mask]
                labels=torch.cat((ys,y[mask])) if is_consistent else y[mask]
                contrast=supcon(features.flatten(1).unsqueeze(1),labels,temperature=.07)
                per=F.cross_entropy(pt[mask],y[mask],reduction='none')
                focal=((1-(-per).exp()).pow(1-fraction)*per).mean()
                tl=tl+fraction*(contrast+focal)
        loss=ce+.001*adv+tl
        optimize(loss,self.og)
        for p in self.dis.parameters():p.requires_grad_(True)
        return {'loss':loss.item(),'cls':ce.item(),'adversarial':adv.item(),'target':tl.item()}


def preprocess(args,source,target):
    if args.model!='MLUDA':return source,target
    # Original guided filter requests two PCA guide channels, unsupported by OpenCV.
    # Use three, the supported RGB guidance case, and record this protocol change.
    import cv2
    from sklearn.decomposition import PCA
    from skimage.exposure import match_histograms
    if not hasattr(cv2,'ximgproc'):raise ImportError('MLUDA needs opencv-contrib-python-headless (see requirements-fair.txt)')
    def guide(image):
        p=PCA(3,svd_solver='full').fit_transform(image.reshape(-1,48)).reshape(*image.shape[:2],3)
        return (p-p.min((0,1)))/np.maximum(np.ptp(p,axis=(0,1)),1e-8)
    sg,tg=guide(source),guide(target)
    gamma=max(float(sg.mean()/max(tg.mean(),1e-8)),1e-8)
    sg=match_histograms(sg**gamma,tg,channel_axis=-1).astype(np.float32)
    tg=tg.astype(np.float32)
    def filtered(image,g):
        # Channel-by-channel destination avoids OpenCV version limits on 48-channel inputs.
        return np.stack([cv2.ximgproc.guidedFilter(g,image[...,i].copy(),1,.009)
                         for i in range(48)],axis=-1).astype(np.float32)
    return filtered(source,sg),filtered(target,tg)


def make_method(args):
    return {'TSTnet':TST,'CLDA':CLDA,'SCLUDA':Contrastive,'MLUDA':Contrastive,
            'SSWADA':SSWADA,'CACL':CACL}[args.model](args)
