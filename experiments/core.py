"""One optimization recipe for all three BiDA architectures and their ablations."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from loss.mmd_loss import MMD_loss
from .data import augment


def pair_indices(source_labels, target_logits, threshold=.5):
    """Same-class pairs from predictions only; no target ground truth argument."""
    p = target_logits.detach().softmax(-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
    pseudo = p.argmax(-1)
    valid = entropy <= threshold * math.log(p.shape[-1])
    source, target = [], []
    for i, label in enumerate(source_labels):
        candidates = torch.where(valid & (pseudo == label))[0]
        if len(candidates):
            source.append(i)
            target.append(candidates[torch.randint(len(candidates), (), device=candidates.device)])
    return (torch.tensor(source, dtype=torch.long, device=source_labels.device),
            torch.stack(target) if target else source_labels.new_empty(0))


@torch.no_grad()
def update_teacher(student, teacher, decay):
    for dst, src in zip(teacher.parameters(), student.parameters()):
        dst.lerp_(src, 1-decay)
    # Copy BN statistics; averaging integer counters is invalid.
    for dst, src in zip(teacher.buffers(), student.buffers()):
        dst.copy_(src)


def soft_ce(teacher, student):
    # Joint coupled-branch soft supervision, as in the repository's BiDA loss.
    return -(teacher.softmax(-1) * student.log_softmax(-1)).sum(-1).mean()


class CoreMethod(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        from models.get_model import get_model
        self.model = get_model(args.model, args.source, args.patch_size, args)
        self.teacher = copy.deepcopy(self.model).requires_grad_(False)
        self.teacher.eval()
        self.mmd = MMD_loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr,
                                        momentum=0., weight_decay=0.)
        self.updates = 0

    def predict(self, x):
        if self.args.model == 'GAHT':
            return self.model(x.unsqueeze(1))
        return self.model(None, x.unsqueeze(1), inference_target_only=True)[1]

    def step(self, xs, ys, xt, target_ids, epoch):
        self.model.train()
        self.teacher.eval()
        a = self.args
        self.optimizer.zero_grad(set_to_none=True)
        if a.model == 'GAHT' or a.ablation == 'source_only':
            logits = self.predict(augment(xs, a.noise_std))
            loss = F.cross_entropy(logits, ys)
            loss.backward(); self.optimizer.step()
            return {'loss': float(loss.detach()), 'cls': float(loss.detach())}
        s, t = augment(xs, a.noise_std).unsqueeze(1), augment(xt, a.noise_std).unsqueeze(1)
        out, feat = self.model(s, t, inference_target_only=False, return_features=True)
        cls = F.cross_entropy(out[0], ys)
        zero = cls.new_zeros(())
        align, distill, con = zero, zero, zero
        if a.ablation != 'no_mmd' and epoch >= a.mmd_start:
            align = self.mmd(feat[0], feat[1])
        if a.ablation != 'no_consistency':
            with torch.no_grad():
                es, et, _, _ = self.teacher(augment(xs, a.noise_std).unsqueeze(1),
                    augment(xt, a.noise_std).unsqueeze(1), inference_target_only=False)
            con = F.mse_loss(out[0].softmax(1), es.softmax(1)) + F.mse_loss(out[1].softmax(1), et.softmax(1))
        pairs = 0
        if a.ablation != 'no_distill' or (a.ablation != 'no_mmd' and epoch >= a.mmd_start):
            with torch.no_grad():
                pseudo_logits = self.teacher(None, xt.unsqueeze(1), inference_target_only=True)[1]
            si, ti = pair_indices(ys, pseudo_logits, a.entropy_threshold)
            if a.ablation == 'no_pairing':
                si = torch.arange(len(xs), device=xs.device)
                ti = torch.arange(len(xt), device=xt.device)
            pairs = len(si)
            if pairs:
                coupled, cf = self.model(s[si], t[ti], inference_target_only=False, return_features=True)
                if a.ablation != 'no_distill':
                    distill = soft_ce(coupled[2], coupled[1]) + soft_ce(coupled[3], coupled[0])
                if a.ablation != 'no_mmd' and epoch >= a.mmd_start:
                    align = align + self.mmd(cf[3], cf[2])
        loss = cls + a.lambda1 * (align + distill) + a.lambda2 * con
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite BiDA loss; refusing to save a score')
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        update_teacher(self.model, self.teacher, min(a.ema_decay, 1-1/(self.updates+1)))
        return {k: float(v.detach()) for k,v in dict(loss=loss, cls=cls, mmd=align,
            distill=distill, consistency=con).items()} | {'pairs': pairs}
