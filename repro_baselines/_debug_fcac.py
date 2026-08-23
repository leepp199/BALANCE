"""Quick debug: run fully_fcac with S1 proto trace."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['LIBSNDFILE_PREVENT_FORK'] = '1'

import torch
import torch.nn.functional as F
import numpy as np

from repro_baselines.train import train_one_experiment
from repro_baselines.methods.cil import build_cil

device = torch.device('cuda:0')

# Monkey-patch: after register_novel_classes, check proto integrity
from repro_baselines.methods.cil.fully_fcac import FullyFCAC
orig_register = FullyFCAC.register_novel_classes

def debug_register(self, support_feats, class_ids, *args, **kwargs):
    old_protos = self._protos[0:80].clone().detach()
    result = orig_register(self, support_feats, class_ids, *args, **kwargs)
    new_protos = self._protos[0:80].detach()
    diff = (old_protos - new_protos).abs().max().item()
    print(f"\n[DEBUG] register_novel_classes: class_ids={list(class_ids)}")
    print(f"[DEBUG] _protos[0:80] max change: {diff:.8f}")
    print(f"[DEBUG] _protos[85:100] are zero: {(self._protos[85:].abs().sum() < 1e-8).item()}")
    return result

FullyFCAC.register_novel_classes = debug_register

# Also patch compute_acc for session 1
import repro_baselines.train as T
orig_compute_acc = T.compute_acc

def debug_compute_acc(model, cil_method, test_loader, n_known, device):
    result = orig_compute_acc(model, cil_method, test_loader, n_known, device)
    if n_known >= 85:
        protos = cil_method.prototypes(n_known)
        feats_list, labs_list = [], []
        for batch in test_loader:
            x, y = batch
            if isinstance(x, (list, tuple)): x = x[0]
            x, y = x.to(device), y.to(device)
            feats = T.extract_feats(model, x)
            feats_list.append(feats); labs_list.append(y)
        feats = torch.cat(feats_list, 0)
        labs = torch.cat(labs_list, 0)
        pf = F.normalize(feats, dim=-1) @ F.normalize(protos, dim=-1).t()
        preds = pf.argmax(dim=1)
        for c in range(0, min(5, n_known)):
            m = (labs == c)
            if m.sum() > 0:
                print(f'[DEBUG] S1 class {c}: acc={(preds[m]==c).float().mean().item():.4f} n={m.sum().item()}')
        for c in range(80, min(n_known, 85)):
            m = (labs == c)
            if m.sum() > 0:
                print(f'[DEBUG] S1 novel class {c}: acc={(preds[m]==c).float().mean().item():.4f} n={m.sum().item()}')
    return result

T.compute_acc = debug_compute_acc

print("Running fully_fcac × costarr debug...")
train_one_experiment(
    cil_name='fully_fcac',
    osr_name='costarr',
    dataset='librispeech',
    dataroot='/data/datasets/librispeech_fscil/',
    num_base=80, num_novel=20, num_all=100,
    device=device,
    seed=3420, n_sessions=5,
    log_dir='/data/lqq/baseline/repro_baselines/logs/librispeech/debug_fcac',
)
print("Done.")
