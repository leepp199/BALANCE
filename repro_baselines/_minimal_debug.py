"""Minimal FCAC debug: test if proto addition causes collapse."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['LIBSNDFILE_PREVENT_FORK'] = '1'

import torch, numpy as np
import torch.nn.functional as F
device = torch.device('cuda:0')

from repro_baselines.train import init_audio_dataset
from data.dataloader import get_pretrain_dataloader, get_testloader

class Args: pass
args = Args()
args.dataset = 'librispeech'
args.dataroot = '/data/datasets/librispeech_fscil/'
args.num_base = 80; args.num_novel = 20; args.num_all = 100
args.way = 5; args.n_ways = 5; args.n_shots = 5; args.n_queries = 15
args.n_open_ways = 5; args.train_episode = 50; args.tmp_train = False
args.test_times = 50; args.feat_dim = 512; args.num_labeled_classes = 100
args.seq_sample = False; args.seed = 3420; args.lr_new = 0.1
from addict import Dict
args.epochs = type('E', (), {'epochs_std': 5, 'epochs_new': 5})()
args.lr = type('L', (), {'lr_std': 0.005, 'lr_new': 0.1, 'lrg': 0.1})()
args.optimizer = Dict({'decay': 5e-4, 'momentum': 0.9})
args.scheduler = Dict({'schedule': 'Step', 'step': 40, 'gamma': 0.5})
args.network = Dict({'temperature': 1, 'base_mode': 'ft_cos', 'new_mode': 'ft_cos'})
args.episode = Dict({'train_episode': 50, 'low_way': 5, 'low_shot': 5, 'episode_way': 5, 'episode_shot': 5, 'episode_query': 15})
args.stdu = Dict({'num_tmpb': 55, 'num_tmpi': 25, 'num_tmps': 14, 'num_incre': 5})
args.dataloader = Dict({'num_workers': 2, 'train_batch_size': 128, 'test_batch_size': 100})
args.extractor = Dict({'sample_rate': 16000, 'window_size': 400, 'hop_size': 160, 'mel_bins': 128, 'fmin': 0, 'fmax': 8000, 'window': 'hann'})
args.train_weight_base = True
LBRS, OpenDS = init_audio_dataset('librispeech')
args.Dataset = type('DS', (), {'LBRS': LBRS, 'Openlbrs': OpenDS, 'NDS': LBRS, 'Opennds': OpenDS, 'FSDCLIPS': LBRS, 'Openfs': OpenDS})()

# Build data
trainset, base_loader = get_pretrain_dataloader(args)
_, test_loader_0 = get_testloader(args, 0)  # classes 0-79
print(f"Base train loader: {len(base_loader)} batches")

# Build fully_fcac
from repro_baselines.methods.cil.fully_fcac import FullyFCAC
method = FullyFCAC(args).to(device)
model = method.model
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Quick train (5 epochs) with CE
from repro_baselines.methods.base import train_backbone_with_loss
from repro_baselines.train import extract_feats, compute_acc

print("\n--- Base Training (5 epochs) ---")
train_backbone_with_loss(model, args, base_loader, epochs=5, tag='debug_fcac')

# Ridge regression fit
method._extract_and_fit(base_loader, args.num_base, device)
acc_0 = compute_acc(model, method, test_loader_0, args.num_base, device)
print(f"\nSession 0 all_acc = {acc_0:.4f}")

# Now test: does adding novel prototypes affect base class accuracy?
print("\n--- Novel prototype injection test ---")
# Get some test features as stand-in novel protos
feats_list = []
for batch in test_loader_0:
    x = batch[0].to(device)
    feats = extract_feats(model, x)
    feats_list.append(feats[:5])
    break
novel = torch.cat(feats_list, 0)  # [5, 512] real features

# Track _protos before
before = method._protos[0:5].clone()

# Register with different class IDs
method.register_novel_classes(novel, [80, 81, 82, 83, 84])

# Check _protos after
after = method._protos[0:5]
diff = (before - after).abs().max().item()
print(f"_protos[0:5] max change: {diff:.8f}")
print(f"_protos[80:85] == novel: {(method._protos[80:85] == novel).all().item()}")

# Evaluate base class accuracy again
acc_0_after = compute_acc(model, method, test_loader_0, args.num_base, device)
print(f"\nSession 0 (after novel register, n_known=80) all_acc = {acc_0_after:.4f}")

# Evaluate with n_known=85
acc_85 = compute_acc(model, method, test_loader_0, 85, device)
print(f"Session 0 test (with n_known=85) all_acc = {acc_85:.4f}")

# Per-class analysis with n_known=85
method.eval(); model.eval()
feats_all, labs_all = [], []
for batch in test_loader_0:
    x, y = batch[0].to(device), batch[1].to(device)
    feats_all.append(extract_feats(model, x))
    labs_all.append(y)
feats_all = torch.cat(feats_all)
labs_all = torch.cat(labs_all)
protos = method.prototypes(85)
logits = F.normalize(feats_all) @ F.normalize(protos).t()
preds = logits.argmax(dim=1)

print("\nPer-class acc (n_known=85):")
for c in [0, 1, 2, 10, 40, 79, 80, 81, 82, 83, 84]:
    m = (labs_all == c)
    if m.sum() > 0:
        acc = (preds[m] == c).float().mean().item()
        print(f"  class {c}: acc={acc:.4f} n={m.sum().item()}")

# Summary: how many of class 0-79 are classified correctly?
mask_base = labs_all < 80
acc_base = (preds[mask_base] == labs_all[mask_base]).float().mean().item()
print(f"\nBase class accuracy (0-79): {acc_base:.4f}")
n_pred_as_novel = (preds[mask_base] >= 80).float().mean().item()
print(f"Base samples predicted as novel (80-84): {n_pred_as_novel:.4f}")

# Check the cosine similarity distribution
logits_fix = F.normalize(feats_all) @ F.normalize(method._protos[:80]).t()  # only 80 base protos
preds_fix = logits_fix.argmax(dim=1)
acc_fix = (preds_fix == labs_all).float().mean().item()
print(f"\nWith ONLY base 80 protos: all_acc = {acc_fix:.4f}")
