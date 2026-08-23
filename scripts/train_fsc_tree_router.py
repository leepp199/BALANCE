#!/usr/bin/env python3
"""Nonlinear FSC group router trained on base-only pseudo-incremental episodes."""
import argparse, joblib, os
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from train_fsc_group_router import score_features

p = argparse.ArgumentParser()
p.add_argument('--geometry', required=True); p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True); p.add_argument('--seed', type=int, default=3420)
p.add_argument('--episodes', type=int, default=200)
a = p.parse_args(); rng = np.random.default_rng(a.seed)
g = torch.load(a.geometry, map_location='cpu', weights_only=True)
c = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
state = c.get('params', c.get('cls_params', c)); pools = [x.float() for x in g['class_features']]
xs, ys = [], []
for _ in range(a.episodes):
    novel = sorted(rng.choice(59, 10, replace=False).tolist()); base = [i for i in range(59) if i not in novel]
    bp = state['fc.weight'][base].float(); support = torch.cat([pools[i][:5] for i in novel])
    np_ = torch.stack([pools[i][:5].mean(0) for i in novel])
    bq = torch.cat([pools[i][5:10] for i in base]); nq = torch.cat([pools[i][5:25] for i in novel])
    xs += [score_features(bq, bp, np_, support), score_features(nq, bp, np_, support)]
    ys += [np.zeros(len(bq), dtype=np.int64), np.ones(len(nq), dtype=np.int64)]
x = torch.cat(xs).numpy(); y = np.concatenate(ys)
clf = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20, max_features=None,
                           class_weight='balanced', n_jobs=-1, random_state=a.seed).fit(x, y)
held = list(range(59,69)); bp=state['fc.weight'][:59].float(); support=torch.cat([pools[i][:5] for i in held]); np_=torch.stack([pools[i][:5].mean(0) for i in held])
bq=torch.cat([pools[i] for i in range(59)]); nq=torch.cat([pools[i][5:] for i in held])
vx=torch.cat([score_features(bq,bp,np_,support), score_features(nq,bp,np_,support)]).numpy()
vy=np.r_[np.zeros(len(bq),dtype=int),np.ones(len(nq),dtype=int)]
os.makedirs(os.path.dirname(a.output),exist_ok=True); joblib.dump(clf,a.output)
print(f'train={len(y)} val_bal_acc={balanced_accuracy_score(vy,clf.predict(vx)):.6f} output={a.output}')
