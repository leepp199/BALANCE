#!/usr/bin/env python3
"""
train_openset_vaze.py

Train a closed-set classifier and apply a simple open-set decision rule
inspired by Vaze et al. (ICLR 2022): use a good closed-set classifier
and thresholding on classifier confidence and feature-centroid similarity
to detect unknowns. Then perform an incremental step by clustering detected
unknowns and adding new prototypes.

This script reuses dataset / model code from the repo; it only replaces the
open-set detection and incremental logic. Defaults follow existing code.

Usage (smoke-run):
  python baseline/train_openset_vaze.py --smoke
"""
import os
import sys
import argparse
import yaml
from types import SimpleNamespace
import time
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)


def build_min_args():
    # minimal args for model construction and data loaders
    a = SimpleNamespace()
    a.dataloader = SimpleNamespace()
    a.dataloader.train_batch_size = 64
    a.dataset = 'librispeech'
    a.dataroot = '/data/datasets/librispeech_fscil/'
    a.num_labeled_classes = 80
    a.num_unlabeled_classes = 5
    a.save_dir = 'baseline/save'
    a.save_result = 'baseline/save_result'
    a.seed = 42
    a.cuda = torch.cuda.is_available()
    a.epochs = SimpleNamespace(epochs_std=1)
    a.way = 5
    a.n_ways = 5
    a.n_shots = 5
    a.n_queries = 5
    a.start_session = 1
    a.num_session = 5
    a.test_times = 1
    a.pretrained_model_path = os.path.join(a.save_dir, 'base_train_for_meta.pth')
    a.train_weight_base = 0
    a.neg_gen_type = 'att'
    a.agg = 'avg'
    a.base_seman_calib = True
    a.n_ways = a.way
    a.pit_num_new_classes = 5
    # audio extractor defaults required by MYNET.set_module_for_audio
    a.extractor = SimpleNamespace()
    a.extractor.sample_rate = 16000
    a.extractor.window_size = 400
    a.extractor.hop_size = 160
    a.extractor.window = 'hann'
    a.extractor.mel_bins = 128
    a.extractor.fmin = 0
    a.extractor.fmax = 8000
    # network defaults
    a.network = SimpleNamespace()
    a.network.new_mode = 'cos'
    a.network.temperature = 10.0
    # optimization defaults used by get_optimizer
    a.lr = SimpleNamespace()
    a.lr.lr_std = 0.01
    a.lr.lrg = 0.01
    a.optimizer = SimpleNamespace()
    a.optimizer.decay = 0.0
    a.scheduler = SimpleNamespace()
    a.scheduler.schedule = 'Step'
    a.scheduler.step = 10
    a.scheduler.gamma = 0.1
    a.scheduler.milestones = []
    return a


def train_closed_classifier(args, model, train_loader, optimizer, scheduler, epochs=1):
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            data, label = [_.cuda() for _ in batch] if torch.cuda.is_available() else batch
            model.mode = 'encoder'
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return model


def compute_class_centroids(model, loader, classes=None):
    model.eval()
    feats = []
    labs = []
    with torch.no_grad():
        for batch in loader:
            data, label = batch[0], batch[1]
            if torch.cuda.is_available():
                data = data.cuda()
            # use base_encode to get features before fc
            if hasattr(model, 'base_encode'):
                f = model.base_encode(data, augment=False)
            else:
                f = model.encode(data)
            if f.dim() > 2:
                f = f.mean(dim=[2,3])
            feats.append(f.cpu())
            labs.append(label)
    feats = torch.cat(feats, dim=0).numpy()
    labs = torch.cat(labs, dim=0).numpy()
    if classes is None:
        classes = np.unique(labs)
    centers = []
    for c in classes:
        idx = np.where(labs == c)[0]
        if len(idx) == 0:
            centers.append(np.zeros(feats.shape[1], dtype=float))
        else:
            centers.append(feats[idx].mean(axis=0))
    return np.vstack(centers), classes


def detect_unknowns(model, loader, centers, classes, prob_thresh=0.5, cos_thresh=0.4):
    """
    Detect unknown samples using (max softmax < prob_thresh) OR (max cosine < cos_thresh).
    Returns list of (feature, predicted_label_or_-1) for all samples in loader.
    """
    model.eval()
    results = []
    centers_t = torch.tensor(centers).float()
    if torch.cuda.is_available():
        centers_t = centers_t.cuda()
    with torch.no_grad():
        for batch in loader:
            data, label = batch[0], batch[1]
            if torch.cuda.is_available():
                data = data.cuda()
            # get logits from classifier
            model.mode = 'encoder'
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            maxp, pred = probs.max(dim=1)
            # get features for centroid similarity
            if hasattr(model, 'base_encode'):
                f = model.base_encode(data, augment=False)
            else:
                f = model.encode(data)
            if f.dim() > 2:
                f = f.mean(dim=[2,3])
            # cosine similarity
            f_norm = F.normalize(f, dim=1)
            c_norm = F.normalize(centers_t, dim=1)
            sim = torch.matmul(f_norm, c_norm.t())
            maxsim, sim_idx = sim.max(dim=1)
            for i in range(f.shape[0]):
                is_unknown = (maxp[i].item() < prob_thresh) or (maxsim[i].item() < cos_thresh)
                feat = f[i].cpu().numpy()
                results.append((feat, -1 if is_unknown else int(sim_idx[i].item())))
    return results


def incremental_add_prototypes(model, unknown_feats, n_new):
    """Cluster unknown_feats into n_new clusters and add their centroids to model.fc.weight."""
    from sklearn.cluster import KMeans
    if len(unknown_feats) == 0:
        return model
    X = np.stack(unknown_feats)
    n_clusters = min(n_new, max(1, X.shape[0] // 5))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(X)
    centroids = torch.tensor(kmeans.cluster_centers_).float()
    if torch.cuda.is_available():
        centroids = centroids.cuda()
    # append centroids to fc.weight (expand classifier)
    with torch.no_grad():
        old_w = model.fc.weight.data
        # ensure centroid dimensionality matches old_w feature dim
        feat_dim_old = old_w.shape[1]
        if centroids.shape[1] != feat_dim_old:
            if centroids.shape[1] < feat_dim_old:
                pad = torch.zeros((centroids.shape[0], feat_dim_old - centroids.shape[1]), device=centroids.device)
                centroids = torch.cat([centroids, pad], dim=1)
            else:
                centroids = centroids[:, :feat_dim_old]
        new_w = torch.cat([old_w, centroids], dim=0)
        model.fc = torch.nn.Linear(new_w.shape[1], new_w.shape[0]).to(new_w.device)
        model.fc.weight.data.copy_(new_w)
    return model


def smoke_run():
    # quick smoke: load small subset and do one train epoch + detection
    import train_unopenset as tu
    parser = tu.args_parser()
    args_ds = parser.parse_known_args([])[0]
    # provide minimal defaults
    if not hasattr(args_ds, 'dataloader'):
        args_ds.dataloader = SimpleNamespace()
        args_ds.dataloader.train_batch_size = 32
        args_ds.dataloader.test_batch_size = 32
        args_ds.dataloader.num_workers = 4
    # ensure num_base exists for dataloader
    if not hasattr(args_ds, 'num_base'):
        args_ds.num_base = getattr(args_ds, 'num_labeled_classes', 20)
    if not hasattr(args_ds, 'num_labeled_classes'):
        args_ds.num_labeled_classes = args_ds.num_base
    if not hasattr(args_ds, 'dataroot'):
        args_ds.dataroot = './librispeech'
    tu.set_up_datasets(args_ds)
    trainset, trainloader = tu.get_pretrain_dataloader(args_ds)

    # build model
    from network import MYNET, get_optimizer
    args_model = build_min_args()
    model = MYNET(args_model, mode='encoder')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    opt, sched = get_optimizer(model, args_model)
    # one epoch
    print('Running one training epoch (smoke)')
    model = train_closed_classifier(args_model, model, trainloader, opt, sched, epochs=1)

    # compute centers
    centers, classes = compute_class_centroids(model, trainloader)
    print('Computed centers for classes:', classes[:10])

    # detect unknowns on same loader (smoke)
    res = detect_unknowns(model, trainloader, centers, classes, prob_thresh=0.5, cos_thresh=0.3)
    unknown_feats = [f for f, lbl in res if lbl == -1]
    print(f'Detected {len(unknown_feats)} unknowns (smoke)')

    # incremental add
    model = incremental_add_prototypes(model, unknown_feats, n_new=5)
    print('Smoke run complete; model now has fc size', model.fc.weight.shape)


def run_incremental_eval_like_baseline(args, model, save_prefix):
    """
    Reuse the same evaluation protocol/format as train_unopenset.py,
    and write a result file with vaze_ prefix for fair comparison.
    """
    import numpy as np
    import train_unopenset as tu
    from network import replace_base_fc

    # train_unopenset.known_test uses global args
    tu.args = args

    data_dict, result = {}, {}
    data_dict['train_set'], _ = tu.get_pretrain_dataloader(args)
    model = replace_base_fc(args, data_dict['train_set'], model)
    # Ensure classifier has base-weight representation initialized
    try:
        cls_params = {'fc.weight': model.fc.weight.data.clone()}
        if torch.cuda.is_available():
            cls_params['fc.weight'] = cls_params['fc.weight'].cuda()
        model.cls_classifier.init_representation(cls_params)
    except Exception as e:
        print('[VAZE] Warning: failed to init cls_classifier representation:', e)

    result_path = os.path.join(args.save_result, f'{save_prefix}test_result.txt')
    print(f'[VAZE] Writing comparison metrics to: {result_path}')

    with open(result_path, 'w') as result_file:
        session0_acc_list = []
        session_ka = [[] for _ in range(args.test_times)]
        session_uka = [[] for _ in range(args.test_times)]
        session_f1s = [[] for _ in range(args.test_times)]
        session_inc = [[] for _ in range(args.test_times)]
        session_all = [[] for _ in range(args.test_times)]

        for j in range(0, args.num_session):
            result[f'sess{j}_ak'] = []
            result[f'sess{j}_au'] = []
            result[f'sess{j}_fs'] = []
            result[f'sess{j}_inc'] = []
            result[f'sess{j}_all'] = []

        for i in range(args.test_times):
            args.current_test = i
            args.num_labeled_classes = args.num_base

            _, base_testloader = tu.get_testloader(args, 0)
            base_acc = tu.test(args, model, base_testloader, 0)
            session0_acc_list.append(base_acc)

            result['sess0_ak'].append(base_acc)
            result['sess0_au'].append(0.0)
            result['sess0_fs'].append(0.0)
            result['sess0_inc'].append(0.0)
            result['sess0_all'].append(base_acc)

            for session in range(args.start_session, args.num_session):
                model.mode = args.network.new_mode
                model.eval()

                _, unlabelled_loader = tu.get_dataloader(args, session)
                unknow_data, unknow_label, know_data, know_label = tu.run_test_fsl(model, args, unlabelled_loader)

                cluster_acc = tu.debug_cluster(args, model, unknow_data, unknow_label, session)
                acc_known, _ = tu.known_test(model, know_data, know_label)
                fscore = tu.calc(args, know_label, unknow_label)

                result[f'sess{session}_ak'].append(acc_known)
                result[f'sess{session}_au'].append(cluster_acc)
                result[f'sess{session}_fs'].append(fscore)

                _, testloader = tu.get_testloader(args, session)
                all_acc = tu.test(args, model, testloader, session)
                _, inc_testloader = tu.get_inc_testloader(args, session)
                inc_acc = tu.test(args, model, inc_testloader, session)

                result[f'sess{session}_inc'].append(inc_acc)
                result[f'sess{session}_all'].append(all_acc)
                args.num_labeled_classes += args.way

                avg_acc_known = sum(result[f'sess{session}_ak']) / len(result[f'sess{session}_ak'])
                avg_acc_unknown = sum(result[f'sess{session}_au']) / len(result[f'sess{session}_au'])
                avg_fscore = sum(result[f'sess{session}_fs']) / len(result[f'sess{session}_fs'])
                avg_inc_acc = sum(result[f'sess{session}_inc']) / len(result[f'sess{session}_inc'])
                avg_all_acc = sum(result[f'sess{session}_all']) / len(result[f'sess{session}_all'])

                session_ka[i].append(avg_acc_known)
                session_uka[i].append(avg_acc_unknown)
                session_f1s[i].append(avg_fscore)
                session_inc[i].append(avg_inc_acc)
                session_all[i].append(avg_all_acc)

                result_line = (
                    'session: {}, aac known: {:.4f}, acc unknown: {:.4f}, '
                    'f1 score: {:.4f}, inc acc: {:.4f}, all acc: {:.4f}\n'
                ).format(session, avg_acc_known, avg_acc_unknown, avg_fscore, avg_inc_acc, avg_all_acc)
                result_file.write(result_line)

        session0_acc_values = np.array(session0_acc_list)
        session0_mean = np.mean(session0_acc_values)
        session0_std = np.std(session0_acc_values)
        result_file.write('\n=== Final Session 0 ===\n')
        result_file.write(f'Average Acc: {session0_mean:.4f} ± {session0_std:.4f}\n')

        session_ka_means = []
        session_uka_means = []
        session_f1s_means = []
        session_inc_means = []
        session_all_means = []

        for ses in range(args.num_session - 1):
            ka_values = [session_ka[t][ses] for t in range(args.test_times)]
            uka_values = [session_uka[t][ses] for t in range(args.test_times)]
            f1s_values = [session_f1s[t][ses] for t in range(args.test_times)]
            inc_values = [session_inc[t][ses] for t in range(args.test_times)]
            all_values = [session_all[t][ses] for t in range(args.test_times)]

            ka_mean, ka_std = np.mean(ka_values), np.std(ka_values)
            uka_mean, uka_std = np.mean(uka_values), np.std(uka_values)
            f1s_mean, f1s_std = np.mean(f1s_values), np.std(f1s_values)
            inc_mean, inc_std = np.mean(inc_values), np.std(inc_values)
            all_mean, all_std = np.mean(all_values), np.std(all_values)

            session_ka_means.append(round(ka_mean, 4))
            session_uka_means.append(round(uka_mean, 4))
            session_f1s_means.append(round(f1s_mean, 4))
            session_inc_means.append(round(inc_mean, 4))
            session_all_means.append(round(all_mean, 4))

            result_row = (
                f'session: {ses+1}, '
                f'total aac known: {ka_mean:.4f} ± {ka_std:.4f}, '
                f'total acc unknown: {uka_mean:.4f} ± {uka_std:.4f}, '
                f'total f1 score: {f1s_mean:.4f} ± {f1s_std:.4f}, '
                f'total incremental acc: {inc_mean:.4f} ± {inc_std:.4f}, '
                f'total all acc: {all_mean:.4f} ± {all_std:.4f}\n'
            )
            result_file.write(result_row)

        aa_known = round(np.mean(session_ka_means), 4)
        aa_unknown = round(np.mean(session_uka_means), 4)
        aa_f1 = round(np.mean(session_f1s_means), 4)
        aa_inc = round(np.mean(session_inc_means), 4)
        aa_all = round(np.mean(session_all_means), 4)
        result_file.write('\n=== 4 Sessions Average Accuracy (AA) ===\n')
        result_file.write(f'Average Acc Known:    {aa_known:.4f}\n')
        result_file.write(f'Average Acc Unknown:  {aa_unknown:.4f}\n')
        result_file.write(f'Average F1 Score:     {aa_f1:.4f}\n')
        result_file.write(f'Average Incremental Acc: {aa_inc:.4f}\n')
        result_file.write(f'Average all Acc: {aa_all:.4f}\n')

        pd_known = round(session_ka_means[0] - session_ka_means[3], 4)
        pd_unknown = round(session_uka_means[0] - session_uka_means[3], 4)
        pd_f1 = round(session_f1s_means[0] - session_f1s_means[3], 4)
        pd_inc = round(session_inc_means[0] - session_inc_means[3], 4)
        pd_all = round(session_all_means[0] - session_all_means[3], 4)
        pd_known_pct = round(pd_known * 100, 2)
        pd_unknown_pct = round(pd_unknown * 100, 2)
        pd_f1_pct = round(pd_f1 * 100, 2)
        pd_inc_pct = round(pd_inc * 100, 2)
        pd_all_pct = round(pd_all * 100, 2)

        result_file.write('\n=== Performance Degradation (PD: Session1 - Session4) ===\n')
        result_file.write(f'PD Acc Known:    {pd_known:.4f} (↓{pd_known_pct}%)\n')
        result_file.write(f'PD Acc Unknown:  {pd_unknown:.4f} (↓{pd_unknown_pct}%)\n')
        result_file.write(f'PD F1 Score:     {pd_f1:.4f} (↓{pd_f1_pct}%)\n')
        result_file.write(f'PD Incremental Acc: {pd_inc:.4f} (↓{pd_inc_pct}%)\n')
        result_file.write(f'PD all Acc: {pd_all:.4f} (↓{pd_all_pct}%)\n')

    return result_path


def main():
    # Use the same args/config as train.py for comparability
    import train as tr

    parser = argparse.ArgumentParser('cluster', parents=[tr.args_parser()])
    parser.add_argument('--smoke', action='store_true')
    cli_args = parser.parse_args()

    if cli_args.smoke:
        smoke_run()
        return

    # load yaml config like train.py
    with open(cli_args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = cfg.get('train', {})
    cfg.update(vars(cli_args))
    args = tr.dict2namespace(cfg)
    tr.set_seed(args.seed if hasattr(args, 'seed') else 42)
    args.cuda = torch.cuda.is_available()

    # prepare datasets and dataloaders using project's helpers
    tr.set_up_datasets(args)
    trainset, trainloader = tr.get_pretrain_dataloader(args)

    # build model and optimizer using project's network
    from network import MYNET, get_optimizer
    model = MYNET(args, mode='encoder')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer, scheduler = get_optimizer(model, args)

    # training loop following standard epochs setting
    total_epochs = args.epochs.epochs_std if hasattr(args, 'epochs') and hasattr(args.epochs, 'epochs_std') else 1
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_prefix = getattr(args, 'save_prefix', f'vaze_{timestamp}_')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_result, exist_ok=True)

    for epoch in range(1, total_epochs + 1):
        print(f'[VAZE] Epoch {epoch}/{total_epochs}')
        model = train_closed_classifier(args, model, trainloader, optimizer, scheduler, epochs=1)

        # save checkpoint with distinct name
        save_name = os.path.join(args.save_dir, f'{save_prefix}epoch_{epoch}.pth')
        torch.save({'epoch': epoch, 'params': model.state_dict()}, save_name)
        # also write a small result metadata file
        result_name = os.path.join(args.save_result, f'{save_prefix}result_epoch_{epoch}.txt')
        with open(result_name, 'w') as f:
            f.write(f'epoch={epoch}\ncheckpoint={save_name}\n')

    # run the same-style incremental evaluation and write comparable test_result file
    result_path = run_incremental_eval_like_baseline(args, model, save_prefix)
    print('[VAZE] Comparable result file saved to:', result_path)

    print('[VAZE] Training finished. Models/results saved with prefix:', save_prefix)


if __name__ == '__main__':
    main()
