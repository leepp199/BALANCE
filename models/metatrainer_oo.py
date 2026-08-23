from __future__ import print_function

import os
import numpy as np
import time
from tqdm import tqdm


import torch
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics
from utils.utils import  AverageMeter
from models.FSEval import run_test_fsl


def meta_train(args, model,train_loader, eval_loader=True):
    params = torch.load(args.pretrained_model_path)['params']
    cls_params = {k: v for k, v in params.items() if 'fc' in k}
    feat_params = {k: v for k, v in params.items() if 'encoder' in k}
    model.cls_classifier.init_representation(cls_params)
    ##### Load Pretrained Weights for Feature Extractor
    model_dict = model.state_dict()
    model_dict.update(feat_params)
    model.load_state_dict(model_dict)
    model.train()
    optim_param = [{'params': model.cls_classifier.parameters()}]
    if getattr(args, 'finetune_encoder', False):
        scale = float(getattr(args, 'encoder_lr_scale', 0.01))
        enc_lr = args.learning_rate * scale
        # 确定解封层：默认 layer4，若指定 finetune_layers 则按列表
        ft_layers = getattr(args, 'finetune_layers', 'layer4')
        if isinstance(ft_layers, str):
            ft_layers = [s.strip() for s in ft_layers.split(',')]
        # 关闭所有 encoder 梯度，再按需开启
        for p in model.encoder.parameters():
            p.requires_grad = False
        ft_params = []
        for layer_name in ft_layers:
            layer = getattr(model.encoder, layer_name, None)
            if layer is not None:
                for p in layer.parameters():
                    p.requires_grad = True
                ft_params.extend(layer.parameters())
        optim_param.append({'params': ft_params, 'lr': enc_lr})
        print(f"==> [v2] finetune_encoder=True, layers={ft_layers}, lr={enc_lr:.6f} (base lr={args.learning_rate}, scale={scale})")
        # Path Y: snapshot init params of the fine-tuned layers for anchor loss
        base_anchor_w = float(getattr(args, 'base_anchor_weight', 0.0))
        if base_anchor_w > 0.0:
            anchor_init = [p.detach().clone() for p in ft_params]
            print(f"==> [Y] base_anchor_weight={base_anchor_w:.4f}, anchoring {len(ft_params)} tensors in layers={ft_layers}")
        else:
            anchor_init = None
    else:
        # 保持默认：encoder 全部冻结梯度
        for p in model.encoder.parameters():
            p.requires_grad = False
        ft_params = []
        anchor_init = None
    optimizer = optim.SGD(optim_param, lr=args.learning_rate, momentum=args.optimizer.momentum, weight_decay=args.optimizer.decay, nesterov=True)
    if args.cosine:
        print("==> training with plateau scheduler ...")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    else:
        print("==> training with MultiStep scheduler ... gamma {} step {}".format(args.lr_decay_rate, args.lr_decay_epochs))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['maxmeta_acc'] = 0.0
    trlog['maxmeta_acc_epoch'] = 0
    trlog['maxmeta_auroc'] = 0.0
    trlog['maxmeta_auroc_epoch'] = 0

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    for epoch in range(1, args.epochs.epochs_meta + 1):
        if args.cosine:
            scheduler.step(trlog['maxmeta_acc'])
        else:
            adjust_learning_rate(epoch, args, optimizer, 0.0001)
            
        train_acc, train_auroc, train_loss, train_msg = train_episode(epoch, train_loader, model, optimizer, args, ft_params=ft_params, anchor_init=anchor_init)

        model.eval()

        #evaluate
        if eval_loader is not None:
            start = time.time()
            config = {'auroc_type':['prob']}
            result = run_test_fsl(model, eval_loader, config)
            meta_test_acc = result['data']['acc']
            open_score_auroc = result['data']['auroc_prob']

            test_time = time.time() - start
            meta_msg = 'Meta Test Acc: {:.4f}, Test std: {:.4f}, AUROC: {:.4f}, Time: {:.1f}'.format(meta_test_acc[0], meta_test_acc[1], open_score_auroc[0], test_time)
            train_msg = train_msg + ' | ' + meta_msg
                
            if trlog['maxmeta_acc'] < train_acc:#meta_test_acc[0]:
                trlog['maxmeta_acc'] = train_acc#meta_test_acc[0]
                trlog['maxmeta_acc_epoch'] = epoch
                acc_auroc = (train_acc,train_auroc)#(meta_test_acc[0], open_score_auroc[0])
                save_model(epoch, 'max_acc', acc_auroc)
            if trlog['maxmeta_auroc'] < train_auroc:#open_score_auroc[0]:
                trlog['maxmeta_auroc'] = train_auroc#open_score_auroc[0]
                trlog['maxmeta_auroc_epoch'] = epoch
                acc_auroc = (train_acc,train_auroc)#(meta_test_acc[0], open_score_auroc[0])
            save_model(epoch, 'max_auroc', acc_auroc)
                
        print(train_msg)
        # # print(meta_test_acc[0])
        # print(trlog['maxmeta_acc'],trlog['maxmeta_acc_epoch'])

        # Save every epoch so a long Meta run can be resumed after an I/O or
        # host failure.  The previous 5-epoch cadence could lose hours of
        # training before the first usable Meta checkpoint was flushed.
        save_model(model, epoch, args, name='epoch_{:03d}'.format(epoch),
                   acc_auroc=(train_acc, train_auroc))
        if epoch % 5 == 0:
            print('The Best Meta Acc {:.4f} in Epoch {}, Best Meta AUROC {:.4f} in Epoch {}'.format(trlog['maxmeta_acc'],trlog['maxmeta_acc_epoch'],trlog['maxmeta_auroc'],trlog['maxmeta_auroc_epoch']))


def train_episode(epoch, train_loader, model, optimizer, args, ft_params=None, anchor_init=None):
    """One epoch training"""
    model.train()
    model.encoder.eval()
    if getattr(args, 'finetune_encoder', False):
        # 解封指定层的 BN running stats 更新
        ft_layers = getattr(args, 'finetune_layers', 'layer4')
        if isinstance(ft_layers, str):
            ft_layers = [s.strip() for s in ft_layers.split(',')]
        for layer_name in ft_layers:
            layer = getattr(model.encoder, layer_name, None)
            if layer is not None:
                layer.train()


    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_funit = AverageMeter()
    acc = AverageMeter()
    auroc = AverageMeter()
    end = time.time()

    with tqdm(train_loader, total=len(train_loader), leave=False) as pbar:
        for idx, data in enumerate(pbar):
            support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label, supp_idx, open_idx,base_ids= data
            # Data Conversion & Packaging

            supp_idx, open_idx,base_ids = supp_idx.long(), open_idx.long(),base_ids.long()
            openset_label = args.n_ways * torch.ones_like(openset_label)
            # print(support_data.shape)  
            # print(query_data.shape)  
            # print(suppopen_data.shape)  
            # print(openset_data.shape)
            # the_img = torch.cat([support_data, query_data, suppopen_data, openset_data], dim=1)'
            # print(type(support_data))
            # print(type(query_data))
            # print(type(suppopen_data))
            # print(type(openset_data))
            # support_data=[support_data]
            # query_data=[query_data]
            # suppopen_data=[suppopen_data]
            # openset_data=[openset_data]
            # the_img     = support_data+query_data+suppopen_data+openset_data
            # 在新维度（如 0 维）上堆叠  
            # support_data=torch.squeeze(support_data,0)
            # query_data=torch.squeeze(query_data,0)
            # suppopen_data=torch.squeeze(suppopen_data,0)
            # # openset_data=torch.squeeze(openset_data,0)
            # print(support_data.shape)  
            # print(query_data.shape)  
            # print(suppopen_data.shape)  
            # print(openset_data.shape)
            # the_img     = support_data+query_data+suppopen_data+openset_data #NS
            #LS FS训练时
            the_img = torch.cat((support_data, query_data, suppopen_data, openset_data), dim=1)  
            # 这样可以保持各自的样本数量，并用相同的特征长度
            the_label   = (support_label,query_label,suppopen_label,openset_label)
            the_conj    = (supp_idx, open_idx)
            model.mode = 'openmeta'
            _, _, probs, loss = model(the_img,the_label,the_conj,base_ids)
            query_cls_probs, openset_cls_probs = probs
            (loss_cls, loss_open_hinge, loss_funit) = loss
            loss_open = args.gamma * loss_open_hinge + args.funit * loss_funit

            loss = loss_open + loss_cls
            # Path Y: base anchor loss -- pull fine-tuned encoder layers back toward init to prevent drift
            anchor_w = float(getattr(args, 'base_anchor_weight', 0.0))
            if anchor_w > 0.0 and ft_params is not None and anchor_init is not None:
                loss_anchor = 0.0
                for p_cur, p_init in zip(ft_params, anchor_init):
                    loss_anchor = loss_anchor + (p_cur - p_init).pow(2).sum()
                loss = loss + anchor_w * loss_anchor
            n = args.n_ways
            
            # 针对 Query Data (已知类)
            q_pos = query_cls_probs[:, :, :n] # [B, N_q, 5]
            q_neg = query_cls_probs[:, :, n:] # [B, N_q, 5]
            
            # 针对 Openset Data (未知类)
            o_pos = openset_cls_probs[:, :, :n] # [B, N_o, 5]
            o_neg = openset_cls_probs[:, :, n:] # [B, N_o, 5]
            
            # 2. 计算 Closed Set Accuracy (只看正类分数)
            # 展平 Batch 和 Query 维度
            # q_pos.reshape(-1, n) -> [75, 5]
            pos_scores_flat = q_pos.reshape(-1, n).detach().cpu().numpy()
            
            close_pred = np.argmax(pos_scores_flat, axis=-1) # [75]
            close_label = query_label.view(-1).cpu().numpy() # [75]
            
            acc.update(metrics.accuracy_score(close_label, close_pred), 1)

            # 3. 计算 Open Set AUROC (基于 1对1 判决)
            # Unknown Score = Max(Neg - Pos)
            # 逻辑：如果最可能的类别的 负分 > 正分，则是未知类
            
            def get_uncertainty_score(p_scores, n_scores):
                # p_scores, n_scores: [B, N, 5]
                # 找到每个样本预测类别的索引
                preds = torch.argmax(p_scores, dim=-1) # [B, N]
                
                # 取出预测类别对应的正分和负分
                # gather: [B, N, 1]
                p_val = torch.gather(p_scores, -1, preds.unsqueeze(-1)).squeeze(-1)
                n_val = torch.gather(n_scores, -1, preds.unsqueeze(-1)).squeeze(-1)
                
                # 分数越高越可能是未知类
                # 原始逻辑：Positive > Negative -> Known
                # 逆转逻辑：Negative - Positive > 0 -> Unknown
                return (n_val - p_val).view(-1).detach().cpu().numpy()

            score_known = get_uncertainty_score(q_pos, q_neg) # 期望很小 (负 < 正)
            score_unknown = get_uncertainty_score(o_pos, o_neg) # 期望很大 (负 > 正)
            
            # 拼接
            scores_all = np.concatenate([score_known, score_unknown])
            # 标签：已知类=0，未知类=1
            labels_all = np.concatenate([np.zeros(len(score_known)), np.ones(len(score_unknown))])
            
            if len(np.unique(labels_all)) > 1:
                auroc.update(metrics.roc_auc_score(labels_all, scores_all), 1)
                
            losses_cls.update(loss_cls.item(), 1)
            losses_funit.update(loss_funit.item(), 1)

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()
                
                
            pbar.set_postfix({"Acc":'{0:.2f}'.format(acc.avg), 
                                "Auroc":'{0:.2f}'.format(auroc.avg), 
                                "cls_ce" :'{0:.2f}'.format(losses_cls.avg), 
                                "funit" :'{0:.4f}'.format(losses_funit.avg), 
                                })

    message = 'Epoch {} Train_Acc {acc.avg:.3f} Train_Auroc {auroc.avg:.3f}'.format(epoch, acc=acc, auroc=auroc)

    return acc.avg, auroc.avg, (losses_cls.avg, losses_funit.avg), message

def save_model(model,epoch, args,name=None, acc_auroc=None):
    state = {
        'epoch': epoch,
        'cls_params': model.state_dict() ,
        'acc_auroc': acc_auroc
    }
    # 'optimizer': self.optimizer.state_dict()['param_groups'],
                 
    file_name = 'epoch_'+str(epoch)+'.pth' if name is None else name + '.pth'
    print('==> Saving', file_name)
    torch.save(state, args.save_dir+file_name)


    
    
def adjust_learning_rate(epoch, opt, optimizer, threshold=1e-6):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0 and opt.learning_rate > threshold:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr          
