import os
    # 环境变量
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import yaml
import torch
from sklearn.preprocessing import RobustScaler
import torch.nn as nn  
from utils.util import cluster_acc,calc
from utils.utils import *
from network import MYNET,get_optimizer,replace_base_fc
from data.dataloader import *
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from tqdm import tqdm
from openmax import *
from models.metatrainer import meta_train
# from models.metaowtrainer import meta_train
from threshold_free import run_test_fsl
from models.AttnClassifier import Classifier
from utils.streamCluster import FStream
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 用于3D可视化
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances  # 
from sklearn.cluster import DBSCAN  # DBSCAN
from matplotlib import rcParams
from sklearn.metrics.pairwise import cosine_similarity 
from enhance_module import LocalFeatureCluster
import math
def set_mcd_mode(model):
    """
    开启 MC Dropout 模式：
    保持 BatchNorm 为 eval 模式（稳定统计量），但强制开启 Dropout（引入随机性）。
    """
    model.eval() # 全局设为 eval
    
    # 遍历所有子模块，单独激活 Dropout
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
def calculate_uncertainty_unlabeled(model, enhancer, sample, n_aug=5, n_forward=5):
    """
    计算无标签样本的不确定度 (基于特征掩码 + MC Dropout)
    """
    # 1. 开启 MC Dropout 模式 (Dropout 生效)
    set_mcd_mode(model)
    
    features_list = []
    device = next(model.parameters()).device
    
    if sample.dim() == 1:
        sample = sample.unsqueeze(0)
    sample = sample.to(device)

    with torch.no_grad():
        # 外层循环：不同的 Mask (通过 augment=True 触发)
        for _ in range(n_aug):
            # 内层循环：不同的 Dropout (通过 MC Dropout 触发)
            for _ in range(n_forward):
                
                # 【关键】调用时开启 augment=True
                # 这会触发 Log Mel 谱图上的随机时间/频率遮挡
                feat = model.hgnn_encode(sample, augment=True) 
                
                # 通过增强模块
                feat, _ = enhancer(feat) 
                
                if feat.dim() > 2:
                    feat = feat.mean(dim=[2,3]) if feat.dim()==4 else feat.mean(dim=1)
                
                features_list.append(feat.squeeze())
    
    P = torch.stack(features_list)
    
    # 计算核范数
    uncertainty = torch.norm(P, p='nuc').item()
    
    return uncertainty

def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    import os
    
    # 基础种子设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 强制确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_randomness():
    """验证随机种子是否生效"""
    print("\n=== Randomness Check ===")
    print(f"Python random: {random.randint(0, 100)}")
    print(f"Numpy random: {np.random.randint(0, 100)}")
    print(f"PyTorch random: {torch.rand(1).item()}")
    print("="*30)
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(args.seed)  # 为初始化过程设种子
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def dict2namespace(dicts):
    for i in dicts:
        if isinstance(dicts[i], dict):
            dicts[i] = dict2namespace(dicts[i]) 
    ns = argparse.Namespace(**dicts)
    return ns


def set_up_datasets(args):
    if args.dataset == 'FMC':
        import data.FMC as Dataset
    elif args.dataset == 'nsynth-100':
        import data.nsynth as Dataset
    elif args.dataset == 'nsynth-200':
        import data.nsynth as Dataset
    elif args.dataset == 'nsynth-300':
        import data.nsynth as Dataset
    elif args.dataset == 'nsynth-400':
        import data.nsynth as Dataset
    elif args.dataset == 'librispeech':
        import data.librispeech as Dataset
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        import data.s2s as Dataset
    args.Dataset=Dataset

def args_parser():
    parser = argparse.ArgumentParser(description='cluster', add_help=False)
    parser.add_argument('-config', type=str, default="/data/lqq/baseline/configs/default.yml") 
    parser.add_argument('-dist_path', type=str, default="/data/lqq/baseline/save/dist.mat") 
    parser.add_argument('-dataset', type=str, default='librispeech',
                        choices=['FMC', 'nsynth-100', 'nsynth-200', 'nsynth-300', 'nsynth-400', 'librispeech',
                        'f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n'])
    # parser.add_argument('--dataroot', type=str,default="/data/datasets/The_NSynth_Dataset/")
    # parser.add_argument('--dataroot', type=str,default="/data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data")
    
    parser.add_argument('--dataroot', type=str,default="/data/datasets/librispeech_fscil/")
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--save_result',type = str,default='/data/lqq/baseline/save_result/')
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    parser.add_argument('--num_labeled_classes', default=80, type=int)
    parser.add_argument('--checkpoint', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cosine', type=bool,default=True, help='using cosine annealing')
    parser.add_argument('--pretrained_model_path', type=str, default="/data/lqq/baseline/save/base_train_for_meta.pth")
    parser.add_argument('--train_weight_base', type=int, default=1, help='enable training base class weights')
    parser.add_argument('--base_seman_calib',type=int, default=1, help='base semantics calibration')
    parser.add_argument('--neg_gen_type', type=str, default='att', choices=['semang', 'attg', 'att', 'mlp'])
    parser.add_argument('--agg', type=str, default='avg', choices=['avg', 'mlp'])
    parser.add_argument('--gamma', type=float, default=1.0, help='loss cofficient for mse loss')
    parser.add_argument('--funit', type=float, default=1.0)
    parser.add_argument('--outer_lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--inner_steps', default=10, type=int) 
    parser.add_argument('--outer_steps', default=5, type=int)
    parser.add_argument('--debug', default=True, type=bool)
    # 在args_parser()中添加以下参数
    parser.add_argument('--pit_weight', type=float, default=0.5, help='weight for pseudo-incremental loss')
    parser.add_argument('--pit_num_new_classes', type=int, default=5, help='number of pseudo new classes')
    parser.add_argument('--pit_base_momentum', type=float, default=0.7, help='momentum for base class weight update')
    parser.add_argument('--pit_mixup_alpha', type=float, default=0.5, help='alpha for mixup augmentation')
    # parser.add_argument('--cluster_threshold', type=float, default=0.7, 
    #                   help='Initial threshold for dynamic clustering')
    # parser.add_argument('--threshold_decay', type=float, default=0.95,
    #                   help='Decay rate for cluster threshold')
    # parser.add_argument('--proto_momentum', type=float, default=0.3,
    #                   help='动量系数用于原型更新')
    # parser.add_argument('--debug', action='store_true', 
    #                   help='Enable debug mode with visualizations')
    return parser

def update_fc_avg(args,model,dataloader,x,label,class_list):
    new_fc=[]
    for batch in dataloader:
        x, label,_ = [_.cuda() for _ in batch]
        data=model(x).detach()
    for class_index in class_list:
        print(class_index)
        data_index=(label==class_index).nonzero().squeeze(-1)
        embedding=data[data_index]
        proto=embedding.mean(0)
        new_fc.append(proto)
        if class_index>=args.num_labeled_classes:   #要计算更新这个数
            model.fc.weight.data[class_index]=proto
        else:
            model.fc.weight.data[class_index]=(proto+model.fc.weight.data[class_index]).mean(0)
        #print(proto)
import time  # 需导入时间模块
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances

def debug_cluster(args, model, data, labels, session=None):
    """改进的特征聚类函数（带时序约束）"""
    with torch.no_grad():
        features = torch.stack([model.hgnn_encode(x).squeeze() for x in data])  # [N,512,H,W]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features,_ = LocalFeatureCluster(k_ratio=0.4)(features)
        features = features.to(device)
        kmeans = KMeans(n_clusters=args.num_unlabeled_classes, n_init=20).fit(features.cpu().numpy())
        
        # 原型更新
        y = kmeans.labels_
        acc, map = cluster_acc(args, np.array(labels), y)
        
        updated = 0
        for cluster_id in np.unique(y):
            if cluster_id in map:
                true_label = map[cluster_id]
                if true_label >= args.num_labeled_classes:
                    indices = np.where(y == cluster_id)[0]
                    if len(indices) > 0:
                        new_proto =features[indices].mean(dim=0).to('cuda')  # 使用压缩后的特征
                        model.fc.weight.data[true_label] = new_proto
                        updated += 1
    
    return acc

def test(args, model, testloader,  session):    
    test_class = args.num_base + session * args.way
    model = model.eval()
    num_batch=0
    va=0.0
    sup_emb, novel_ids = None, None
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            model.mode = 'incre'
            query = model.encode(data)
            # query,_ = LocalFeatureCluster(k_ratio=0.3)(query)
            # print(f"Original query shape: {query.shape}")
            proto = model.fc.weight[:test_class, :].detach()
            logits=F.cosine_similarity(query.unsqueeze(1), proto, dim=-1)
            acc = count_acc(logits, test_label)
            num_batch+=1
            va+=acc
    return float(va/num_batch)

#baseline
def known_test(model,data,label):
    feats=[]
    label = torch.tensor(label)
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(len(data)):
        feat = model.hgnn_encode(data[i])
        feat,_ = LocalFeatureCluster(k_ratio=0.4)(feat)
        feat = feat.to(device)
        feats.append(feat)
    proto = model.fc.weight[:args.num_labeled_classes,:].detach().unsqueeze(0).unsqueeze(0)
    feats = torch.stack(feats)
    logits=F.cosine_similarity(feats, proto, dim=-1)
    logits=torch.squeeze(logits)
    acc = count_acc(logits, label.to('cuda'))
    preds = torch.argmax(logits, dim=1)
    score = f1_score(label.cpu().numpy(),preds.cpu().numpy(),average='macro')
    return acc,score

def train(args: dict):   
    # ============ base session training ==============
    device = torch.device("cuda" if args.cuda else "cpu")
    model = MYNET(args, mode='encoder')
    model = model.to(device)
    model.apply(weights_init)  # 使用固定种子的初始化
    set_up_datasets(args)
    if args.checkpoint:
        best_model_dir = args.save_dir+'/'+'epoch_5.pth'
        #meta-train negative prototype
        params = torch.load(best_model_dir, weights_only=True)['cls_params']
        cls_params = {k: v for k, v in params.items() if 'fc' in k}
        model.cls_classifier.init_representation(cls_params)
        model_dict = model.state_dict()
        model_dict.update(params)
        model.load_state_dict(model_dict)
    else:
        best_model_dir=base_train(args,model)
        # best_model_dir=os.path.join(args.save_dir, f'base_train_for_meta.pth')#
        # state_dict = torch.load(best_model_dir)
        # model.load_state_dict(state_dict['params'], strict=True)
        open_train_val_loader= get_dataloaders(args,'openmeta')
        meta_train(args, model,open_train_val_loader, eval_loader=None)
    data_dict,result={},{}
    data_dict['train_set'],_=get_pretrain_dataloader(args)
    model = replace_base_fc(args,data_dict['train_set'], model) 
    with open(os.path.join(args.save_result,'test_result.txt'),'w')as result_file:
        session0_acc_list = []
        session_ka = [[] for _ in range(args.test_times)]
        session_uka = [[] for _ in range(args.test_times)]
        session_f1s = [[] for _ in range(args.test_times)]
        session_inc = [[] for _ in range(args.test_times)]
        session_all = [[] for _ in range(args.test_times)]
        for j in range(0, args.num_session): 
            result['sess{}_ak'.format(j)]=[]
            result['sess{}_au'.format(j)]=[]
            result['sess{}_fs'.format(j)]=[]
            result['sess{}_inc'.format(j)]=[]
            result['sess{}_all'.format(j)]=[]
        for i in range(args.test_times):
            args.current_test = i  # 记录当前测试轮次
            args.num_labeled_classes = args.num_base
            print(f"\n=== Base Session Pure Evaluation (Round {i}) ===")
            _, base_testloader = get_testloader(args, 0)  
            base_acc = test(args, model, base_testloader, 0)  
            session0_acc_list.append(base_acc)
            # 记录结果（未知类指标设为0）
            result['sess0_ak'].append(base_acc)
            result['sess0_au'].append(0.0)
            result['sess0_fs'].append(0.0)
            result['sess0_inc'].append(0.0)
            result['sess0_all'].append(base_acc)
            # 打印session 0结果
            print(f"Session 0: acc known: {base_acc:.4f}, acc unknown: 0.0000, "
                  f"f1 score: 0.0000, inc acc:0.0000, all acc: {base_acc:.4f}")
            for session in range(args.start_session, args.num_session):  
                print("Inference session: [%d]" % session)
                print(f"test_time: {i}")
                model.mode = args.network.new_mode
                model.eval()
                _,unlabelled_loader = get_dataloader(args, session) #已知5类+未知5类
                #OSR_DETECTION
                unknow_data,unknow_label,know_data,know_label=run_test_fsl(model,args,unlabelled_loader)
                #K means
                cluster_acc=debug_cluster(args,model,unknow_data,unknow_label,session)
                acc_known,_ = known_test(model,know_data,know_label)
                fscore=calc(args,know_label,unknow_label)
                result['sess{}_ak'.format(session)]+=[acc_known]
                result['sess{}_au'.format(session)]+=[cluster_acc]
                result['sess{}_fs'.format(session)]+=[fscore]
                #incremental learning
                _,testloader = get_testloader(args,session)
                all_acc=test(args, model, testloader,  session)
                _,inc_testloader = get_inc_testloader(args,session)
                inc_acc = test(args, model, inc_testloader,  session)
                result['sess{}_inc'.format(session)]+=[inc_acc]
                result['sess{}_all'.format(session)]+=[all_acc]
                args.num_labeled_classes += args.way
                avg_acc_known = sum(result['sess{}_ak'.format(session)]) / len(result['sess{}_ak'.format(session)])  
                avg_acc_unknown = sum(result['sess{}_au'.format(session)]) / len(result['sess{}_au'.format(session)])  
                avg_fscore = sum(result['sess{}_fs'.format(session)]) / len(result['sess{}_fs'.format(session)])  
                avg_inc_acc = sum(result['sess{}_inc'.format(session)]) / len(result['sess{}_inc'.format(session)])  
                avg_all_acc = sum(result['sess{}_all'.format(session)]) / len(result['sess{}_all'.format(session)])  
                session_ka[i].append(avg_acc_known)
                session_uka[i].append(avg_acc_unknown)
                session_f1s[i].append(avg_fscore)
                session_inc[i].append(avg_inc_acc)
                session_all[i].append(avg_all_acc)
                # avg_session0_acc = sum(session0_acc_list) / len(session0_acc_list)
                # print(f"\n=== Final Average Session 0 Acc: {avg_session0_acc:.4f} ===")
                # result_file.write(f"\nAverage Session 0 Acc: {avg_session0_acc:.4f}\n")
                # 写入文件  
                result_line = "session: {}, aac known: {:.4f}, acc unknown: {:.4f}, f1 score: {:.4f}, inc acc: {:.4f}, all acc: {:.4f}\n".format(  
                    session, avg_acc_known, avg_acc_unknown, avg_fscore, avg_inc_acc,avg_all_acc)  
                result_file.write(result_line)  
                print("session:{},acc known:{:.4f},acc unknown:{:.4f},f1 score:{:.4f},incremental acc:{:.4f},all acc:{:.4f}".format(session,(sum(result['sess{}_ak'.format(session)])/len(result['sess{}_ak'.format(session)])), 
           (sum(result['sess{}_au'.format(session)])/len(result['sess{}_au'.format(session)])),(sum(result['sess{}_fs'.format(session)])/len(result['sess{}_fs'.format(session)])), sum(result['sess{}_inc'.format(session)])/len(result['sess{}_inc'.format(session)]),(sum(result['sess{}_all'.format(session)]) / len(result['sess{}_all'.format(session)]) )))
            best_model_dir = os.path.join(args.save_dir, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=model.state_dict()), best_model_dir)
        session0_acc_values = np.array(session0_acc_list)
        session0_mean = np.mean(session0_acc_values)
        session0_std = np.std(session0_acc_values)
        
        print(f"\n=== Final Session 0 ===")
        print(f"Average Acc: {session0_mean:.4f} ± {session0_std:.4f}")
        result_file.write(f"\n=== Final Session 0 ===\n")
        result_file.write(f"Average Acc: {session0_mean:.4f} ± {session0_std:.4f}\n")
        session_ka_means = []  # 存储每个session的 known_acc 均值
        session_uka_means = [] # 存储每个session的 unknown_acc 均值
        session_f1s_means = [] # 存储每个session的 f1_score 均值
        session_inc_means = [] # 存储每个session的 incremental_acc 均值
        session_all_means = [] # 存储每个session的 incremental_acc 均值
        for ses in range(args.num_session-1):
            # 计算均值和标准差
            ka_values = [session_ka[time][ses] for time in range(args.test_times)]
            uka_values = [session_uka[time][ses] for time in range(args.test_times)]
            f1s_values = [session_f1s[time][ses] for time in range(args.test_times)]
            inc_values = [session_inc[time][ses] for time in range(args.test_times)]
            all_values = [session_all[time][ses] for time in range(args.test_times)]
            ka_mean = np.mean(ka_values)
            ka_std = np.std(ka_values)
            uka_mean = np.mean(uka_values)
            uka_std = np.std(uka_values)
            f1s_mean = np.mean(f1s_values)
            f1s_std = np.std(f1s_values)
            inc_mean = np.mean(inc_values)
            inc_std = np.std(inc_values)
            all_mean = np.mean(all_values)
            all_std = np.std(all_values)
            session_ka_means.append(round(ka_mean,4))
            session_uka_means.append(round(uka_mean,4))
            session_f1s_means.append(round(f1s_mean,4))
            session_inc_means.append(round(inc_mean,4))
            session_all_means.append(round(all_mean,4))
            # 打印带标准差的结果（保持原有打印格式）
            print(f"total session{ses+1} acc known is {ka_mean:.4f} ± {ka_std:.4f}")
            print(f"total session{ses+1} acc unknown is {uka_mean:.4f} ± {uka_std:.4f}")
            print(f"total session{ses+1} f1 score is {f1s_mean:.4f} ± {f1s_std:.4f}")
            print(f"total session{ses+1} incremental acc is {inc_mean:.4f} ± {inc_std:.4f}")
            print(f"total session{ses+1} all acc is {all_mean:.4f} ± {all_std:.4f}")
            # 写入文件（保持原有格式）
            result_row = (
                f"session: {ses+1}, "
                f"total aac known: {ka_mean:.4f} ± {ka_std:.4f}, "
                f"total acc unknown: {uka_mean:.4f} ± {uka_std:.4f}, "
                f"total f1 score: {f1s_mean:.4f} ± {f1s_std:.4f}, "
                f"total incremental acc: {inc_mean:.4f} ± {inc_std:.4f}, "
                f"total all acc: {all_mean:.4f} ± {all_std:.4f}\n"
            )
            result_file.write(result_row)  
        aa_known = round(np.mean(session_ka_means), 4)    # known_acc的平均准确率
        aa_unknown = round(np.mean(session_uka_means), 4) # unknown_acc的平均准确率
        aa_f1 = round(np.mean(session_f1s_means), 4)      # f1_score的平均准确率
        aa_inc = round(np.mean(session_inc_means), 4)     # incremental_acc的平均准确率    
        aa_all = round(np.mean(session_all_means), 4)
        print("\n=== 4 Sessions Average Accuracy (AA) ===")
        print(f"Average Acc Known:    {aa_known:.4f}")
        print(f"Average Acc Unknown:  {aa_unknown:.4f}")
        print(f"Average F1 Score:     {aa_f1:.4f}")
        print(f"Average Incremental Acc: {aa_inc:.4f}")
        print(f"Average all Acc: {aa_all:.4f}")
        result_file.write("\n=== 4 Sessions Average Accuracy (AA) ===\n")
        result_file.write(f"Average Acc Known:    {aa_known:.4f}\n")
        result_file.write(f"Average Acc Unknown:  {aa_unknown:.4f}\n")
        result_file.write(f"Average F1 Score:     {aa_f1:.4f}\n")
        result_file.write(f"Average Incremental Acc: {aa_inc:.4f}\n")
        result_file.write(f"Average all Acc: {aa_all:.4f}\n")
        pd_known = round(session_ka_means[0] - session_ka_means[3], 4)    # known_acc的性能下降
        pd_unknown = round(session_uka_means[0] - session_uka_means[3], 4) # unknown_acc的性能下降
        pd_f1 = round(session_f1s_means[0] - session_f1s_means[3], 4)      # f1_score的性能下降
        pd_inc = round(session_inc_means[0] - session_inc_means[3], 4)     # incremental_acc的性能下降
        pd_all = round(session_all_means[0] - session_all_means[3], 4)     # incremental_acc的性能下降
        # 计算百分比并保留两位小数
        pd_known_pct = round(pd_known * 100, 2)
        pd_unknown_pct = round(pd_unknown * 100, 2)
        pd_f1_pct = round(pd_f1 * 100, 2)
        pd_inc_pct = round(pd_inc * 100, 2)
        pd_all_pct = round(pd_all * 100, 2)
        # 打印性能下降率（PD）
        print("\n=== Performance Degradation (PD: Session1 - Session4) ===")
        print(f"PD Acc Known:    {pd_known:.4f} (↓{pd_known_pct}%)")
        print(f"PD Acc Unknown:  {pd_unknown:.4f} (↓{pd_unknown_pct}%)")
        print(f"PD F1 Score:     {pd_f1:.4f} (↓{pd_f1_pct}%)")
        print(f"PD Incremental Acc: {pd_inc:.4f} (↓{pd_inc_pct}%)")
        print(f"PD all Acc: {pd_all:.4f} (↓{pd_all_pct}%)")
        # 写入文件
        result_file.write("\n=== Performance Degradation (PD: Session1 - Session4) ===\n")
        result_file.write(f"PD Acc Known:    {pd_known:.4f} (↓{pd_known_pct}%)\n")
        result_file.write(f"PD Acc Unknown:  {pd_unknown:.4f} (↓{pd_unknown_pct}%)\n")
        result_file.write(f"PD F1 Score:     {pd_f1:.4f} (↓{pd_f1_pct}%)\n")
        result_file.write(f"PD Incremental Acc: {pd_inc:.4f} (↓{pd_inc_pct}%)\n")
        result_file.write(f"PD all Acc: {pd_all:.4f} (↓{pd_all_pct}%)\n")
        result_file.close()
def get_class_difficulty(args, model, full_loader):
    """
    [创新点：基于类级不确定性的难度评估]
    计算每个基类样本的平均不确定性，用于排序。
    """
    print("Evaluating Base Class Difficulty...")
    model.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # 记录每个类的累积不确定性和样本数
    class_unc_sum = {}
    class_counts = {}
    
    # 临时开启 MC Dropout
    # 如果 model.get_uncertainty 内部已经处理了 eval/train 切换，这里保持 eval 即可
    
    with torch.no_grad():
        # 只跑一部分数据估算即可，不需要全量，节省时间
        for i, batch in enumerate(tqdm(full_loader, desc="Difficulty Est", leave=False)):
            if i > 200: break # 抽样 200 个 batch
            data, label = [_.to(device) for _ in batch]
            
            # 计算 Batch 不确定性
            # 调用 model.get_uncertainty (支持 batch 计算)
            # n_aug=2, n_forward=2 快速估算
            if hasattr(model, 'module'):
                uncs = model.module.get_uncertainty(data, n_aug=5, n_forward=5)
            else:
                uncs = model.get_uncertainty(data, n_aug=5, n_forward=5)
            
            if isinstance(uncs, torch.Tensor): uncs = uncs.cpu().numpy()
            label = label.cpu().numpy()
            
            for l, u in zip(label, uncs):
                if l not in class_unc_sum:
                    class_unc_sum[l] = 0.0
                    class_counts[l] = 0
                class_unc_sum[l] += u
                class_counts[l] += 1
    
    # 计算平均值
    class_avg_unc = []
    # 确保涵盖所有基类 (0 ~ num_base-1)
    for i in range(args.num_base):
        if i in class_unc_sum and class_counts[i] > 0:
            class_avg_unc.append(class_unc_sum[i] / class_counts[i])
        else:
            class_avg_unc.append(0.0) # 默认简单
            
    # 返回按难度排序的类别索引 (Uncertainty 小 -> 大)
    sorted_classes = np.argsort(class_avg_unc)
    return sorted_classes
from models.uncertainty import get_base_class_uncertainty
def get_initial_difficulty(args, model, full_loader):
    """
    【零样本难度评估】
    利用 ResNet 预训练权重的特征分布来确定初始课程顺序。
    原理：类内特征越紧凑(方差小) -> 越简单；越发散 -> 越困难。
    """
    print("\n=== [Curriculum Init] Sorting classes by Feature Compactness (Zero-shot) ===")
    
    model.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    class_vectors = {} 
    
    # 1. 提取特征 (不经过 FC 层)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(full_loader, desc="Difficulty Est")):
            data, label = [_.to(device) for _ in batch]
            
            # 使用 base_encode 提取特征 (注意 augment=False)
            if hasattr(model, 'module'):
                feats = model.module.base_encode(data, augment=False) 
            else:
                feats = model.base_encode(data, augment=False)
            
            # L2 归一化 (关键：因为后续通常基于 Cosine 相似度)
            feats = F.normalize(feats, p=2, dim=1)
            
            for f, l in zip(feats, label):
                l = l.item()
                if l not in class_vectors: class_vectors[l] = []
                class_vectors[l].append(f.cpu())

    # 2. 计算每个类的“紧密度”
    class_scores = {}
    for cls, feats in class_vectors.items():
        if len(feats) < 2: 
            class_scores[cls] = 0.0
            continue
            
        feats_tensor = torch.stack(feats)
        # 计算类中心
        center = feats_tensor.mean(dim=0, keepdim=True)
        center = F.normalize(center, p=2, dim=1)
        
        # 距离 = 1 - CosineSimilarity (越小越简单)
        distances = 1.0 - torch.mm(feats_tensor, center.t())
        class_scores[cls] = distances.mean().item()

    # 3. 排序：简单 -> 困难
    sorted_classes = np.array(sorted(class_scores, key=class_scores.get))
    
    print(f"[Result] Top-5 Easiest: {sorted_classes[:5]}")
    print(f"[Result] Top-5 Hardest: {sorted_classes[-5:]}")
    
    return sorted_classes
from torch.utils.data import WeightedRandomSampler
from models.uncertainty import get_base_class_uncertainty

# =========================================================================
# 辅助函数：获取数据集的标签列表 (用于构建采样权重)
# =========================================================================
def get_dataset_labels(dataset):
    # 尝试常见的属性名
    if hasattr(dataset, 'targets'): return np.array(dataset.targets)
    if hasattr(dataset, 'labels'): return np.array(dataset.labels)
    if hasattr(dataset, '_labels'): return np.array(dataset._labels)
    
    # 如果都没有，只能遍历一遍 (稍微花点时间，但为了加权是值得的)
    print("Extracting labels for weighted sampling...")
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
    all_labels = []
    for _, y in loader:
        all_labels.append(y)
    return torch.cat(all_labels).cpu().numpy()

# =========================================================================
# 核心函数：base_train (极速课程 + 困难加权版)
# =========================================================================
def base_train(args, model):
    # 1. 准备数据
    full_dataset, full_loader = get_pretrain_dataloader(args) 
    save_model_path = os.path.join(args.save_dir, 'base_train_for_meta.pth')
    optimizer, scheduler = get_optimizer(model, args)
    
    total_epochs = args.epochs.epochs_std
    
    # -----------------------------------------------------
    # Step 1: 零样本初始排序 (Zero-shot Sort)
    # -----------------------------------------------------
    # 依然保留这个，作为第一阶段的引导
    sorted_classes = get_initial_difficulty(args, model, full_loader)
    
    # -----------------------------------------------------
    # Step 2: 极速课程阶段 (Fast Curriculum) - 前 30% 轮次
    # -----------------------------------------------------
    # 目标：快速让模型学会简单样本，建立特征骨架，不要浪费太多时间
    curriculum_epochs = int(total_epochs * 0.3) 
    # 分两步走：先练 Top 50% 简单，再练 Top 75%
    phases = [
        (0.50, int(curriculum_epochs * 0.5)), 
        (0.75, int(curriculum_epochs * 0.5))
    ]
    
    current_epoch = 0
    print(f"\n=== [Phase 1] Fast Curriculum Guidance ({curriculum_epochs} Epochs) ===")
    
    for ratio, n_epochs in phases:
        num_keep = int(args.num_base * ratio)
        active_classes = sorted_classes[:num_keep]
        active_classes_idx = np.sort(active_classes)
        
        print(f"--> Training Top {int(ratio*100)}% Easiest Classes ({len(active_classes)})")
        
        # 构造简单类的 Loader
        curr_loader = get_subset_dataloader(args, active_classes_idx)
        
        for _ in range(n_epochs):
            model, _ = standard_base_train_with_metrics(
                args, model, curr_loader, optimizer, scheduler, current_epoch
            )
            current_epoch += 1
            if scheduler is not None: scheduler.step()

    # -----------------------------------------------------
    # Step 3: 困难感知全量训练 (Uncertainty-Weighted Full Training)
    # -----------------------------------------------------
    # 剩下的 70% 轮次，我们训练所有类，但是！给困难类更高的权重！
    print(f"\n=== [Phase 2] Uncertainty-Weighted Full Training ({total_epochs - current_epoch} Epochs) ===")
    
    # A. 重新计算不确定度 (这时候模型已经不是小白了，UNCG 结果很准)
    print(">>> Re-evaluating Difficulty with UNCG...")
    unc_scores = get_base_class_uncertainty(
        model, full_loader, 
        device=torch.device("cuda" if args.cuda else "cpu"), 
        k_dropout=5, a_mask=4
    )
    
    # B. 制作采样权重 (Hard Class Reweighting)
    # 策略：不确定度越高 -> 权重越大
    # 归一化不确定度到 [1.0, 3.0] 之间，让困难类比简单类多被采 3 倍
    unc_values = np.array([unc_scores.get(c, 0) for c in range(args.num_base)])
    min_u, max_u = unc_values.min(), unc_values.max()
    # 线性映射到 [1, 2.5] (倍率可以自己调，2.5倍既重视又不至于由于梯度爆炸)
    class_weights = 1.0 + (unc_values - min_u) / (max_u - min_u + 1e-6) * 1.5
    
    print(f"Class Weights Map (Top 5 Hardest): {np.sort(class_weights)[-5:]}")
    
    # C. 为每个样本分配权重
    all_labels = get_dataset_labels(full_dataset)
    # 确保 label 都在范围内
    all_labels = all_labels[all_labels < args.num_base] 
    
    samples_weights = torch.tensor([class_weights[l] for l in all_labels], dtype=torch.float)
    
    # D. 创建加权采样器
    weighted_sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    # E. 构造带 Sampler 的 Full Loader
    # 注意：使用了 sampler 就不能用 shuffle=True
    weighted_loader = torch.utils.data.DataLoader(
        full_dataset, 
        batch_size=args.dataloader.train_batch_size, 
        sampler=weighted_sampler, # <--- 关键！
        num_workers=8, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # F. 全量冲刺训练
    remaining_epochs = total_epochs - current_epoch
    for _ in range(remaining_epochs):
        model, _ = standard_base_train_with_metrics(
            args, model, weighted_loader, optimizer, scheduler, current_epoch
        )
        current_epoch += 1
        if scheduler is not None: scheduler.step()

    torch.save(dict(params=model.state_dict()), save_model_path)
    print(f"Base training finished. Model saved to {save_model_path}")
    return save_model_path

# =========================================================
# 别忘了保留下面的辅助函数 (如果你还没定义的话)
# =========================================================
def get_subset_dataloader(args, active_classes_idx):
    if args.dataset == 'FMC':
        curr_dataset = args.Dataset.FSDCLIPS(root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
    elif 'nsynth' in args.dataset:
        curr_dataset = args.Dataset.NDS(root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
    elif 'librispeech' in args.dataset:
        curr_dataset = args.Dataset.LBRS(root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
    else:
        curr_dataset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
    
    loader = torch.utils.data.DataLoader(
        curr_dataset, 
        batch_size=args.dataloader.train_batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    return loader

def standard_base_train_with_metrics(args, model, trainloader, optimizer, scheduler, epoch):
    tl = Averager()
    ta = Averager()
    model = model.train()
    model.mode = 'encoder'
    
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)
        
        total_loss = loss
        
        if scheduler is not None:
            lrc = scheduler.get_last_lr()[0]
        else:
            lrc = optimizer.param_groups[0]['lr']

        tqdm_gen.set_description(
            'Epoch {}, lr={:.4f}, loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        
        tl.add(total_loss.item())
        ta.add(acc)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    return model, {'acc': ta.item(), 'loss': tl.item()}
# def base_train(args,model):
#     data_dict = {}
#     data_dict['train_set'],data_dict['trainloader']= get_pretrain_dataloader(args) 
#     net_dict = {}
    
#     net_dict['optimizer'], net_dict['scheduler'] = get_optimizer(model, args)
#     save_model_path = os.path.join(args.save_dir, f'base_train_for_meta.pth')
#     #encoder pretrain
    
#     for epoch in range(args.epochs.epochs_std):
#         model=standard_base_train(args, model,data_dict['trainloader'],net_dict['optimizer'], net_dict['scheduler'], epoch) #要不要打印，改对了吗
#         net_dict['epoch'] = epoch
#         net_dict['scheduler'].step()
#     torch.save(dict(params=model.state_dict()), save_model_path)
   
#     return save_model_path
# 别忘了在文件开头导入你写好的 uncertainty 模块

# def standard_base_train(args, model, trainloader, optimizer, scheduler, epoch):
#     num_base = args.num_base
#     tl = Averager()
#     ta = Averager()
#     model = model.train()
#     model.mode = 'encoder'
#     # standard classification for pretrain
#     tqdm_gen = tqdm(trainloader)
#     for i, batch in enumerate(tqdm_gen, 1):
#         data, train_label = [_.cuda() for _ in batch]

#         logits = model(data)
#         loss = F.cross_entropy(logits, train_label)
#         # feat, proj_feat = model.encode(data, return_proj=True)
#         # contrast_loss = model.compute_contrastive_loss(proj_feat[:len(proj_feat)//2], 
#         #                                              proj_feat[len(proj_feat)//2:])
#         # loss = F.cross_entropy(model.fc(feat), train_label)
#         acc = count_acc(logits, train_label)
#         total_loss = loss
#         # total_loss = loss

#         lrc = scheduler.get_last_lr()[0]
#         tqdm_gen.set_description(
#                 'Standard train, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
#         tl.add(total_loss.item())
#         ta.add(acc)
        
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#     tl = tl.item()
#     ta = ta.item()
#     print('ta:{},tl:{}'.format(ta,tl))
#     return model

def print_version_info(model, message):
    print(message)
    for name, param in model.named_parameters():
        
            print(f"{name}: version {param._version}")

if __name__ == '__main__':
    # parse training arguments
    parser = argparse.ArgumentParser('cluster', parents=[args_parser()])
    args = parser.parse_args()
    with open(args.config) as f:           #training configuration file
        cfg = yaml.safe_load(f)
    cfg = cfg['train']
    cfg.update(vars(args))
    args = dict2namespace(cfg)
    set_seed(args.seed)  
    args.cuda = torch.cuda.is_available()
    check_randomness()
    train(args)
#切换数据集训练时，metatrainer132行和network49行要修改

# def standard_base_train(args, model, trainloader, optimizer, scheduler, epoch):
#     num_base = args.num_base
#     tl = Averager()
#     ta = Averager()
#     model = model.train()
#     model.mode = 'encoder'
#     # standard classification for pretrain
#     tqdm_gen = tqdm(trainloader)
#     for i, batch in enumerate(tqdm_gen, 1):
#         data, train_label = [_.cuda() for _ in batch]

#         logits = model(data)
#         loss = F.cross_entropy(logits, train_label)
#         # feat, proj_feat = model.encode(data, return_proj=True)
#         # contrast_loss = model.compute_contrastive_loss(proj_feat[:len(proj_feat)//2], 
#         #                                              proj_feat[len(proj_feat)//2:])
#         # loss = F.cross_entropy(model.fc(feat), train_label)
#         acc = count_acc(logits, train_label)
#         total_loss = loss
#         # total_loss = loss

#         lrc = scheduler.get_last_lr()[0]
#         tqdm_gen.set_description(
#                 'Standard train, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
#         tl.add(total_loss.item())
#         ta.add(acc)
        
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#     tl = tl.item()
#     ta = ta.item()
#     print('ta:{},tl:{}'.format(ta,tl))
#     return model

def print_version_info(model, message):
    print(message)
    for name, param in model.named_parameters():
        
            print(f"{name}: version {param._version}")

if __name__ == '__main__':
    # parse training arguments
    parser = argparse.ArgumentParser('cluster', parents=[args_parser()])
    args = parser.parse_args()
    with open(args.config) as f:           #training configuration file
        cfg = yaml.safe_load(f)
    cfg = cfg['train']
    cfg.update(vars(args))
    args = dict2namespace(cfg)
    set_seed(args.seed)  
    args.cuda = torch.cuda.is_available()
    check_randomness()
    train(args)
#切换数据集训练时，metatrainer132行和network49行要修改
