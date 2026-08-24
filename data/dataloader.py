import numpy as np
import torch
from .sampler import SupportsetSampler, TrueIncreTrainCategoriesSampler
import random
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 2. 创建生成器
g = torch.Generator()
g.manual_seed(0) # 或者传入 args.seed


def _incremental_episode_seed(args, session):
    """Seed one support/stream episode independently of unrelated RNG calls."""
    return (int(args.seed) +
            1009 * int(getattr(args, 'current_test', 0)) +
            int(session))


def _episode_generators(episode_seed):
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(int(episode_seed))
    worker_generator = torch.Generator()
    # Keep DataLoader worker seed consumption separate from support selection.
    worker_generator.manual_seed(int(episode_seed) + 1_000_003)
    return sampler_generator, worker_generator


def get_test_dataloader(args):
    class_new = np.arange(args.num_base,100)
    testset = args.Dataset.Openlbrs(root=args.dataroot,index=class_new,args=args,partition='test', fix_seed=True)
    meta_test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,num_workers=args.dataloader.num_workers, pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=g)
    return meta_test_loader
def get_dataloaders(args,mode='openmeta'):
    # dataloader
    num_base = args.num_base
    class_index = np.arange(num_base)
    # class_index = args.known
    assert mode == 'openmeta'

    if 'librispeech' in args.dataset:
        trainset = args.Dataset.Openlbrs(root=args.dataroot,index=class_index,args=args,partition='train', fix_seed=True)
        open_train_loader = torch.utils.data.DataLoader(trainset, 
                                       batch_size=1, shuffle=False,
                                              num_workers=args.dataloader.num_workers, pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=g)
        return open_train_loader
    elif 'nsynth' in args.dataset:
        trainset = args.Dataset.Opennds(root=args.dataroot,index=class_index,args=args,partition='train', fix_seed=True)
        open_train_loader = torch.utils.data.DataLoader(trainset, 
                                       batch_size=1, shuffle=False,
                                              num_workers=args.dataloader.num_workers, pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=g)
        return open_train_loader
    else:
        trainset = args.Dataset.Openfs(root=args.dataroot,index=class_index,args=args,partition='train', fix_seed=True)
        open_train_loader = torch.utils.data.DataLoader(trainset, 
                                       batch_size=1, shuffle=False,
                                              num_workers=args.dataloader.num_workers, pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=g)
        return open_train_loader

def get_dataloader(args, session):
    if session == 0:
        trainset,trainloader= get_base_dataloader_stdu(args)
        return trainset,  trainloader
    else:
        trainset, trainloader= get_new_dataloader(args, session)
        return trainset, trainloader

def get_mixed_openworld_dataloader(args, session):
    """Balanced stream episode: 5 previously seen classes + 5 current novel classes."""
    assert session > 0
    seen_end = args.num_base + (session - 1) * args.way
    episode_seed = _incremental_episode_seed(args, session)
    rng = np.random.RandomState(episode_seed)
    known_classes = np.sort(rng.choice(np.arange(seen_end), size=args.way, replace=False))
    novel_classes = np.arange(args.num_base + (session - 1) * args.way,
                              args.num_base + session * args.way)
    stream_classes = np.concatenate([known_classes, novel_classes])
    if 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=stream_classes,
                                     k=None, base_sess=True, args=args, session=session)
    elif 'nsynth' in args.dataset:
        trainset = args.Dataset.NDS(root=args.dataroot, phase='train', index=stream_classes,
                                    k=None, base_sess=True, args=args)
    elif args.dataset == 'FMC':
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='train', index=stream_classes,
                                        k=None, base_sess=True, args=args)
    else:
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='train',
                                    index=stream_classes, k=None, args=args)
    sampler_generator, worker_generator = _episode_generators(episode_seed)
    sampler = SupportsetSampler(label=trainset.targets, n_cls=2 * args.way,
                                n_per=args.n_shots, n_batch=1,
                                seq_sample=args.seq_sample,
                                generator=sampler_generator)
    loader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler,
                                         num_workers=args.dataloader.num_workers, pin_memory=True,
                                         worker_init_fn=seed_worker,
                                         generator=worker_generator)
    print(f'[EPISODE-SEED] round={int(getattr(args, "current_test", 0))} '
          f'session={int(session)} seed={episode_seed} mixed=True')
    return trainset, loader
    

def get_testloader(args, session):

    # test on all encountered classes
    if session==0:
        class_new = np.arange(0,args.num_base)
    else:
        class_new = np.arange(0,args.num_base+session*args.way)

    if args.dataset == 'FMC':
        testset = args.Dataset.FSDCLIPS(root=args.dataroot, phase="test",
                                      index=class_new, k=None,args=args)
    elif 'nsynth' in args.dataset:
        testset = args.Dataset.NDS(root=args.dataroot, phase="test",
                                      index=class_new, k=None, args=args)
    elif 'librispeech' in args.dataset:
        testset = args.Dataset.LBRS(root=args.dataroot, phase="test",
                                      index=class_new, k=None, args=args)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        testset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase="test",
                                index=class_new, k=None, args=args)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.dataloader.test_batch_size, shuffle=False,
                                             num_workers=args.dataloader.num_workers, pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=g)

    return testset, testloader
def get_inc_testloader(args, session):

    # test on all encountered classes
    if session==0:
        class_new = np.arange(0,args.num_base)
    else:
        class_new = np.arange(args.num_base,args.num_base+session*args.way)

    if args.dataset == 'FMC':
        testset = args.Dataset.FSDCLIPS(root=args.dataroot, phase="test",
                                      index=class_new, k=None,args=args)
    elif 'nsynth' in args.dataset:
        testset = args.Dataset.NDS(root=args.dataroot, phase="test",
                                      index=class_new, k=None, args=args)
    elif 'librispeech' in args.dataset:
        testset = args.Dataset.LBRS(root=args.dataroot, phase="test",
                                      index=class_new, k=None, args=args)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        testset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase="test",
                                index=class_new, k=None, args=args)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.dataloader.test_batch_size, shuffle=False,
                                             num_workers=args.dataloader.num_workers, pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=g)

    return testset, testloader

def get_pretrain_dataloader(args):
    num_base = args.num_base
    class_index = np.arange(num_base)
    # class_index = args.known
    if args.dataset == 'FMC':
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase="train",
                                             index=class_index, base_sess=True,args=args)
        valset = args.Dataset.FSDCLIPS(root=args.dataroot, phase="val", index=class_index, base_sess=True,args=args)
    elif 'nsynth' in args.dataset:
        trainset = args.Dataset.NDS(root=args.dataroot, phase="train",
                                             index=class_index, base_sess=True, args=args)
        valset = args.Dataset.NDS(root=args.dataroot, phase="val", index=class_index, base_sess=True, args=args)
    elif 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase="train",
                                             index=class_index, base_sess=True, args=args)
        valset = args.Dataset.LBRS(root=args.dataroot, phase="val", index=class_index, base_sess=True, args=args)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase="train",
                                             index=class_index, base_sess=True, args=args)
        valset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase="val", index=class_index, base_sess=True, args=args)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.dataloader.train_batch_size, shuffle=True,
                                              num_workers=args.dataloader.num_workers, pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=g)
    
    return trainset,trainloader

def get_base_dataloader_stdu(args):
    if args.tmp_train:
        num_base_class = args.stdu.num_tmpb
        num_incre_class = args.stdu.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    class_index = np.arange(num_base_class + num_incre_class)

    if args.dataset == 'FMC':
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='train', index=class_index, k=None,args=args)
        valset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='val', index=class_index, k=100,args=args) # k is same as new_loader's testset k
    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    elif 'nsynth' in args.dataset:
        trainset = args.Dataset.NDS(root=args.dataroot, phase='train', index=class_index, k=None, args=args,base_sess=True)
        valset = args.Dataset.NDS(root=args.dataroot, phase='val', index=class_index, k=None, args=args) # k is same as new_loader's testset k
    elif 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=class_index, k=None, args=args,base_sess=True)
        valset = args.Dataset.LBRS(root=args.dataroot, phase='val', index=class_index, k=None, args=args) # k is same as new_loader's testset k
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='train', index=class_index, k=None, args=args)
        valset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='val', index=class_index, k=None, args=args) # k is same as new_loader's testset k
    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    train_sampler = TrueIncreTrainCategoriesSampler(label=trainset.targets, n_batch=args.episode.train_episode, 
                                    na_base_cls=num_base_class, na_inc_cls=num_incre_class, 
                                    np_base_cls=args.episode.low_way, np_inc_cls=args.episode.episode_way,
                                    nb_shot=args.episode.low_shot,nn_shot=args.episode.episode_shot, n_query=args.episode.episode_query)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=args.dataloader.num_workers,
                                                pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=g)

    #valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=args.dataloader.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset,trainloader

def get_dataset_for_data_init(args):
    num_base_class = args.num_base

    class_index = np.arange(num_base_class)

    if args.dataset == 'FMC':
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='train', index=class_index, k=None,args=args)
    elif 'nsynth' in args.dataset:
        trainset = args.Dataset.NDS(root=args.dataroot, phase='train', index=class_index, k=None, args=args)
    elif 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=class_index, k=None, base_sess=True,args=args)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:   
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='train', index=class_index, k=None, args=args)
    return trainset

def get_know_dataloader(args, session):
    num_base_class = args.num_base
    session_classes = np.arange(num_base_class )

    if args.dataset == 'FMC':
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='train', index=session_classes, k=None,args=args)
    elif 'nsynth' in args.dataset:
        trainset = args.Dataset.NDS(root=args.dataroot, phase='train', index=session_classes, k=None, base_sess=True,args=args)
    elif 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=session_classes, k=None, base_sess=True,args=args)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='train', index=session_classes, k=None, args=args)
    

    trainloader = torch.utils.data.DataLoader(dataset=trainset,batch_size=args.dataloader.train_batch_size,shuffle=False,num_workers=args.dataloader.num_workers,
                                                pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=g)
  
   
    return trainset, trainloader

def get_new_dataloader(args, session):
    num_base_class = args.num_base
    #args.episode.episode_way*2
    assert session > 0
    if args.dataset == 'FMC':
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='train', index=session_classes,
                                        k=None, base_sess=True, args=args)
    elif 'nsynth' in args.dataset:
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.NDS(root=args.dataroot, phase='train', index=session_classes, k=None, args=args)
    elif 'librispeech' in args.dataset:
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=session_classes, k=None, base_sess=True, args=args, session=session)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='train', index=session_classes, k=None, args=args)
    episode_seed = _incremental_episode_seed(args, session)
    sampler_generator, worker_generator = _episode_generators(episode_seed)
    train_sampler = SupportsetSampler(label=trainset.targets, n_cls=args.way,
                                n_per=args.n_shots, n_batch=1,
                                seq_sample=args.seq_sample,
                                generator=sampler_generator)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=args.dataloader.num_workers,
                                                pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=worker_generator)
    print(f'[EPISODE-SEED] round={int(getattr(args, "current_test", 0))} '
          f'session={int(session)} seed={episode_seed} mixed=False')
                                                
   
    return trainset, trainloader
def get_unknow_dataloader(args, session):
    num_base_class = args.num_base
    #args.episode.episode_way*2
    assert session > 0
    if args.dataset == 'FMC':
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='train', index=session_classes, k=None,args=args)
    elif 'nsynth' in args.dataset:
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.NDS(root=args.dataroot, phase='train', index=session_classes, k=None, args=args)
    elif 'librispeech' in args.dataset:
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='cluster', index=session_classes, k=None, base_sess=True, args=args, session=session)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='train', index=session_classes, k=None, args=args)
    train_sampler = SupportsetSampler(label=trainset.targets, n_cls=args.episode.episode_way, 
                                n_per=args.episode.episode_shot,n_batch=1, seq_sample=args.seq_sample)
    

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=args.dataloader.num_workers,
                                                pin_memory=True,worker_init_fn=seed_worker,  # <--- 新增
        generator=g)
    return trainset, trainloader

def get_session_classes(args,  session):
    num_base_class = args.num_base
    class_list = np.arange(num_base_class + session * args.way)
    return class_list
# [在 baseline/data/dataloader.py 中添加]
# [baseline/data/dataloader.py]

class CurriculumMetaDataset(torch.utils.data.Dataset):
    def __init__(self, root, easy_idx, medium_idx, hard_idx, args):
        self.root = root
        # Chunk 1: 已知 (Memory / Context)
        self.easy_idx = easy_idx      
        # Chunk 2: 当前 (Support / Query)
        self.medium_idx = medium_idx  
        # Chunk 3: 未知 (OpenSet Rejection)
        self.hard_idx = hard_idx      
        self.args = args
        
        # 实例化原始 Dataset 获取全量数据
        all_indices = np.concatenate([easy_idx, medium_idx, hard_idx])
        # 根据 args.dataset 动态选择类，这里假设是 LBRS 为例，请按需修改
        if 'librispeech' in args.dataset:
            self.source = args.Dataset.LBRS(root, phase='train', index=all_indices, base_sess=True, args=args)
        else: # 默认 fallback
            self.source = args.Dataset.FSDCLIPS(root, phase='train', index=all_indices, base_sess=True, args=args)
            
        self.cls_map = {} # class_id -> [sample_indices]
        targets = np.array(self.source.targets) # 或 labels
        for cls in all_indices:
            self.cls_map[cls] = np.where(targets == cls)[0]

    def __len__(self):
        return 500 # 每个 Episode 的迭代次数

    def __getitem__(self, item):
        # 1. 采样 Current Task (Medium)
        # 选 N_way 个 Medium 类
        # 如果 Medium 类不够，允许重复
        replace_medium = len(self.medium_idx) < self.args.n_ways
        selected_medium = np.random.choice(self.medium_idx, self.args.n_ways, replace=replace_medium)
        
        support, query = [], []
        support_labels, query_labels = [], []
        
        for i, cls in enumerate(selected_medium):
            indices = self.cls_map[cls]
            # 采样 Support + Query
            sel_idx = np.random.choice(indices, self.args.n_shots + self.args.n_queries, replace=(len(indices)<(self.args.n_shots+self.args.n_queries)))
            
            for idx in sel_idx[:self.args.n_shots]:
                img, _ = self.source[idx]
                support.append(img)
                support_labels.append(i) # Label: 0 ~ N-1
            
            for idx in sel_idx[self.args.n_shots:]:
                img, _ = self.source[idx]
                query.append(img)
                query_labels.append(i)

        # 2. 采样 Unknown (Hard) -> 作为 OpenSet 数据
        # 如果 Hard 为空（最后一轮），则不采样 (返回空tensor或用Medium充当)
        openset_data, openset_labels = [], []
        if len(self.hard_idx) > 0:
            replace_hard = len(self.hard_idx) < self.args.n_ways
            selected_hard = np.random.choice(self.hard_idx, self.args.n_ways, replace=replace_hard)
            for cls in selected_hard:
                indices = self.cls_map[cls]
                sel_idx = np.random.choice(indices, self.args.n_queries, replace=True)
                for idx in sel_idx:
                    img, _ = self.source[idx]
                    openset_data.append(img)
                    openset_labels.append(self.args.n_ways) # Label: N (Unknown)
        else:
            # 填充全0防止报错，或者在 loss 里 handle
            openset_data = [torch.zeros_like(support[0]) for _ in range(self.args.n_ways * self.args.n_queries)]
            openset_labels = [self.args.n_ways for _ in range(len(openset_data))]

        # 3. 准备 Generator 输入
        # 用户要求: 输入是 [当前类别(Medium) + 已知类别(Easy)]
        # Medium 信息包含在 support 中
        # Easy 信息我们传递 global indices，模型内部通过 fc.weight[base_ids] 获取原型
        base_ids = torch.tensor(self.easy_idx).long()
        
        # 堆叠
        return (torch.stack(support), torch.tensor(support_labels), 
                torch.stack(query), torch.tensor(query_labels), 
                torch.stack(support), torch.tensor(support_labels), # suppopen (generator input)
                torch.stack(openset_data), torch.tensor(openset_labels),
                torch.tensor(selected_medium), # supp_idx
                torch.tensor(selected_hard) if len(self.hard_idx)>0 else torch.tensor([]), # open_idx
                base_ids) # easy_idx

# 获取 Loader 的辅助函数
def get_curriculum_loader(args, easy, medium, hard):
    ds = CurriculumMetaDataset(args.dataroot, easy, medium, hard, args)
    return torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4, pin_memory=True)
