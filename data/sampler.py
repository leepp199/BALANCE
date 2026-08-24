import torch
import numpy as np
import copy
import math

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

class TrueIncreTrainCategoriesSampler():

    def __init__(self, label, n_batch, na_base_cls, na_inc_cls, np_base_cls, np_inc_cls, nb_shot, nn_shot, n_query):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.na_base_cls = na_base_cls
        self.na_inc_cls = na_inc_cls
        self.np_base_cls = np_base_cls
        self.np_inc_cls = np_inc_cls
        self.nb_shot = nb_shot
        self.nn_shot = nn_shot
        self.n_query = n_query
        self.base_samples_per_cls = nb_shot + n_query
        self.novel_samples_per_cls = nn_shot + n_query
        # self.n_base_test_samples = np_inc_cls * n_query
        self.all_cls = np.arange(na_base_cls + na_inc_cls)

        label = np.array(label)  # all data label
        self.tmp_base_ind = []  # the data index of each temp base class
        for i in range(self.na_base_cls):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            # ind = torch.from_numpy(ind)
            self.tmp_base_ind.append(ind)

        self.tmp_incre_ind = []  # the data index of each incremental train class
        for i in range(self.na_base_cls, self.na_base_cls + self.na_inc_cls):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            # ind = torch.from_numpy(ind)
            self.tmp_incre_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            base_batch = []
            tmp_base_classes = torch.randperm(len(self.tmp_base_ind))[:self.np_base_cls]
            for c in tmp_base_classes:
                l = torch.from_numpy(self.tmp_base_ind[c])  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.base_samples_per_cls]  # sample n_per data index of this class
                base_batch.append(l[pos])
            base_batch = torch.stack(base_batch).t().reshape(-1)

            incre_fs_batch = []
            inc_classes = torch.randperm(len(self.tmp_incre_ind))[:self.np_inc_cls]  # random sample num_class indexs,e.g. 5
            for c in inc_classes:
                l = torch.from_numpy(self.tmp_incre_ind[c])  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.novel_samples_per_cls]  # sample n_per data index of this class
                incre_fs_batch.append(l[pos])
            incre_fs_batch = torch.stack(incre_fs_batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd

            batch = torch.concat([base_batch, incre_fs_batch])
            yield batch



class SupportsetSampler():

    def __init__(self, label, n_cls, n_per, n_batch=1, seq_sample=False,
                 generator=None):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per
        self.seq_sample = seq_sample
        self.generator = generator
        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(min(label), max(label) + 1):
            if i in label:
                ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
                ind = torch.from_numpy(ind)
                self.m_ind.append(ind)
        # print(f"label: {label}")
        # print("len(self.m_ind):", len(self.m_ind))
        # print("self.n_cls:", self.n_cls)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            # print("len(self.m_ind):", len(self.m_ind))
            # print("self.n_cls:", self.n_cls)
            assert len(self.m_ind) == self.n_cls
            if self.seq_sample:
                classes =  list(range(len(self.m_ind)))[:self.n_cls]
            else:
                classes = torch.randperm(
                    len(self.m_ind), generator=self.generator)[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                if self.seq_sample:
                    pos = list(range(len(l)))[:self.n_per]
                else:
                    pos = torch.randperm(
                        len(l), generator=self.generator)[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
class CurriculumSampler():
    """
    支持 UACL 课程学习策略的采样器
    """
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.label = np.array(label)
        
        # 索引所有类别的数据
        self.m_ind = {}
        for i in range(max(self.label) + 1):
            ind = np.argwhere(self.label == i).reshape(-1)
            if len(ind) > 0:
                self.m_ind[i] = torch.from_numpy(ind)
        
        # 初始状态：所有基础类都是活跃的 (用于 Warm-up)
        self.active_classes = list(self.m_ind.keys())

    def set_active_classes(self, class_list):
        """
        [关键修改] 更新当前允许采样的类别列表
        """
        # 过滤无效类
        valid = [c for c in class_list if c in self.m_ind]
        
        # 确保活跃类别数至少满足 n_way (否则无法组成 episode)
        if len(valid) < self.n_cls:
            print(f"[Sampler Warning] Active classes {len(valid)} < n_way {self.n_cls}. Filling with random base classes.")
            needed = self.n_cls - len(valid)
            remain = [c for c in self.m_ind.keys() if c not in valid]
            if len(remain) >= needed:
                valid += list(np.random.choice(remain, needed, replace=False))
            else:
                 valid += list(np.random.choice(remain, needed, replace=True))

        self.active_classes = valid
        print(f"==> [Sampler] Active Pool Updated: {len(self.active_classes)} classes available.")

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # [关键修改] 只从 active_classes 中采样类别
            # 这是实现 "Easy-to-Hard" 课程的核心
            if len(self.active_classes) >= self.n_cls:
                classes = np.random.choice(self.active_classes, self.n_cls, replace=False)
            else:
                classes = np.random.choice(self.active_classes, self.n_cls, replace=True)
                
            for c in classes:
                l = self.m_ind[c]
                # 在选定类中随机采样样本
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
