import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


# class Classifier(nn.Module):
#     def __init__(self, args, feat_dim,train_weight_base=False):
#         super().__init__()

        
#         self.calibrator = SupportCalibrator(nway=args.n_ways, feat_dim=feat_dim, n_head=1, base_seman_calib=args.base_seman_calib, neg_gen_type=args.neg_gen_type)
#         self.open_generator = OpenSetGenerater(args.n_ways, feat_dim, n_head=1, neg_gen_type=args.neg_gen_type, agg=args.agg)
#         self.metric  = Metric_Cosine()

#     def forward(self, features, cls_ids, test=False):

#         (support_feat, query_feat, openset_feat) = features
#         (supp_ids, base_ids) = cls_ids
#         base_weights= self.weight_base[base_ids,:]
#         support_feat = torch.mean(support_feat, dim=1)

#         supp_protos= self.calibrator(support_feat, base_weights,test)

#         fakeclass_protos, recip_unit = self.open_generator(supp_protos, base_weights,test)
#         cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=1)
#         #cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=1)
#         #recip_unit = torch.cat([recip_unit,fakeclass_protos],dim=1)
#         query_cls_scores = self.metric(cls_protos, query_feat,test)
#         openset_cls_scores = self.metric(cls_protos, openset_feat,test)

#         test_cosine_scores = (query_cls_scores,openset_cls_scores)
#         query_funit_distance = 1.0- self.metric(recip_unit, query_feat,test)
#         qopen_funit_distance = 1.0-self.metric(recip_unit, openset_feat,test)
#         #funit_distance = torch.cat([query_funit_distance,qopen_funit_distance],dim=1)

#         return test_cosine_scores, supp_protos, fakeclass_protos, query_funit_distance
class Classifier(nn.Module):
    def __init__(self, args, feat_dim,train_weight_base=False):
        super().__init__()

        self.args = args
        self.calibrator = SupportCalibrator(nway=args.n_ways, feat_dim=feat_dim, n_head=1, base_seman_calib=args.base_seman_calib, neg_gen_type=args.neg_gen_type)
        self.open_generator = OpenSetGenerater(args.n_ways, feat_dim, n_head=1, neg_gen_type=args.neg_gen_type, agg=args.agg)
        self.metric  = Metric_Cosine()

    def forward(self, features, cls_ids, test=False):

        (support_feat, query_feat, openset_feat) = features
        (supp_ids, base_ids) = cls_ids
        base_weights= self.weight_base[base_ids,:]
        
        # [修改点 1] 获取均值
        support_feat = torch.mean(support_feat, dim=1)

        # 使用原始特征生成正原型（不再注入噪声，噪声只是遮羞布）
        supp_protos = self.calibrator(support_feat, base_weights, test)

        # 使用干净的原型生成负原型（伪类）
        # 训练时注入噪声帮助 open_generator 泛化到新类原型（测试时 incre_forward 有 osr_noise_std 匹配）
        if not test:
            noise_std = float(getattr(self.args, 'train_noise_std', 0.1))
            supp_for_gen = supp_protos + torch.randn_like(supp_protos) * noise_std
        else:
            supp_for_gen = supp_protos
        fakeclass_protos, recip_unit = self.open_generator(supp_for_gen, base_weights, test)
        
        cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=1)

        # ===== 训练诊断: fake vs supp 的余弦距离 =====
        if not test:
            with torch.no_grad():
                _s = F.normalize(supp_protos, dim=-1)          # [B, nway, D]
                _f = F.normalize(fakeclass_protos, dim=-1)     # [B, nway, D]
                _cos_diag = (_s * _f).sum(dim=-1)              # [B, nway]
                _c = torch.einsum('bnd,bmd->bnm', _s, _f)      # [B, nway, nway]
                _nway = _c.size(1)
                _off_mask = ~torch.eye(_nway, dtype=torch.bool, device=_c.device)
                _cos_offdiag = _c[:, _off_mask].mean()
                _cos_diag_mean = _cos_diag.mean()
                print(f'  [FAKE-DIAG] cos(fake_i,supp_i)={_cos_diag_mean:.4f}  '
                      f'cos(fake_i,supp_j)={_cos_offdiag:.4f}  '
                      f'(1.0=identical, -1.0=opposite)')
        # ===== 诊断结束 =====
        #cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=1)
        #recip_unit = torch.cat([recip_unit,fakeclass_protos],dim=1)
        query_cls_scores = self.metric(cls_protos, query_feat,test)
        openset_cls_scores = self.metric(cls_protos, openset_feat,test)

        test_cosine_scores = (query_cls_scores,openset_cls_scores)
        query_funit_distance = 1.0- self.metric(recip_unit, query_feat,test)
        qopen_funit_distance = 1.0-self.metric(recip_unit, openset_feat,test)
        #funit_distance = torch.cat([query_funit_distance,qopen_funit_distance],dim=1)

        return test_cosine_scores, supp_protos, fakeclass_protos, query_funit_distance  
    def incre_forward(self, features, proto, cls_ids, use_uop=False, uop_chunk_size=5, uop_mode='antimean',
                       osr_noise_std=0.1):
        base_weights = self.weight_base[cls_ids, :]
        supp_protos = self.calibrator(proto, base_weights, True)  # [N, 1, D]

        if not use_uop:
            # 匹配训练分布: 注入噪声防止 att 残差导致 fake≈supp (训练时 Classifier.forward 加了 0.1 噪声)
            if osr_noise_std > 0:
                supp_protos_gen = supp_protos + torch.randn_like(supp_protos) * osr_noise_std
            else:
                supp_protos_gen = supp_protos
            fakeclass_protos, recip_unit = self.open_generator(supp_protos_gen, base_weights, True)
            cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=0)
            query_cls_scores = self.metric(cls_protos, features, True)
            # === 测试时 fake vs supp 余弦距离 ===
            _st = F.normalize(supp_protos.squeeze(1), dim=-1)
            _ft = F.normalize(fakeclass_protos.squeeze(1), dim=-1)
            _cdt = (_st * _ft).sum(dim=-1)
            print(f'  [FAKE-TEST] cos(fake_i,supp_i) mean={_cdt.mean().item():.4f} '
                  f'min={_cdt.min().item():.4f} max={_cdt.max().item():.4f} N={_st.shape[0]}')
            return query_cls_scores

        # ---------- UOP path ----------
        N = supp_protos.shape[0]
        D = supp_protos.shape[-1]
        supp_flat = supp_protos.reshape(N, D)                    # [N, D]
        proto_norm = supp_flat.norm(dim=-1).mean().clamp(min=1e-6)

        if uop_mode == 'chunk':
            chunk = max(1, int(uop_chunk_size))
            fake_list = []
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                sub_supp = supp_protos[start:end]
                sub_base = base_weights[start:end] if base_weights.dim() == 2 else base_weights
                if sub_supp.shape[0] < chunk:
                    pad = chunk - sub_supp.shape[0]
                    sub_supp = torch.cat([sub_supp, sub_supp[-1:].repeat(pad, 1, 1)], dim=0)
                    if sub_base.dim() == 2 and sub_base.shape[0] < chunk:
                        sub_base = torch.cat([sub_base, sub_base[-1:].repeat(pad, 1)], dim=0)
                sub_fake, _ = self.open_generator(sub_supp, sub_base, True)
                real_k = end - start
                fake_list.append(sub_fake[:real_k].reshape(real_k, -1))
            fake_stack = torch.cat(fake_list, dim=0)             # [N, D]
            uop_vec = fake_stack.mean(dim=0)                     # [D]

        elif uop_mode == 'antimean':
            # 归一化 → 求均值 → 反向 → 重标准化并赋 proto 典型范数
            supp_n = F.normalize(supp_flat, dim=-1)
            uop_vec = -supp_n.mean(dim=0)                        # [D]
            uop_vec = F.normalize(uop_vec, dim=-1) * proto_norm

        elif uop_mode == 'anti_nn':
            # antimean 基础上再减去其在 supp span 上的正投影 (进一步 push away)
            supp_n = F.normalize(supp_flat, dim=-1)
            anti = -supp_n.mean(dim=0)                           # [D]
            # projection of anti onto each supp_n, 再平均扣除
            proj_coeff = (supp_n * anti.unsqueeze(0)).sum(dim=-1, keepdim=True)  # [N,1]
            proj = (proj_coeff * supp_n).mean(dim=0)             # [D]
            uop_vec = anti - proj
            uop_vec = F.normalize(uop_vec, dim=-1) * proto_norm

        else:
            raise ValueError(f'unknown uop_mode={uop_mode}')

        uop = uop_vec.view(1, 1, D)                              # [1, 1, D]
        cls_protos = torch.cat([supp_protos, uop], dim=0)        # [N+1, 1, D]
        query_cls_scores = self.metric(cls_protos, features, True)
        return query_cls_scores

    def init_representation(self, params):
        self.weight_base = nn.Parameter(params['fc.weight'], requires_grad=False)

    
    def get_representation(self, cls_ids, base_ids, randpick=False):
        if base_ids is not None:
            base_weights = self.weight_base[base_ids,:]   ## bs*54*D
        else:
            bs = cls_ids.size(0)
            base_weights = self.weight_base.repeat(bs,1,1)
        if randpick:
            num_base = base_weights.shape[1]
            base_size = self.base_size
            idx = np.random.choice(list(range(num_base)), size=base_size, replace=False)
            base_weights = base_weights[:, idx, :]
        return base_weights


class SupportCalibrator(nn.Module):
    def __init__(self, nway, feat_dim, n_head=1,base_seman_calib=True, neg_gen_type='semang'):
        super().__init__()
        self.nway = nway
        self.feat_dim = feat_dim
        self.base_seman_calib = base_seman_calib

        self.calibrator = MultiHeadAttention(feat_dim//n_head, feat_dim//n_head, (feat_dim,feat_dim))

        self.neg_gen_type = neg_gen_type
        if neg_gen_type == 'semang':
            self.task_visfuse = nn.Linear(feat_dim+300,feat_dim)
            self.task_semfuse = nn.Linear(feat_dim+300,300)

    def _seman_calib(self, seman):
        seman = self.map_sem(seman)
        return seman


    def forward(self, support_feat, base_weights,test=False):
        ## support_feat: bs*nway*640, base_weights: bs*num_base*640, support_seman: bs*nway*300, base_seman:bs*num_base*300        
        n_bs, n_base_cls = base_weights.size()[:2]
        support_feat = support_feat.view(-1,1,self.feat_dim)

        
        if self.neg_gen_type == 'semang':
            support_seman = self._seman_calib(support_seman)
            if self.base_seman_calib:
                base_seman = self._seman_calib(base_seman)

            base_seman = base_seman.unsqueeze(dim=1).repeat(1,self.nway,1,1).view(-1, n_base_cls, 300)
            support_seman = support_seman.view(-1, 1, 300)

            base_mem_vis = base_weights
            task_mem_vis = base_weights
            
            base_mem_seman = base_seman
            task_mem_seman = base_seman
            avg_task_mem = torch.mean(torch.cat([task_mem_vis,task_mem_seman],-1), 1, keepdim=True)

            gate_vis = torch.sigmoid(self.task_visfuse(avg_task_mem)) + 1.0
            gate_sem = torch.sigmoid(self.task_semfuse(avg_task_mem)) + 1.0

            base_weights = base_mem_vis * gate_vis 
            base_seman = base_mem_seman * gate_sem

        elif self.neg_gen_type == 'attg':
            # 修复: 将 base_weights 扩展到与 support_feat 相同的 batch 维度
            # train: base_weights=[bs,num_base,D] → [bs*nway,num_base,D]
            # test:  base_weights=[num_base,D]    → [N,num_base,D] (N=num_labeled_classes)
            if base_weights.dim() == 2:
                _bw = base_weights.unsqueeze(0).expand(support_feat.shape[0], -1, -1).contiguous()
            else:
                _bw = base_weights.unsqueeze(1).expand(-1, self.nway, -1, -1).reshape(-1, base_weights.size(1), self.feat_dim)
            base_mem_vis = _bw
            base_weights = _bw
            base_seman = None
            support_seman = None

        elif self.neg_gen_type == 'att':
            base_weights = support_feat
            base_mem_vis = support_feat
            support_seman = None
            base_seman = None

        else:
            return support_feat.view(n_bs,self.nway,-1), None

        support_center = self.calibrator(support_feat, base_weights, base_mem_vis, support_seman, base_seman)
        if not test:
            support_center = support_center.view(n_bs,self.nway,-1) #训练的时候需要放出来
        #support_attn = support_attn.view(n_bs,self.nway,-1)
        return support_center


class OpenSetGenerater(nn.Module):
    def __init__(self, nway, featdim, n_head=1, neg_gen_type='semang', agg='avg'):
        super().__init__()
        self.nway = nway
        self.att = MultiHeadAttention(featdim//n_head, featdim//n_head, (featdim,featdim))
        self.featdim = featdim

        self.neg_gen_type = neg_gen_type
        if neg_gen_type == 'semang':
            self.task_visfuse = nn.Linear(featdim+300,featdim)
            self.task_semfuse = nn.Linear(featdim+300,300)


        self.agg = agg
        if agg == 'mlp':
            self.agg_func = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim))
            
        

    def _seman_calib(self, seman):
        ### feat: bs*d*feat_dim, seman: bs*d*300
        seman = self.map_sem(seman)
        return seman

    def forward(self, support_center, base_weights,test=False):
        ## support_center: bs*nway*D
        ## weight_base: bs*nbase*D
        bs = support_center.shape[0]
        n_bs, n_base_cls = base_weights.size()[:2]
        support_center = support_center.view(-1, 1, self.featdim)
       
        if self.neg_gen_type=='semang':
            support_seman = self._seman_calib(support_seman)
            base_seman = base_seman.unsqueeze(dim=1).repeat(1,self.nway,1,1).view(-1, n_base_cls, 300)
            support_seman = support_seman.view(-1, 1, 300)
            
            base_mem_vis = base_weights
            task_mem_vis = base_weights
            
            base_mem_seman = base_seman
            task_mem_seman = base_seman
            avg_task_mem = torch.mean(torch.cat([task_mem_vis,task_mem_seman],-1), 1, keepdim=True)

            gate_vis = torch.sigmoid(self.task_visfuse(avg_task_mem)) + 1.0
            gate_sem = torch.sigmoid(self.task_semfuse(avg_task_mem)) + 1.0
            
            base_weights = base_mem_vis * gate_vis 
            base_seman = base_mem_seman * gate_sem

        
        elif self.neg_gen_type == 'attg':
            # 修复: 将 base_weights 扩展到与 support_center 相同的 batch 维度
            # train: [bs,num_base,D] → [bs*nway,num_base,D]; test: [num_base,D] → [N,num_base,D]
            if base_weights.dim() == 2:
                _bw = base_weights.unsqueeze(0).expand(support_center.shape[0], -1, -1).contiguous()
            else:
                _bw = base_weights.unsqueeze(1).expand(-1, self.nway, -1, -1).reshape(-1, base_weights.size(1), self.featdim)
            base_weights = _bw
            base_mem_vis = _bw
            support_seman = None
            base_seman = None

        elif self.neg_gen_type == 'att':
            base_weights = support_center
            base_mem_vis = support_center
            support_seman = None
            base_seman = None

        else:
            fakeclass_center = support_center.mean(dim=0, keepdim=True)
            if self.agg == 'mlp':
                fakeclass_center = self.agg_func(fakeclass_center)
            return fakeclass_center, support_center.view(bs, -1, self.featdim)
            

        output= self.att(support_center, base_weights, base_mem_vis, support_seman, base_seman, mark_res=False)  ## bs*nway*nbase (*NO residual: 避免 fake=supp)

        output = output.view(bs, -1, self.featdim)
        # print(output.shape)
        # if not test:
        #     fakeclass_center = output.mean(dim=1,keepdim=True) #训练要改成1
        # else:
        #     fakeclass_center = output.mean(dim=0,keepdim=True)
        fakeclass_center = output
        if self.agg == 'mlp':
            fakeclass_center = self.agg_func(fakeclass_center)
        
        return fakeclass_center, output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_k, d_v, d_model, n_head=1, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        #### Visual feature projection head
        self.w_qs = nn.Linear(d_model[0], n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model[1], n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model[-1], n_head * d_v, bias=False)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model[0] + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model[1] + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model[-1] + d_v)))


        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))            
        self.fc = nn.Linear(n_head * d_v, d_model[0], bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, q_sem=None, k_sem=None, mark_res=True):
        ### q: bs*nway*D
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv


        output= self.attention(q, k, v, q_sem, k_sem)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        if mark_res:
            output = output + residual

        return output
    



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, q_sem = None, k_sem = None):

        attn_score = torch.bmm(q, k.transpose(1, 2))

        attn_score = attn_score / self.temperature
        attn = self.softmax(attn_score)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)
        return output
        

class Metric_Cosine(nn.Module):
    def __init__(self, temperature=10):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, supp_center, query_feature,test=False):
        ## supp_center: bs*nway*D
        ## query_feature: bs*(nway*nquery)*D
        supp_center = F.normalize(supp_center, dim=-1) # eps=1e-6 default 1e-12
        query_feature = F.normalize(query_feature, dim=-1)
        if not test:
            logits = torch.bmm(query_feature.unsqueeze(0), supp_center.transpose(1,2)) #训练.transpose(1,2)  测试是.transpose(0,1)
        else:
            logits = torch.bmm(query_feature.unsqueeze(0), supp_center.transpose(0,1).transpose(1,2))
        result = logits * self.temp
        return  result
    
