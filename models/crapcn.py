import os
import torch
import copy
import math
import numpy as np
from torch import nn, einsum
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
import torch.nn.functional as F
from torch import einsum
from models.utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, grouping_operation, get_nearest_index, indexing_neighbor

from models.topo_module_v2 import TopoReasoningBlockV2

from esp import EspAttention



# VA from Point Transformer
class VectorAttention(nn.Module):
    def __init__(self, in_channel = 128, dim = 64, n_knn = 16, attn_hidden_multiplier = 4):
        super().__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )
        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, query, support):
        pq, fq = query
        ps, fs = support

        identity = fq 
        query, key, value = self.conv_query(fq), self.conv_key(fs), self.conv_value(fs) 
        
        B, D, N = query.shape

        pos_flipped_1 = ps.permute(0, 2, 1).contiguous() 
        pos_flipped_2 = pq.permute(0, 2, 1).contiguous() 
        idx_knn = query_knn(self.n_knn, pos_flipped_1, pos_flipped_2)

        key = grouping_operation(key, idx_knn) 
        qk_rel = query.reshape((B, -1, N, 1)) - key  

        pos_rel = pq.reshape((B, -1, N, 1)) - grouping_operation(ps, idx_knn)  
        pos_embedding = self.pos_mlp(pos_rel) 

        attention = self.attn_mlp(qk_rel + pos_embedding) 
        attention = torch.softmax(attention, -1)

        value = grouping_operation(value, idx_knn) + pos_embedding  
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  
        output = self.conv_end(agg) + identity
        
        return output

def hierarchical_fps(pts, rates):
    pts_flipped = pts.permute(0, 2, 1).contiguous()
    B, _, N = pts.shape
    now = N
    fps_idxs = []
    for i in range(len(rates)):
        now = now // rates[i]
        if now == N:
            fps_idxs.append(None)
        else:
            fps_idxs.append(furthest_point_sample(pts_flipped, now))
    return fps_idxs

# project f from p1 onto p2
def three_inter(f, p1, p2):
    # print(f.shape, p1.shape, p2.shape)
    # p1_flipped = p1.permute(0, 2, 1).contiguous()
    # p2_flipped = p2.permute(0, 2, 1).contiguous()
    idx, dis = get_nearest_index(p2, p1, k=3, return_dis=True) 
    dist_recip = 1.0 / (dis + 1e-8)
    norm = torch.sum(dist_recip, dim = 2, keepdim = True) 
    weight = dist_recip / norm
    proj_f = torch.sum(indexing_neighbor(f, idx) * weight.unsqueeze(1), dim=-1)
    return proj_f

# Cross-Resolution Transformer
class CRT(nn.Module):
    def __init__(self, dim_in = 128, is_inter = True, down_rates = [1, 4, 2], knns = [16, 12, 8]):
        super().__init__()
        self.down_rates = down_rates
        self.is_inter = is_inter
        self.num_scale = len(down_rates)

        self.attn_lists = nn.ModuleList()
        self.q_mlp_lists = nn.ModuleList()
        self.s_mlp_lists = nn.ModuleList()
        for i in range(self.num_scale):
            self.attn_lists.append(VectorAttention(in_channel = dim_in, dim = 64, n_knn = knns[i]))

        for i in range(self.num_scale - 1):
            self.q_mlp_lists.append(MLP_Res(in_dim = 128*2, hidden_dim = 128, out_dim = 128))
            self.s_mlp_lists.append(MLP_Res(in_dim = 128*2, hidden_dim = 128, out_dim = 128))

    def forward(self, query, support, fps_idxs_q = None, fps_idxs_s = None):
        pq, fq = query
        ps, fs = support
        # prepare fps_idxs_q and fps_idxs_s
        if fps_idxs_q == None:
            fps_idxs_q = hierarchical_fps(pq, self.down_rates)
        
        if fps_idxs_s == None:
            if self.is_inter:
                fps_idxs_s = hierarchical_fps(ps, self.down_rates) # inter-level
            else:
                fps_idxs_s = fps_idxs_q # intra-level
        
        # top-down aggregation
        pre_f = None
        pre_pos = None
        
        for i in range(self.num_scale - 1, -1, -1):
            if fps_idxs_q[i] == None:
                _pos1 = pq
            else:
                _pos1 = gather_operation(pq, fps_idxs_q[i])
            
            if fps_idxs_s[i] == None:
                _pos2 = ps
            else:
                _pos2 = gather_operation(ps, fps_idxs_s[i])

            if i == self.num_scale - 1:
                if fps_idxs_q[i] == None:
                    _f1 = fq
                else:
                    _f1 = gather_operation(fq, fps_idxs_q[i])
                if fps_idxs_s[i] == None:
                    _f2 = fs
                else:
                    _f2 = gather_operation(fs, fps_idxs_s[i])   
                
            else: 
                proj_f1 = three_inter(pre_f, pre_pos, _pos1)
                proj_f2 = three_inter(pre_f, pre_pos, _pos2)
                if fps_idxs_q[i] == None:
                    _f1 = fq
                else:
                    _f1 = gather_operation(fq, fps_idxs_q[i])
                if fps_idxs_s[i] == None:
                    _f2 = fs
                else:
                    _f2 = gather_operation(fs, fps_idxs_s[i]) 
                
                _f1 = self.q_mlp_lists[i](torch.cat([_f1, proj_f1], dim = 1))
                _f2 = self.s_mlp_lists[i](torch.cat([_f2, proj_f2], dim = 1))

            f = self.attn_lists[i]([_pos1, _f1], [_pos2, _f2])

            pre_f = f
            pre_pos = _pos1
        
        agg_f = pre_f
        return agg_f, fps_idxs_q, fps_idxs_s

# encoder
class Encoder(nn.Module):
    def __init__(self, out_dim = 512, n_knn = 16):
        super().__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all = False, if_bn = False, if_idx = True)
        self.crt_1 = CRT(dim_in = 128, is_inter = False, down_rates = [1, 2, 2], knns = [16, 12, 8])
        
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all = False, if_bn = False, if_idx = True)
        self.conv_21 = nn.Conv1d(256, 128, 1)
        self.crt_2 = CRT(dim_in = 128, is_inter = False, down_rates = [1, 2, 2], knns = [16, 12, 8])
        self.conv_22 = nn.Conv1d(128, 256, 1)

        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all = True, if_bn = False)

    def forward(self, partial_cloud):
        l0_xyz = partial_cloud
        l0_points = partial_cloud

        l1_xyz, l1_points, _ = self.sa_module_1(l0_xyz, l0_points)  
        l1_points, _, _ = self.crt_1([l1_xyz, l1_points], [l1_xyz, l1_points], None, None)

        l2_xyz, l2_points, _ = self.sa_module_2(l1_xyz, l1_points)
        l2_points_dim128 = self.conv_21(l2_points)
        l2_points_dim128, _, _ = self.crt_2([l2_xyz, l2_points_dim128], [l2_xyz, l2_points_dim128], None, None)
        l2_points = self.conv_22(l2_points_dim128) + l2_points

        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  

        return l2_xyz, l2_points, l3_points

class UpTransformer(nn.Module):
    def __init__(self, in_channel=128, out_channel=128, dim=64, n_knn=20, up_factor=2,
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        attn_out_channel = dim if attn_channel else 1

        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)



        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        # attention layers
        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor,1), (up_factor,1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # upsample previous feature
        self.upsample1 = nn.Upsample(scale_factor=(up_factor,1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        # residual connection
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos1, query, pos2, key):
        """
        Inputs:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
        """

        value = key # (B, dim, N)
        identity = query
        key = self.conv_key(key) # (B, dim, N)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = query.shape

        pos1_flipped = pos1.permute(0, 2, 1).contiguous()
        pos2_flipped = pos2.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos2_flipped, pos1_flipped) # b, N1, k

        key = grouping_operation(key, idx_knn)  # (B, dim, N1, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos1.reshape((b, -1, n, 1)) - grouping_operation(pos2, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        # attention
        attention = self.attn_mlp(qk_rel + pos_embedding) # (B, dim, N*up_factor, k)

        # softmax function
        attention = self.scale(attention)

        # knn value is correct
        value = grouping_operation(value, idx_knn) + pos_embedding # (B, dim, N, k)
        value = self.upsample1(value) # (B, dim, N*up_factor, k)

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor)
        y = self.conv_end(agg) # (B, out_dim, N*up_factor)

        # shortcut
        identity = self.residual_layer(identity) # (B, out_dim, N)
        identity = self.upsample2(identity) # (B, out_dim, N*up_factor)

        return y+identity

# seed generator using upsample transformer
class SeedGenerator(nn.Module):
    def __init__(self, feat_dim = 512, seed_dim = 128, n_knn = 16, factor = 2, attn_channel = True):
        super().__init__()
        self.uptrans = UpTransformer(in_channel = 256, out_channel = 128, dim = 64, n_knn = n_knn, attn_channel = attn_channel, up_factor = factor, scale_layer = None)
        self.mlp_1 = MLP_Res(in_dim = feat_dim + 128, hidden_dim = 128, out_dim = 128)
        self.mlp_2 = MLP_Res(in_dim = 128, hidden_dim = 64, out_dim = 128)
        self.mlp_3 = MLP_Res(in_dim = feat_dim + 128, hidden_dim = 128, out_dim = seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat, partial):
        x1 = self.uptrans(patch_xyz, patch_feat, patch_xyz, patch_feat)  # (B, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256)
        seed = self.mlp_4(x3)  # (B, 3, 256)
        x = fps_subsample(torch.cat([seed.permute(0, 2, 1).contiguous(), partial], dim=1), 512).permute(0, 2, 1).contiguous() # b, 3, 512
        return seed, x3, x

# seed generator using deconvolution
class SeedGenerator_Deconv(nn.Module):
    def __init__(self, feat_dim = 512, seed_dim = 128, n_knn = 16, factor = 2, attn_channel = True):
        super().__init__()
        num_pc = 256
        self.ps = nn.ConvTranspose1d(feat_dim, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim = feat_dim + 128, hidden_dim = 128, out_dim = 128)
        self.mlp_2 = MLP_Res(in_dim = 128, hidden_dim = 64, out_dim = 128)
        self.mlp_3 = MLP_Res(in_dim = feat_dim + 128, hidden_dim = 128, out_dim = seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat, partial):
        x1 = self.ps(feat)  # (B, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256)
        seed = self.mlp_4(x3)  # (B, 3, 256)
        x = fps_subsample(torch.cat([seed.permute(0, 2, 1).contiguous(), partial], dim=1), 512).permute(0, 2, 1).contiguous() # b, 3, 512
        return seed, x3, x

# mini-pointnet
class PN(nn.Module):
    def __init__(self, feat_dim = 512):
        super().__init__()
        self.mlp_1 = MLP_CONV(in_channel = 3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel = 128 * 2 + feat_dim , layer_dims=[512, 256, 128])
        
    def forward(self, xyz, global_feat):
        b, _, n = xyz.shape
        feat = self.mlp_1(xyz)
        feat4cat = [feat, torch.max(feat, 2, keepdim=True)[0].repeat(1, 1, n), global_feat.repeat(1, 1, n)]
        point_feat = self.mlp_2(torch.cat(feat4cat, dim=1))
        return point_feat

class DeConv(nn.Module):
    def __init__(self, up_factor = 4):
        super().__init__()
        self.decrease_dim = MLP_CONV(in_channel = 128, layer_dims = [64, 32], bn = True)
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias = False)  
        self.mlp_res = MLP_Res(in_dim = 128 * 2, hidden_dim = 128, out_dim = 128)
        self.upper = nn.Upsample(scale_factor = up_factor)
        self.xyz_mlp = MLP_CONV(in_channel = 128, layer_dims = [64, 3])
    def forward(self, xyz, feat):
        feat_child = self.ps(self.decrease_dim(feat))
        feat_child = self.mlp_res(torch.cat([feat_child, self.upper(feat)], dim=1)) # b, 128, n*r
        delta = self.xyz_mlp(torch.relu(feat_child)) 
        new_xyz = self.upper(xyz) + torch.tanh(delta)
        return new_xyz


# upsampling block 
class UpBlock(nn.Module):
    def __init__(self, feat_dim = 512, down_rates = [1, 4, 2], knns = [16, 12, 8], up_factor = 4):
        super().__init__()
        self.pn = PN()
        self.inter_crt = CRT(dim_in = 128, is_inter = True, down_rates = down_rates, knns = knns)
        self.intra_crt = CRT(dim_in = 128, is_inter = False, down_rates = down_rates, knns = knns)
        self.deconv = DeConv(up_factor = up_factor)

    def forward(self, p, gf, pre, fps_idxs_1, fps_idxs_2):
        h = self.pn(p, gf)
        g, fps_idxs_q1, fps_idxs_s1 = self.inter_crt([p, h], pre, None, fps_idxs_1)
        f, _, _ = self.intra_crt([p, g], [p, g], fps_idxs_q1, fps_idxs_q1)
        new_xyz = self.deconv(p, f)
        return new_xyz, f, fps_idxs_q1, fps_idxs_s1

# decoder

class Decoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ub0 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 1)
        self.ub1 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 4)
        self.ub2 = UpBlock(feat_dim = 512, down_rates = [1, 4, 2], knns = [16, 12, 8], up_factor = 8)

    def forward(self, global_f, p0, p_sd, f_sd):
        p1, f0, p0_fps_idxs_122, _ = self.ub0(p0, global_f, [p_sd, f_sd], None, None)
        p2, f1, p1_fps_idxs_122, _ = self.ub1(p1, global_f, [p0, f0], None, p0_fps_idxs_122)
        p3, _ , _______________, _ = self.ub2(p2, global_f, [p1, f1], None, None)
        
        all_pc = [p_sd.permute(0, 2, 1).contiguous(), p1.permute(0, 2, 1).contiguous(), \
            p2.permute(0, 2, 1).contiguous(), p3.permute(0, 2, 1).contiguous()]
        return all_pc

class Decoder_sn55(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ub0 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 1)
        self.ub1 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 4)
        self.ub2 = UpBlock(feat_dim = 512, down_rates = [1, 4, 2], knns = [16, 12, 8], up_factor = 4)

    def forward(self, global_f, p0, p_sd, f_sd):
        p1, f0, p0_fps_idxs_122, _ = self.ub0(p0, global_f, [p_sd, f_sd], None, None)
        p2, f1, p1_fps_idxs_122, _ = self.ub1(p1, global_f, [p0, f0], None, p0_fps_idxs_122)
        p3, _ , _______________, _ = self.ub2(p2, global_f, [p1, f1], None, None)
        
        all_pc = [p_sd.permute(0, 2, 1).contiguous(), p1.permute(0, 2, 1).contiguous(), \
            p2.permute(0, 2, 1).contiguous(), p3.permute(0, 2, 1).contiguous()]
        return all_pc

class Decoder_mvp(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ub0 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 1)
        self.ub1 = UpBlock(feat_dim = 512, down_rates = [1, 2, 2], knns = [16, 12, 8], up_factor = 4)

    def forward(self, global_f, p0, p_sd, f_sd):
        p1, f0, p0_fps_idxs_122, _ = self.ub0(p0, global_f, [p_sd, f_sd], None, None)
        p2, f1, _, __ = self.ub1(p1, global_f, [p0, f0], None, p0_fps_idxs_122)
        
        all_pc = [p_sd.permute(0, 2, 1).contiguous(), p1.permute(0, 2, 1).contiguous(), \
            p2.permute(0, 2, 1).contiguous()]
        return all_pc


# CRA-PCN 
class CRAPCN(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator()
        self.decoder = Decoder()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

class CRAPCN_sn55(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator()
        self.decoder = Decoder_sn55()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

class CRAPCN_mvp(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator()
        self.decoder = Decoder_mvp()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

# CRA-PCN 
class CRAPCN_d(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator_Deconv()
        self.decoder = Decoder()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

class CRAPCN_sn55_d(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator_Deconv()
        self.decoder = Decoder_sn55()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

class CRAPCN_mvp_d(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator_Deconv()
        self.decoder = Decoder_mvp()

    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc

class GraphTopoLayer(nn.Module):
    #这个类包含：图构建 + 单层消息传递(图注意力块)
    
    """
    图拓扑消息传递层（GraphTopoLayer）

    功能：
      - 在当前点坐标 xyz 上构建 k 近邻图（kNN 图）
      - 使用节点特征 h 在图上做一次“信息传递”（message passing）
      - 通过残差连接更新特征：h_out = h + msg

    输入:
      xyz: (B, N, 3)   每个点的三维坐标
      h:   (B, N, C)   每个点的隐藏特征（hidden feature）

    输出:
      h_out: (B, N, C) 更新后的隐藏特征
    """
    def __init__(self, hidden_dim=128, k=16):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim

        # 边特征维度 = h_i (C) + (h_j - h_i)(C) + (x_j - x_i)(3) = 2C + 3
        edge_dim = hidden_dim * 2 + 3

        # 用于计算邻居“重要性”的 MLP，对应注意力分数 score_ij
        self.edge_mlp_attn = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(inplace=True),
            nn.Linear(edge_dim, 1)        # 输出一个标量得分
        )

        # 用于生成邻居“建议消息”的 MLP，对应 m_ij
        self.edge_mlp_msg = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, xyz, h):
        """
        xyz: (B, N, 3)
        h:   (B, N, C)
        """
        from models.utils import query_knn, grouping_operation

        B, N, C = h.shape

        # 1) 在 xyz 上构建 kNN 图
        #    注意：query_knn 期望输入坐标为 (B,N,3)，所以这里不用转置
        idx_knn = query_knn(self.k, xyz, xyz)          # (B,N,k)

        # 2) 取邻居特征
        #    h_t: (B,C,N)，grouping_operation 输出 (B,C,N,k)
        h_t = h.permute(0, 2, 1).contiguous()          # (B,C,N)
        neigh_h = grouping_operation(h_t, idx_knn)     # (B,C,N,k)
        neigh_h = neigh_h.permute(0, 2, 3, 1)          # (B,N,k,C)

        # 中心点特征: (B,N,1,C) -> (B,N,k,C)
        center_h = h.unsqueeze(2).expand(-1, -1, self.k, -1)

        # 3) 取邻居坐标
        #    xyz_t: (B,3,N)，grouping_operation 输出 (B,3,N,k)
        xyz_t = xyz.permute(0, 2, 1).contiguous()      # (B,3,N)
        neigh_xyz = grouping_operation(xyz_t, idx_knn) # (B,3,N,k)
        neigh_xyz = neigh_xyz.permute(0, 2, 3, 1)      # (B,N,k,3)

        # 中心点坐标: (B,N,1,3) -> (B,N,k,3)
        center_xyz = xyz.unsqueeze(2).expand(-1, -1, self.k, -1)

        # 4) 构造边特征 e_ij = [h_i, h_j - h_i, x_j - x_i]
        edge_feat = torch.cat(
            [center_h, neigh_h - center_h, neigh_xyz - center_xyz],
            dim=-1
        )  # (B,N,k,2C+3)

        B_, N_, K_, D_ = edge_feat.shape
        edge_flat = edge_feat.reshape(B_ * N_ * K_, D_)  # (B*N*k, 2C+3)

        # 5) 计算注意力权重 α_ij
        scores = self.edge_mlp_attn(edge_flat).view(B_, N_, K_)  # (B,N,k)
        alpha = F.softmax(scores, dim=-1)                        # (B,N,k)

        # 6) 生成邻居消息并聚合
        msg_flat = self.edge_mlp_msg(edge_flat).view(B_, N_, K_, C)  # (B,N,k,C)
        msg = (alpha.unsqueeze(-1) * msg_flat).sum(dim=2)            # (B,N,C)

        # 7) 残差更新
        h_out = h + msg
        return h_out
class TopoReasoner(nn.Module):
    """
    拓扑图推理模块（TopoReasoner）

    对应前面设计思想中的四个部分：
      1) 图构建（Graph Construction）：以点坐标 xyz 为节点，基于 kNN 建图
      2) 特征嵌入（Feature Embedding）：将 [坐标, 额外特征] 映射到隐藏特征空间
      3) 拓扑消息传递（Topological Message Passing）：多层 GraphTopoLayer
      4) 几何残差预测（Geometric Residual Prediction）：输出每个节点的位移 Δx

    输入:
      xyz:        (B, N, 3)   节点坐标（例如中间层补全点 p2）
      extra_feat: (B, N, Cg)  节点附加特征（例如广播后的全局特征），可为 None

    输出:
      refined_xyz: (B, N, 3)  修正后的点坐标
      delta_xyz:   (B, N, 3)  每个点的位移 Δx
    """
    def __init__(self,
                 in_dim,          # 节点输入特征维度（3 + Cg）
                 hidden_dim=128,  # 图中隐藏特征维度
                 k=16,            # k 近邻邻居数
                 num_layers=2):   # 拓扑消息传递层数
        super().__init__()
        self.k = k
        self.num_layers = num_layers

        # 1) 特征嵌入：将 [xyz, extra_feat] -> 隐藏特征 h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 2) 多层拓扑消息传递层（每层都是一个 GraphTopoLayer）
        self.layers = nn.ModuleList([
            GraphTopoLayer(hidden_dim=hidden_dim, k=k)
            for _ in range(num_layers)
        ])

        # 3) 几何残差预测层：从最终节点特征 h_L 中预测位移 Δx
        self.delta_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, xyz, extra_feat=None):
        """
        xyz:        (B, N, 3)
        extra_feat: (B, N, Cg) 或 None
        """
        B, N, _ = xyz.shape

        # 节点输入特征：如果有额外特征，则拼接 [xyz, extra_feat]；否则仅用 xyz
        if extra_feat is not None:
            node_feat = torch.cat([xyz, extra_feat], dim=-1)  # (B,N,3+Cg)
        else:
            node_feat = xyz                                   # (B,N,3)

        # 1) 特征嵌入
        h = self.node_mlp(node_feat)   # (B,N,H)

        # 2) L 层拓扑消息传递
        for layer in self.layers:
            h = layer(xyz, h)          # 每层都在当前坐标 xyz 上建图

        # 3) 根据最终特征预测每个点的位移 Δx
        delta_xyz = self.delta_mlp(h)  # (B,N,3)

        # 4) 叠加位移得到修正后的坐标
        refined_xyz = xyz + delta_xyz  # (B,N,3)

        return refined_xyz, delta_xyz
class TopoCRAPCN(nn.Module):
    #把拓扑块嵌到到 CRA-PCN 里，并把残差传到 dense 点（p3）。
    """
    带拓扑图推理模块的 CRA-PCN：TopoCRAPCN

    整体流程：
      - 仍使用 CRA-PCN 的 Encoder / SeedGenerator / Decoder 作为几何主干
      - 在中间层点云（这里选 p2）上构建图，并用 TopoReasoner 进行拓扑推理
      - 得到 p2 上的几何位移 Δx，并更新 p2 -> p2_refined
      - 再将 p2 的位移传播到最终稠密点云 p3，得到 p3_refined

    这样就对应了设计思想中的第五步：
      5) 几何修正向 dense 点传播（Propagation to Dense Layer）
    """
    def __init__(self,
                 topo_hidden_dim=128,
                 topo_k=16,
                 topo_layers=2,
                 delta_scale=0.2):   # 新增：缩放系数，控制位移大小
        super().__init__()

        self.delta_scale = delta_scale

        self.encoder = Encoder()
        self.seed_generator = SeedGenerator()
        self.decoder = Decoder()

        self.global_feat_dim = 512

        topo_in_dim = 3 + self.global_feat_dim
        self.topo_reasoner = TopoReasoner(
            in_dim=topo_in_dim,
            hidden_dim=topo_hidden_dim,
            k=topo_k,
            num_layers=topo_layers
        )
   

    def forward(self, xyz, return_all=False):
        """
        xyz: (B, N_in, 3) 残缺点云（partial）

        返回：
          - return_all == False:
              [p_sd, p1, p2_refined, p3_refined]
          - return_all == True:
              一个包含中间变量的字典，便于可视化和设计拓扑损失
        """
        from models.utils import query_knn, grouping_operation  # 用于将位移从 p2 传播到 p3

        # -------- 1) 编码 + 种子生成 + 解码（与原 CRAPCN 相同） --------
        # Encoder 期望输入形状为 (B,3,N)，因此需要转置
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        # global_f: (B, 512, 1)  全局语义特征

        # 种子生成：p_sd 是最粗层的点云，f_sd 是其特征，p0 是中间点云
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)

        # Decoder 生成多尺度补全结果：all_pc = [p_sd, p1, p2, p3]
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        # 这些点云的形状均为 (B, Ni, 3)
        p_sd_xyz = all_pc[0]   # 最粗种子点
        p1_xyz   = all_pc[1]   # 第一层 upsampling 结果
        p2_xyz   = all_pc[2]   # 第二层（中间层）点云 —— 我们在这里做拓扑推理
        p3_xyz   = all_pc[3]   # 最稠密的输出点云

        B, N2, _ = p2_xyz.shape
        _, N3, _ = p3_xyz.shape

        # -------- 2) 构造 p2 节点的额外特征：广播全局特征 --------
        # global_f: (B,512,1) -> (B,512) -> (B,1,512) -> (B,N2,512)
        global_vec = global_f.squeeze(-1)                        # (B,512)
        global_feat_expand = global_vec.unsqueeze(1).repeat(1, N2, 1)  # (B,N2,512)

        # -------- 3) 在 p2 上做拓扑图推理，得到修正后的 p2_refined --------
        
        # 原始拓扑推理
        p2_raw_refined, delta_p2_raw = self.topo_reasoner(
            xyz=p2_xyz,
            extra_feat=global_feat_expand
        )

        # --- 新增：去掉每个样本的全局位移分量，使 Δx 总和为 0 ---
        delta_p2_centered = delta_p2_raw - delta_p2_raw.mean(dim=1, keepdim=True)  # (B,N2,3)

        # 然后再缩放，限制幅度
        delta_p2 = self.delta_scale * delta_p2_centered
        p2_refined = p2_xyz + delta_p2

      



        # -------- 4) 将 p2 上的位移 Δx 传播到最终层 p3 --------
        # 思路：
        #   - 把 p2_refined 作为“支持点”（support），p3 作为“查询点”（query）
        #   - 使用 query_knn 在 p2_refined 上为每个 p3 点找到最近的 k=1 个邻居
        #   - 用 grouping_operation 取出这些邻居的位移 delta_p2，并（可选）做平均

        
       # -------- 4) 将 p2 上的位移 Δx 传播到最终层 p3 --------
        # 这里 query_knn 期望的也是 (B,N,3) 形式的坐标
        # support = p2_refined (B,N2,3), query = p3_xyz (B,N3,3)
        idx_p3 = query_knn(1, p2_refined, p3_xyz)                 # (B,N3,1)

        # delta_p2: (B,N2,3) -> (B,3,N2) 作为“特征”供 grouping_operation 使用
        delta_p2_t = delta_p2.permute(0, 2, 1).contiguous()       # (B,3,N2)

        # 在 p2 上按照 idx_p3 取出最近邻的位移：得到 (B,3,N3,1)
        delta_p3_neigh = grouping_operation(delta_p2_t, idx_p3)   # (B,3,N3,1)

        # 去掉最后一维 -> (B,3,N3)，再转成 (B,N3,3)
        delta_p3_t = delta_p3_neigh.squeeze(-1)                   # (B,3,N3)
        delta_p3 = delta_p3_t.permute(0, 2, 1).contiguous()       # (B,N3,3)

        # 最终修正后的稠密点云
        p3_refined = p3_xyz + delta_p3


        # -------- 5) 组织输出 --------
        if not return_all:
            # 返回多尺度点云，其中 p2/p3 已经包含拓扑修正
            return [p_sd_xyz, p1_xyz, p2_refined, p3_refined]
        else:
            # 返回带更多中间量的字典，便于后续设计拓扑损失或可视化
            return {
                "p_sd": p_sd_xyz,
                "p1": p1_xyz,
                "p2_raw": p2_xyz,
                "p2_refined": p2_refined,
                "p3_raw": p3_xyz,
                "p3_refined": p3_refined,
                "delta_p2": delta_p2,
                "delta_p3": delta_p3,
            }

class TopoCRAPCN_V2(nn.Module):
    """
    使用 TopoReasoningBlockV2 的 CRA-PCN 版本：
      - backbone: Encoder / SeedGenerator / Decoder（全部加载 PCN / CRA-PCN 预训练权重）
      - 在中间层点云 p2 上做拓扑推理，得到位移 Δx
      - 将 Δx 施加到 p2，并通过最近邻传播到最终稠密点 p3
      - 额外输出 p2 的 backbone 特征 feat_p2 与 topo 特征 topo_feat，便于做特征一致性损失
    """
    def __init__(self,
                 topo_hidden_dim=128,
                 topo_k=16,
                 delta_scale=0.2):
        super().__init__()

        self.delta_scale = delta_scale

        # ---- CRA-PCN 主干 ----
        self.encoder = Encoder()
        self.seed_generator = SeedGenerator()
        self.decoder = Decoder()

        # Encoder 输出的全局特征维度（l3_points）
        self.global_feat_dim = 512

        # p2 上局部语义特征的维度：
        # PN 的输出通道固定为 128，在 Decoder 的 ub1 中已经实例化过一个 PN；
        # 这里直接共享这份权重，保证使用的是 backbone 里已有的特征。
        self.semantic_dim = 128
        self.pn_p2 = self.decoder.ub1.pn  # 共享 ub1 的 PN 权重

        # 拓扑推理模块：输入 feat_dim=128 的局部特征 + global_feat
        self.topo_reasoner = TopoReasoningBlockV2(
            feat_dim=self.semantic_dim,
            global_dim=self.global_feat_dim,
            hidden_dim=topo_hidden_dim,
            k=topo_k,
            num_layers=3,   # ✅ 三层 DGC，多层消息传递
        )

    def forward(self, xyz, return_all=False):
        """
        xyz: (B, N_in, 3) 残缺点云（partial）

        return_all == False:
            返回 [p_sd, p1, p2_refined, p3_refined]
        return_all == True:
            返回包含中间结果的 dict，供训练时计算各种损失：
              - p2_raw / p2_refined
              - p3_raw / p3_refined
              - delta_p2 / delta_p3
              - feat_p2_backbone / topo_feat
        """
        from models.utils import query_knn, grouping_operation

        # -------- 1) CRA-PCN 原始前向（不带 topo） --------
        # Encoder 期望输入为 (B,3,N)
        pp, fp, global_f = self.encoder(xyz.permute(0, 2, 1).contiguous())
        # global_f: (B,512,1)

        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        # all_pc: [p_sd, p1, p2, p3]，每个都是 (B,Ni,3)
        p_sd_xyz = all_pc[0]
        p1_xyz   = all_pc[1]
        p2_xyz   = all_pc[2]
        p3_xyz   = all_pc[3]

        B, N2, _ = p2_xyz.shape
        _, N3, _ = p3_xyz.shape

        # -------- 2) 构造 p2 的局部语义特征 feat_p2 --------
        # PN 期望输入 xyz: (B,3,N)，global_feat: (B,512,1)
        p2_xyz_t = p2_xyz.permute(0, 2, 1).contiguous()          # (B,3,N2)
        feat_p2_conv = self.pn_p2(p2_xyz_t, global_f)            # (B,128,N2)
        feat_p2 = feat_p2_conv.permute(0, 2, 1).contiguous()     # (B,N2,128)

        # 同时准备全局向量
        global_vec = global_f.squeeze(-1)                        # (B,512)

        # -------- 3) 在 p2 上做拓扑推理，得到 Δx 和 topo_feat --------
        delta_raw, score, topo_feat = self.topo_reasoner(
            xyz=p2_xyz,           # (B,N2,3)
            feat=feat_p2,         # (B,N2,128)
            global_feat=global_vec,  # (B,512)
        )                           # delta_raw: (B,N2,3), score: (B,N2,1), topo_feat: (B,N2,H)

        # 去掉样本级整体平移分量，使每个样本的 Δx 和为 0
        delta_centered = delta_raw - delta_raw.mean(dim=1, keepdim=True)  # (B,N2,3)

        # 缩放 + 按置信度调节
        delta_p2 = self.delta_scale * delta_centered * score              # (B,N2,3)
        p2_refined = p2_xyz + delta_p2                                    # (B,N2,3)

        # -------- 4) 将 Δx 从 p2 传播到 p3 --------
        idx_p3 = query_knn(1, p2_refined, p3_xyz)                 # (B,N3,1)

        # delta_p2: (B,N2,3) -> (B,3,N2)
        delta_p2_t = delta_p2.permute(0, 2, 1).contiguous()       # (B,3,N2)

        # 在 p2 上按照 idx_p3 取出最近邻的位移：得到 (B,3,N3,1)
        delta_p3_neigh = grouping_operation(delta_p2_t, idx_p3)   # (B,3,N3,1)

        # 去掉最后一维 -> (B,3,N3) -> (B,N3,3)
        delta_p3_t = delta_p3_neigh.squeeze(-1)                   # (B,3,N3)
        delta_p3 = delta_p3_t.permute(0, 2, 1).contiguous()       # (B,N3,3)

        p3_refined = p3_xyz + delta_p3                            # (B,N3,3)

        # -------- 5) 组织输出 --------
        if not return_all:
            # 返回多尺度点云，其中 p2/p3 已经包含拓扑修正
            return [p_sd_xyz, p1_xyz, p2_refined, p3_refined]
        else:
            # 返回带更多中间量的字典，方便训练时取特征/位移做损失
            return {
                "p_sd": p_sd_xyz,
                "p1": p1_xyz,
                "p2_raw": p2_xyz,
                "p2_refined": p2_refined,
                "p3_raw": p3_xyz,
                "p3_refined": p3_refined,
                "delta_p2": delta_p2,
                "delta_p3": delta_p3,
                "feat_p2_backbone": feat_p2,     # (B,N2,128)
                "topo_feat": topo_feat,          # (B,N2,H)
            }

class TopoCRAPCN_V3(nn.Module):
    """
    V3: 在 V2 (TopoCRAPCN_V2) 的基础上，增加 ESPAttention 精修层。

      - 先用 TopoCRAPCN_V2 得到 p3_refined 和 delta_p3；
      - 把 [p3_refined, delta_p3] 映射到特征空间，输入 ESPAttention；
      - 基于 ESP 输出预测精修位移 delta_fine，得到最终点云 p3_final。
    """
    def __init__(self,
                 topo_hidden_dim=128,
                 topo_k=16,
                 topo_layers=2,      # 保留参数签名，方便兼容旧脚本，不实际使用
                 delta_scale=0.2,
                 esp_feat_dim=64,
                 use_topo_v2=True):
        super().__init__()

        self.use_topo_v2 = use_topo_v2

        if self.use_topo_v2:
            # ✅ 使用带 TopoReasoningBlockV2 的版本作为 backbone
            self.backbone = TopoCRAPCN_V2(
                topo_hidden_dim=topo_hidden_dim,
                topo_k=topo_k,
                delta_scale=delta_scale,
            )
        else:
            # 若想退回旧的 GraphTopoLayer 版本，可以把 use_topo_v2=False
            self.backbone = TopoCRAPCN(
                topo_hidden_dim=topo_hidden_dim,
                topo_k=topo_k,
                topo_layers=topo_layers,
                delta_scale=delta_scale,
            )

        self.esp_feat_dim = esp_feat_dim

        # 将 [坐标, coarse 位移] -> 特征 (B,N,6) -> (B,N,C)
        self.feat_mlp = nn.Sequential(
            nn.Linear(6, esp_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(esp_feat_dim, esp_feat_dim),
        )

        # ESP 注意力模块（Earth Mover / Sliced Wasserstein 注意力）
    

        self.esp = EspAttention(
            dim=esp_feat_dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            interp=None,
            learnable=True,
            temperature=10,
            qkv_bias=False,
            max_points=1024,  # 🔹 限制参与 OT 的点数
        )


        # 使用 ESP 输出 + 坐标预测精修位移 Δfine
        self.delta_head = nn.Sequential(
            nn.Linear(esp_feat_dim + 3, esp_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(esp_feat_dim, 3),
        )

    def forward(self, xyz, return_all=False):
        """
        xyz: (B, N_in, 3) 残缺点云
        """
        # 先跑 V2 主干，拿到 p3_refined、delta_p3 等中间结果
        out = self.backbone(xyz, return_all=True)
        p3_refined = out["p3_refined"]                      # (B,N3,3)
        # 有的版本可能没有 delta_p3，这里做个安全回退
        delta_p3 = out.get("delta_p3", torch.zeros_like(p3_refined))

        # 构造 ESP 的输入特征：[坐标, coarse 位移]
        feat_in = torch.cat([p3_refined, delta_p3], dim=-1) # (B,N3,6)
        feat = self.feat_mlp(feat_in)                       # (B,N3,C)

        # 通过 ESPAttention 做非局部精修
        esp_out = self.esp(feat)                            # 可能返回 (feat,) 或 (feat, attn)
        if isinstance(esp_out, tuple):
            feat_esp = esp_out[0]
        else:
            feat_esp = esp_out                              # (B,N3,C)

        # 再拼回坐标，预测精修位移 Δfine
        feat_cat = torch.cat([feat_esp, p3_refined], dim=-1)  # (B,N3,C+3)
        delta_fine = self.delta_head(feat_cat)                # (B,N3,3)
        p3_final = p3_refined + delta_fine

        if not return_all:
            # 输出多尺度点云列表（最后一层换成 p3_final）
            return [out["p_sd"], out["p1"], out["p2_refined"], p3_final]
        else:
            # 在 V2 的中间结果基础上追加 V3 的输出
            out_v3 = dict(out)
            out_v3["p3_final"] = p3_final
            out_v3["delta_fine"] = delta_fine
            return out_v3
