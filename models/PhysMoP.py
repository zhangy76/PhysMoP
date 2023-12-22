import os 
import copy
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import numpy as np

from models.siMLPe_mlp import build_mlps
from einops.layers.torch import Rearrange

from utils.utils import remove_singlular_batch, smoothness_constraint

import config
import constants

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

        seq_len = config.pred_length
        config.motion_mlp.seq_len = config.hist_length
        config.motion_mlp.num_layers = 16

        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_fc_in_data = nn.Linear(config.dim, config.dim)
        self.motion_mlp_fusion_data = build_mlps(config.motion_mlp)
        self.motion_fc_in_physics = nn.Linear(config.dim, config.dim)
        self.motion_mlp_fusion_physics = build_mlps(config.motion_mlp)
        self.motion_fc_out = nn.Linear(config.dim*3, 1)

        # self.motion_fusion_feats = nn.Linear(config.dim, 1)

    def forward(self, motion_pred_data, motion_pred_physics, motion_feats, t):

        motion_fusion_feats_data = self.motion_fc_in_data(motion_pred_data)
        motion_fusion_feats_data = self.arr0(motion_fusion_feats_data)
        motion_fusion_feats_data = self.arr1(self.motion_mlp_fusion_data(motion_fusion_feats_data))

        motion_fusion_feats_physics = self.motion_fc_in_physics(motion_pred_physics)
        motion_fusion_feats_physics = self.arr0(motion_fusion_feats_physics)
        motion_fusion_feats_physics = self.arr1(self.motion_mlp_fusion_physics(motion_fusion_feats_physics))

        motion_fusion_feats = torch.cat([motion_fusion_feats_data, motion_fusion_feats_physics, motion_feats], dim=2) + t.unsqueeze(2)

        motion_fusion_feats = self.motion_fc_out(motion_fusion_feats) 

        return motion_fusion_feats

class Regression(nn.Module):
    def __init__(self, physics=True, data=True):
        super(Regression, self).__init__()

        self.physics = physics
        self.data = data

        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        """Dataset Config"""
        ## data-driven branch
        config.motion_mlp.seq_len = config.hist_length
        config.motion_mlp.num_layers = 48        
        self.motion_fc_in = nn.Linear(config.dim, config.dim)
        self.motion_mlp = build_mlps(config.motion_mlp)
        self.motion_fc_out = nn.Linear(config.dim, config.dim)

        ## physics branch
        config.motion_mlp.seq_len = 3
        config.motion_mlp.num_layers = 16
        self.motion_mlp_ode = build_mlps(config.motion_mlp)
        self.motion_feats_fc = nn.Linear(config.hist_length*config.dim, config.dim)

        z = config.dim*4
        Jactuation_layer_size = [z, 512, 256, 63]
        C_layer_size = [z, 512, 256, 63]
        M_layer_size = [z, 512, 512, 63*32]

        self.jactuation_net_FC1 = nn.Linear(Jactuation_layer_size[0], Jactuation_layer_size[1])
        self.jactuation_net_relu1 = nn.ReLU()
        self.jactuation_net_FC2 = nn.Linear(Jactuation_layer_size[1], Jactuation_layer_size[2])
        self.jactuation_net_relu2 = nn.ReLU()
        self.jactuation_net_FC3 = nn.Linear(Jactuation_layer_size[2], Jactuation_layer_size[3])

        self.M_net_FC1 = nn.Linear(M_layer_size[0], M_layer_size[1])
        self.M_net_relu1 = nn.ReLU()
        self.M_net_FC2 = nn.Linear(M_layer_size[1], M_layer_size[2])
        self.M_net_relu2 = nn.ReLU()
        self.M_net_FC3 = nn.Linear(M_layer_size[2], M_layer_size[3])

        self.C_net_FC1 = nn.Linear(C_layer_size[0], C_layer_size[1])
        self.C_net_relu1 = nn.ReLU()
        self.C_net_FC2 = nn.Linear(C_layer_size[1], C_layer_size[2])
        self.C_net_relu2 = nn.ReLU()
        self.C_net_FC3 = nn.Linear(C_layer_size[2], C_layer_size[3])

        # fusion
        self.fusion_net = Fusion()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def physics_forward(self, motion_feats_all, motion_current, B, D):

        feature_state = self.motion_fc_in(motion_current)
        feature_state = self.arr0(feature_state)
        feature_state = self.motion_mlp_ode(feature_state)
        feature_state = self.arr1(feature_state).reshape([B, 3*D])
        feature_t = torch.cat([motion_feats_all, feature_state], dim=1)

        # jactuation estimation
        x_jactuation = self.jactuation_net_FC1(feature_t.clone())
        x_jactuation = self.jactuation_net_relu1(x_jactuation)
        x_jactuation = self.jactuation_net_FC2(x_jactuation)
        x_jactuation = self.jactuation_net_relu2(x_jactuation)
        pred_jactuation = self.jactuation_net_FC3(x_jactuation.clone())

        # mass estimation
        x_M = self.M_net_FC1(feature_t.clone())
        x_M = self.M_net_relu1(x_M)
        x_M = self.M_net_FC2(x_M)
        x_M = self.M_net_relu2(x_M)
        pred_M_vector = self.M_net_FC3(x_M.clone())

        # C estimation
        x_C = self.C_net_FC1(feature_t.clone())
        x_C = self.C_net_relu1(x_C)
        x_C = self.C_net_FC2(x_C)
        x_C = self.C_net_relu2(x_C)
        pred_C = self.C_net_FC3(x_C.clone())

        pred_M_inv = torch.zeros((B, D, D)).float().to(feature_state.device)
        tril_indices = torch.tril_indices(row=D, col=D, offset=0)
        pred_M_inv[:,tril_indices[0], tril_indices[1]] = pred_M_vector
        pred_M_inv[:,tril_indices[1], tril_indices[0]] = pred_M_vector
        pred_q_ddot = (pred_M_inv @ (pred_jactuation - pred_C).unsqueeze(2)).squeeze(2) 

        return pred_q_ddot

    def forward(self, motion_input, gt_motion, mode, fusion=True):
        
        motion_feats = self.motion_fc_in(motion_input)
        motion_feats = self.arr0(motion_feats)
        motion_feats = self.motion_mlp(motion_feats)
        motion_feats = self.arr1(motion_feats)

        B, N, D = motion_feats.shape
        ## data-driven 
        motion_pred_data = torch.zeros([B, config.total_length, D]).float().to(motion_input.device)
        if self.data:
            motion_pred_data[:, :config.hist_length] = motion_input
            motion_pred_data[:, config.hist_length:] = self.motion_fc_out(motion_feats) + motion_pred_data[:, config.hist_length-1:config.hist_length]

        ## physics-driven and fusion
        pred_q_ddot_physics_gt = torch.zeros([B, config.total_length-2, D]).float().to(motion_input.device)
        motion_pred_physics_gt = torch.zeros([B, config.total_length, D]).float().to(motion_input.device)
        pred_q_ddot_physics_pred = torch.zeros([B, config.total_length-2, D]).float().to(motion_input.device)
        motion_pred_physics_pred = torch.zeros([B, config.total_length, D]).float().to(motion_input.device)

        motion_pred_fusion = torch.zeros([B, config.total_length, D]).float().to(motion_input.device)

        motion_feats_all = self.motion_feats_fc(motion_feats.reshape([B, N*D]))

        # fusion_weights = torch.tanh(self.fusion_net(motion_feats.reshape([B, N*D]))) ** 2

        motion_pred_physics_gt[:, :3] = motion_input[:, :3].clone()
        motion_pred_physics_pred[:, :config.hist_length] = motion_input[:, :config.hist_length].clone()
        motion_pred_fusion[:, :config.hist_length] = motion_input[:, :config.hist_length].clone()

        for t in range(config.total_length-3):
            ## physics gt history
            if self.physics and mode=='train':
                pred_q_ddot_physics_gt[:, t+1] = self.physics_forward(motion_feats_all.clone(), gt_motion[:, t:t+3], B, D)
                motion_pred_physics_gt[:, t+3] = 2*gt_motion[:, t+2] - gt_motion[:, t+1] + pred_q_ddot_physics_gt[:, t+1].clone() * constants.dt**2                

            if t > config.hist_length-4:
                ## physics pred history
                if mode=='train' and not fusion:
                    pred_q_ddot_physics_pred = pred_q_ddot_physics_gt
                    motion_pred_physics_pred = motion_pred_physics_gt
                else:
                    pred_q_ddot_physics_pred[:, t+1] = self.physics_forward(motion_feats_all.clone(), motion_pred_physics_pred[:, t:t+3], B, D)
                    motion_pred_physics_pred[:, t+3] = 2*motion_pred_physics_pred[:, t+2] - motion_pred_physics_pred[:, t+1] + pred_q_ddot_physics_pred[:, t+1].clone() * constants.dt**2
        ## fusion
        if fusion:
            time_idx = torch.arange(config.pred_length).float().to(motion_input.device).expand(B, -1) / config.pred_length
            weight_t = torch.tanh(self.fusion_net(motion_pred_data[:, config.hist_length:].clone().detach(), motion_pred_physics_pred[:, config.hist_length:].clone().detach(), motion_feats.clone().detach(), time_idx.clone().detach())) ** 2

            for t in range(config.total_length-3):
                if t > config.hist_length-4:
                    pred_q_ddot_physics_pred_fusion = self.physics_forward(motion_feats_all.clone(), motion_pred_fusion[:, t:t+3], B, D)
                    motion_pred_fusion_t = 2*motion_pred_fusion[:, t+2] - motion_pred_fusion[:, t+1] +\
                                            pred_q_ddot_physics_pred_fusion.clone() * constants.dt**2

                    motion_pred_fusion[:, t+3] = (1-weight_t[:, t+3-config.hist_length]) * motion_pred_fusion_t + weight_t[:, t+3-config.hist_length] * motion_pred_data[:, t+3]
                    # motion_pred_fusion[:, t+3] = 0.5 * motion_pred_fusion_t + 0.5 * motion_pred_data[:, t+3]
        else:
            weight_t = torch.FloatTensor(1).fill_(0.).to(motion_input.device)
        return motion_pred_data, motion_pred_physics_gt, motion_pred_physics_pred, motion_pred_fusion, pred_q_ddot_physics_gt, weight_t

class PhysMoP(nn.Module):
    def __init__(
            self,
            hist_length,
            physics=True, 
            data=True,
            fusion=False
    ):

        super(PhysMoP, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hist_length = hist_length
        self.fusion=fusion

        self.regressor = Regression(physics, data)

    def forward_dynamics(self, gt_mesh, gt_q, gt_q_ddot, gt_M_inv, gt_JcT, device, mode='train'):
        # gt_mesh: NxTx6890x3
        # gt_q: NxTx63
        gt_q = gt_q.reshape([-1, config.total_length, 63])
        motion_pred_data, motion_pred_physics_gt, motion_pred_physics_pred, motion_pred_fusion, pred_q_ddot_physics_gt, weight_t = self.regressor(gt_q[:, :self.hist_length], gt_q, mode, self.fusion)
        _, pred_q_ddot_data, _ = smoothness_constraint(motion_pred_data.clone(), constants.dt)

        return (motion_pred_data, motion_pred_physics_gt, motion_pred_physics_pred, motion_pred_fusion, pred_q_ddot_data, pred_q_ddot_physics_gt, weight_t)
