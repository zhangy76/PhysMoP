from __future__ import division

import pickle
from collections import namedtuple

import torch
import numpy as np

from utils.utils import batch_global_rigid_transformation

import config

ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints', 'joints_smpl'
                         ])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)

class SMPL(object):
    def __init__(self, model_path=config.SMPL_MODEL_PATH, device=None):
        super(SMPL, self).__init__()
        # -- Load SMPL params --        
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        # Mean template vertices: [6890, 3]
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float32)
        self.faces = params['f']
        # joint regressor of the official model
        self.regressor = torch.from_numpy(params['V_regressor']).type(torch.float32).transpose(1, 0)
        self.J_regressor = torch.from_numpy(params['J_regressor']).type(torch.float32).transpose(1, 0)
        # Parent for 24 and 37
        self.parents = params['kintree_table'].astype(np.int32)
        self.J_parents = params['kintree_table_J'].astype(np.int32)
        # Shape blend shape basis: [6890, 3, 10]
        # transposed to [10, 6890, 3]
        self.shapedirs = torch.from_numpy(params['shapedirs'].transpose(2,0,1)).type(torch.float32)
        # Pose blend shape basis: [6890, 3, 207]
        # transposed to [207, 6890, 3]
        self.posedirs = torch.from_numpy(params['posedirs'].transpose(2,0,1)).type(torch.float32)
        # LBS weights [6890, 24]
        self.weights = torch.from_numpy(params['weights']).type(torch.float32)
        self.joints_num = 24
        self.verts_num = 6890

        self.device = device if device is not None else torch.device('cpu')
        for name in ['v_template', 'J_regressor', 'regressor', 'weights', 
                     'posedirs', 'shapedirs',
                     ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(self.device))


    def forward(self, betas, rotmat, update_physics=False):    
        """
        Obtain SMPL 3D vertices and joints.
        Args:
            thetas: [N, 6], pose parameters, represented in a axis-angle format. 
            root joint it's global orientation (first three elements).

            betas: [N, 10] shape parameters, as coefficients of
            PCA components.
         Returns:
            verts: [N, 6890, 3], 3D vertices position in camera frame,
            joints: [N, J, 3], 3D joints positions in camera frame. The value 
            of J depends on the joint regressor type.
        """

        N = betas.size()[0]
        v_shape_mean = self.v_template.clone().repeat(N,1,1)
        # 1. Add shape blend shapes
        # (N, 10) x (10, 6890, 3) = [N, 6890, 3]
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1],[0])) + self.v_template.unsqueeze(0)

        # 2. Add pose blend shapes
        # 2.1 Infer shape-dependent joint locations.
        # transpose [N, 6890, 3] to [N, 3, 6890] and perform multiplication
        # transpose results [N, 3, J] to [N, J, 3]
        J = torch.matmul(v_shaped.transpose(1,2), self.regressor).transpose(1,2)
        # 2.2 add pose blend shapes 
        # rotation matrix [N,24,3,3]
        Rs = rotmat
        # ignore global rotation [N,23,3,3]
        pose = Rs[:, 1:, :, :]
        # rotation of T-pose
        pose_I = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # weight matrix [N, 207]
        lrotmin = (pose - pose_I).view(-1, 207)
        # blended model [N,6890,3]
        v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1],[0]))

        # 3. Do LBS
        # obtain the transformed transformation matrix
        _, A = batch_global_rigid_transformation(Rs, J, self.parents)
        # repeat the weight matrix to [N,6890,24]
        W = self.weights.repeat(N,1,1)
        # calculate the blended transformation matrix 
        # [N,6890,24] * [N,24,16] = [N,6890,16] > [N,6890,4,4]
        T = torch.matmul(W, A.view(N,24,16)).view(N,6890,4,4)
        # homegeous form of blended model [N,6890,4]
        v_posed_homo = torch.cat([v_posed, 
                                  torch.ones([N, self.verts_num, 1]).type(torch.float32).to(self.device)], dim=2)
        # calculate the transformed 3D vertices position
        # [N,6890,4,4] * [N,6890,4,1] = [N,6890,4,1]
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        verts = v_homo[:,:,:3,0] # [N,6890,3]

        # estimate 3D joint locations
        joint_regressed = torch.matmul(verts.transpose(1,2), self.J_regressor).transpose(1,2)
        # estimate 3D joint locations
        joint_regressed_smpl = torch.matmul(verts.transpose(1,2), self.regressor).transpose(1,2)

        output = ModelOutput(vertices=verts,
                             joints=joint_regressed,
                             joints_smpl=joint_regressed_smpl)
        return output

class SMPLH(object):
    def __init__(self, gender='neutral', num_betas=10, device=None):
        super(SMPLH, self).__init__()
        # -- Load SMPL params --        
        if gender == 'male':
            params = np.load(config.SMPLH_M_PATH)
        elif gender == 'female':
            params = np.load(config.SMPLH_F_PATH)
        else:
            params = np.load(config.SMPLH_N_PATH)
        # Mean template vertices: [6890, 3]
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float32)
        self.faces = params['f']
        # joint regressor of the official model
        with open(config.SMPL_MODEL_PATH, 'rb') as f:
            params_smpl = pickle.load(f)
        # 24 and 37
        self.regressor = torch.from_numpy(params_smpl['V_regressor']).type(torch.float32).transpose(1, 0)
        self.J_regressor = torch.from_numpy(params_smpl['J_regressor']).type(torch.float32).transpose(1, 0)
        # Parent for 24 and 37
        self.parents = params_smpl['kintree_table'].astype(np.int32)
        self.J_parents = params_smpl['kintree_table_J'].astype(np.int32)

        # transposed to [10, 6890, 3]
        self.shapedirs = torch.from_numpy(params['shapedirs'].transpose(2,0,1)).type(torch.float32)[:num_betas]
        # Pose blend shape basis: [6890, 3, 207]
        # transposed to [207, 6890, 3]
        self.posedirs = torch.from_numpy(params['posedirs'].transpose(2,0,1)).type(torch.float32)[:9*21]
        # LBS weights [6890, 24]
        self.weights = torch.from_numpy(params['weights_prior']).type(torch.float32)
        self.joints_num = 24
        self.verts_num = 6890

        self.device = device if device is not None else torch.device('cpu')
        for name in ['v_template', 'J_regressor', 'regressor', 'weights', 
                     'posedirs', 'shapedirs',
                     ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(self.device))


    def forward(self, betas, rotmat, update_physics=False):    
        """
        Obtain SMPL 3D vertices and joints.
        Args:
            thetas: [N, 6], pose parameters, represented in a axis-angle format. 
            root joint it's global orientation (first three elements).

            betas: [N, 10] shape parameters, as coefficients of
            PCA components.
         Returns:
            verts: [N, 6890, 3], 3D vertices position in camera frame,
            joints: [N, J, 3], 3D joints positions in camera frame. The value 
            of J depends on the joint regressor type.
        """

        N = betas.size()[0]
        v_shape_mean = self.v_template.clone().repeat(N,1,1)
        # 1. Add shape blend shapes
        # (N, 10) x (10, 6890, 3) = [N, 6890, 3]
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1],[0])) + self.v_template.unsqueeze(0)

        # 2. Add pose blend shapes
        # 2.1 Infer shape-dependent joint locations.
        # transpose [N, 6890, 3] to [N, 3, 6890] and perform multiplication
        # transpose results [N, 3, J] to [N, J, 3]
        J = torch.matmul(v_shaped.transpose(1,2), self.regressor).transpose(1,2)
        # 2.2 add pose blend shapes 
        # rotation matrix [N,24,3,3]
        Rs = rotmat
        # ignore global rotation [N,23,3,3]
        pose = Rs[:, 1:-2, :, :]
        # rotation of T-pose
        pose_I = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # weight matrix [N, 207]
        lrotmin = (pose - pose_I).view(-1, 9*21)
        # blended model [N,6890,3]
        v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1],[0]))

        # 3. Do LBS
        # obtain the transformed transformation matrix
        _, A = batch_global_rigid_transformation(Rs, J, self.parents)
        # repeat the weight matrix to [N,6890,24]
        W = self.weights.repeat(N,1,1)
        # calculate the blended transformation matrix 
        # [N,6890,24] * [N,24,16] = [N,6890,16] > [N,6890,4,4]
        T = torch.matmul(W, A.view(N,24,16)).view(N,6890,4,4)
        # homegeous form of blended model [N,6890,4]
        v_posed_homo = torch.cat([v_posed, 
                                  torch.ones([N, self.verts_num, 1]).type(torch.float32).to(self.device)], dim=2)
        # calculate the transformed 3D vertices position
        # [N,6890,4,4] * [N,6890,4,1] = [N,6890,4,1]
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        verts = v_homo[:,:,:3,0] # [N,6890,3]

        # estimate 3D joint locations
        joint_regressed = torch.matmul(verts.transpose(1,2), self.J_regressor).transpose(1,2)
        # estimate 3D joint locations
        joint_regressed_smpl = torch.matmul(verts.transpose(1,2), self.regressor).transpose(1,2)

        output = ModelOutput(vertices=verts,
                             joints=joint_regressed,
                             joints_smpl=joint_regressed_smpl)
        return output
