import io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from dataset.mixed_dataset import MixedDataset
from dataset.base_dataset_test import BaseDataset_test

from models.PhysMoP import PhysMoP
from models.humanmodel import SMPL, SMPLH

from utils.utils import smoothness_constraint, batch_roteulerSMPL, remove_singlular_batch, keypoint_3d_loss
from utils.utils import compute_errors

from train.base_trainer import BaseTrainer

import config
import constants

class Trainer(BaseTrainer):
    
    def init_fn(self):
        self.train_ds = MixedDataset(hist_length=config.hist_length, dataset_name='AMASS')
        
        self.model = PhysMoP(hist_length=config.hist_length,
                                       physics=self.options.physics,
                                       data=self.options.data,
                                       fusion=self.options.fusion
                                       ).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=1e-4)

        self.smpl = SMPL(device=self.device)
        self.smplh_m = SMPLH(gender='male', device=self.device)
        self.smplh_f = SMPLH(gender='female', device=self.device)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.options.checkpoint_steps//2, gamma=0.95, last_epoch=-1)

        self.criterion_mae = nn.L1Loss().to(self.device)
        self.criterion_mse = nn.MSELoss(reduction='none').to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

        # load pretrained model
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        if self.options.fix_weight:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            for name, param in self.model.regressor.fusion_net.named_parameters():
                param.requires_grad = True

        self.process_size = config.total_length * self.options.batch_size

    def forward_kinematics(self, pose, shape, gender_id, process_size_test, vertices=False, joints_smpl=False):
        rotmat, rotMat_individual = batch_roteulerSMPL(pose)

        output_smplh_m = self.smplh_m.forward(betas=shape[gender_id==0], rotmat=rotmat[gender_id==0])
        output_smplh_f = self.smplh_f.forward(betas=shape[gender_id==1], rotmat=rotmat[gender_id==1])
        output_smpl = self.smpl.forward(betas=shape[gender_id==2], rotmat=rotmat[gender_id==2])
        
        if vertices:
            vertices = torch.zeros([process_size_test, 6890, 3]).float().to(self.device)
            vertices[gender_id==0] = output_smplh_m.vertices
            vertices[gender_id==1] = output_smplh_f.vertices
            vertices[gender_id==2] = output_smpl.vertices
        else:
            vertices = None

        joints = torch.zeros([process_size_test, 17, 3]).float().to(self.device)
        joints[gender_id==0] = output_smplh_m.joints[:, :17]
        joints[gender_id==1] = output_smplh_f.joints[:, :17]
        joints[gender_id==2] = output_smpl.joints[:, :17]

        if joints_smpl==True:
            joints_smpl = torch.zeros([process_size_test, 1, 3]).float().to(self.device)
            joints_smpl[gender_id==0] = output_smplh_m.joints_smpl[:, :1]
            joints_smpl[gender_id==1] = output_smplh_f.joints_smpl[:, :1]
            joints_smpl[gender_id==2] = output_smpl.joints_smpl[:, :1]
        else:
            joints_smpl = None

        return vertices, joints, joints_smpl, rotMat_individual

    def train_step(self, input_batch):
        self.model.train()

         # Get data from the batch
        # NxTx63 to N*Tx3
        gt_q = input_batch['q'].type(torch.float32)
        # NxTx10 to N*Tx10
        gt_shape = input_batch['shape'].type(torch.float32).view(self.process_size, 10)
        # NxTx1 to N*Tx1
        gt_gender_id = input_batch['gender_id'].type(torch.float32).view(self.process_size)

        # 3D information
        gt_q[:,:,:3] = gt_q[:,:,:3] - gt_q[:,0:1,:3]
        gt_q = remove_singlular_batch(gt_q)
        gt_q_dot, gt_q_ddot, _ = smoothness_constraint(gt_q, constants.dt)

        gt_pose = torch.zeros([self.options.batch_size, config.total_length, 72]).type(torch.float32).to(self.device) # SMPL pose parameters
        gt_pose[:, :, constants.G2Hpose_idx] = gt_q[:,:,3:]
        gt_pose = gt_pose.view(self.process_size, 72)

        gt_vertices, gt_joints, gt_joints_smpl, gt_rotMat_individual = self.forward_kinematics(gt_pose, gt_shape, gt_gender_id, self.process_size, vertices=True)

        gt_vertices_norm, gt_M_inv, gt_JcT = None, None, None

        model_output = self.model.forward_dynamics(gt_vertices_norm, gt_q, gt_q_ddot, gt_M_inv, gt_JcT, self.device)
        pred_q_data, pred_q_physics_gt, pred_q_physics_pred, pred_q_fusion, pred_q_ddot_data, pred_q_ddot_physics_gt, weight_t = model_output 

        # data-driven
        pred_pose_data = torch.zeros([self.options.batch_size,config.total_length,constants.n_smplpose]).type(torch.float32).to(self.device)
        pred_pose_data[:,:,constants.G2Hpose_idx] = pred_q_data[:,:,3:]
        pred_pose_data = pred_pose_data.reshape([self.process_size, 72])
        _, pred_joints_data, _, _ = self.forward_kinematics(pred_pose_data, gt_shape, gt_gender_id, self.process_size)

        loss_keypoints_data = keypoint_3d_loss(self.criterion_mae, pred_joints_data, gt_joints)
        loss_pose_data = self.criterion_mae(pred_pose_data, gt_pose).mean()

        # physics-gt
        pred_pose_physics_gt = torch.zeros([self.options.batch_size,config.total_length,constants.n_smplpose]).type(torch.float32).to(self.device)
        pred_pose_physics_gt[:,:,constants.G2Hpose_idx] = pred_q_physics_gt[:,:,3:]
        pred_pose_physics_gt = pred_pose_physics_gt.reshape([self.process_size, 72])
        _, pred_joints_physics_gt, _, _ = self.forward_kinematics(pred_pose_physics_gt, gt_shape, gt_gender_id, self.process_size)

        loss_keypoints_physics_gt = keypoint_3d_loss(self.criterion_mae, pred_joints_physics_gt, gt_joints)
        loss_pose_physics_gt = self.criterion_mae(pred_pose_physics_gt, gt_pose).mean()

        # fusion
        pred_pose_fusion = torch.zeros([self.options.batch_size,config.total_length,constants.n_smplpose]).type(torch.float32).to(self.device)
        pred_pose_fusion[:,:,constants.G2Hpose_idx] = pred_q_fusion[:,:,3:]
        pred_pose_fusion = pred_pose_fusion.reshape([self.process_size, 72])
        _, pred_joints_fusion, _, _ = self.forward_kinematics(pred_pose_fusion, gt_shape, gt_gender_id, self.process_size)

        loss_keypoints_fusion = keypoint_3d_loss(self.criterion_mae, pred_joints_fusion, gt_joints)
        loss_pose_fusion = self.criterion_mae(pred_pose_fusion, gt_pose).mean()

        loss_fusion_weight_reg = (weight_t).abs().mean()

        loss = 5000 * (self.options.keypoint_loss_weight_data * loss_keypoints_data +\
               self.options.pose_loss_weight_data * loss_pose_data +\
               self.options.keypoint_loss_weight_physics_gt * loss_keypoints_physics_gt +\
               self.options.pose_loss_weight_physics_gt * loss_pose_physics_gt +\
               self.options.keypoint_loss_weight_fusion * loss_keypoints_fusion +\
               self.options.pose_loss_weight_fusion * loss_pose_fusion +\
               self.options.fusion_weight_reg_loss_weight * loss_fusion_weight_reg)

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        lr_current = self.optimizer.param_groups[0]['lr']

        output = {'pred_q_data': pred_q_data.detach().cpu().numpy(),
                  'pred_q_physics_gt': pred_q_physics_gt.detach().cpu().numpy(),
                  'pred_q_fusion': pred_q_fusion.detach().cpu().numpy(),
                  'gt_q': gt_q.detach().cpu().numpy(),
                  'pred_q_ddot_data': pred_q_ddot_data.detach().cpu().numpy(),
                  'pred_q_ddot_physics_gt': pred_q_ddot_physics_gt.detach().cpu().numpy(),
                  'gt_q_ddot': gt_q_ddot.detach().cpu().numpy(),
                  }

        losses = {'loss': loss.detach().item(),
                  'loss_keypoints_data': self.options.keypoint_loss_weight_data * loss_keypoints_data.detach().item(),
                  'loss_pose_data': self.options.pose_loss_weight_data * loss_pose_data.detach().item(),
                  'loss_keypoints_physics_gt': self.options.keypoint_loss_weight_physics_gt * loss_keypoints_physics_gt.detach().item(),
                  'loss_pose_physics_gt': self.options.pose_loss_weight_physics_gt * loss_pose_physics_gt.mean().detach().item(),
                  'loss_keypoints_fusion': self.options.keypoint_loss_weight_fusion * loss_keypoints_fusion.detach().item(),
                  'loss_pose_fusion': self.options.pose_loss_weight_fusion * loss_pose_fusion.detach().item(),
                  'fusion_weight': self.options.fusion_weight_reg_loss_weight * loss_fusion_weight_reg.detach().item(),
                  'lr':  lr_current,}

        return output, losses

    def train_summaries(self, input_batch, options, output, losses):
        io_buf = io.BytesIO()
        fig = plt.figure(figsize=(12,12), dpi=100)
        fig.suptitle('Pred and Label motion over time')
        q_idx = [6,7,8]
        num_vis = 2

        for i in range(num_vis):
            for j, q_i in enumerate(q_idx):
                ax = fig.add_subplot(num_vis*2, len(q_idx), j+1+i*(len(q_idx)))
                ax.title.set_text('q prediction (q%d)' % q_i)
                ax.plot(range(config.hist_length, config.total_length), output['gt_q'][i,config.hist_length:,q_i], '-r', lw=2, label='q%d'%q_i)
                ax.plot(range(config.hist_length, config.total_length), output['pred_q_data'][i,config.hist_length:,q_i], '-g', lw=2, label='q%d_data'%q_i)
                ax.plot(range(config.hist_length, config.total_length), output['pred_q_physics_gt'][i,config.hist_length:,q_i], '-b', lw=2, label='q%d_phygt'%q_i)
                ax.plot(range(config.hist_length, config.total_length), output['pred_q_fusion'][i,config.hist_length:,q_i], '-k', lw=2, label='q%d_fusion'%q_i)
                ax.legend()
                ax.set_xlim(config.hist_length-1, config.total_length)
                ax = fig.add_subplot(num_vis*2, len(q_idx),j+1+i*(len(q_idx))+num_vis*len(q_idx))
                ax.title.set_text('q_ddot prediction (q%d)' % q_i)
                ax.plot(range(config.hist_length-2, config.total_length-2), output['gt_q_ddot'][i,config.hist_length-2:,q_i], '--r', lw=2, label='q%d'%q_i)
                ax.plot(range(config.hist_length-2, config.total_length-2), output['pred_q_ddot_physics_gt'][i,config.hist_length-2:,q_i], '--b', lw=2, label='q%d_phygt'%q_i)
                ax.legend()
                ax.set_xlim(config.hist_length-3, config.total_length-2)
        fig.savefig(io_buf, format='raw', dpi=100)
        io_buf.seek(0)
        image = np.frombuffer(io_buf.getvalue(), dtype='uint8').reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)).transpose((2, 0, 1))
        self.summary_writer.add_image('Pred vs. Label motion', image, self.step_count)
        io_buf.close()
        plt.clf()
        plt.close(fig)
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)

    def test(self):
        pass
