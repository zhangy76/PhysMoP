#!/usr/bin/env python3
# -*- coding: utf-8 -*-

amass_splits = {
    'train': [
        'ACCAD',
        'CMU',
        'EKUT',
        'Eyes_Japan_Dataset',
        'KIT',
        'MPI_Limits',
        'TotalCapture',
        'TCD_handMocap',
    ],
    'test': [
        'BioMotionLab_NTroje',
    ],
}

import os
import os.path as osp
from tqdm import tqdm

import numpy as np
import torch
device = torch.device('cpu')

from utils.utils import batch_rodrigues, rotmat2eulerzyx, rotmat2euleryzx, rotmat2eulerzxy
from scipy.ndimage import gaussian_filter1d, median_filter

# %%
def read_data(trunk_datafolder, trunk_path, folder, sequences, seqlen, overlap=0.75, fps=30):

    for seq_name in sequences:
        print(f'Reading {seq_name} sequence...')
        seq_folder = osp.join(folder, seq_name)

        read_sequence(trunk_datafolder, trunk_path, seq_folder, seq_name, fps, seqlen, overlap)

    return trunk_path

def read_sequence(trunk_datafolder, trunk_path, folder, seq_name, fps, seqlen, overlap):
    
    subjects = sorted(list(filter(lambda x: osp.isdir(os.path.join(folder, x)), os.listdir(folder))))
    for subject in tqdm(subjects):
        actions = [x for x in os.listdir(osp.join(folder, subject)) if x.endswith('.npz')]

        for action in actions:
            fname = osp.join(folder, subject, action)
            
            if fname.endswith('shape.npz'):
                continue

            data = np.load(fname)
            mocap_framerate = int(data['mocap_framerate'])
            try:
                gender = str(data['gender']).decode()
            except:
                gender = str(data['gender'])

            if len(gender)==4:
                gender_id = 0
            if len(gender)==6 or len(gender)==9:
                gender_id = 1
            
            # pose       
            sampling_freq = mocap_framerate // fps
            pose = data['poses'][::sampling_freq]
            pose = torch.from_numpy(pose[:, :22*3]).type(torch.float32).to(device)
            rotmat = batch_rodrigues(pose.view(-1,1,3)).view(-1,22,3,3)

            if pose.shape[0] < 80:
                continue

            euler_root = rotmat2euleryzx(rotmat[:,:1,:,:].clone().view(-1,3,3)).view(-1,1,3)
            euler_s = rotmat2eulerzyx(rotmat[:,1:16,:,:].clone().view(-1,3,3)).view(-1,15,3)
            euler_shoulder = rotmat2eulerzxy(rotmat[:,16:18,:,:].clone().view(-1,3,3)).view(-1,2,3)
            euler_elbow = rotmat2euleryzx(rotmat[:,18:20,:,:].clone().view(-1,3,3)).view(-1,2,3)
            euler_e = rotmat2eulerzyx(rotmat[:,20:,:,:].clone().view(-1,3,3)).view(-1,2,3)

            euler = torch.cat((euler_root,euler_s,euler_shoulder,euler_elbow,euler_e), dim=1).detach().numpy()
            euler = np.delete(euler, [9,10], 1).reshape([-1,20*3])

            euler_smooth = euler.copy()
            for angle in range(euler.shape[1]):
                for t in range(euler.shape[0]-1):
                    while np.abs(euler[t+1,angle]-euler[t,angle])>np.pi:
                        if euler[t+1,angle]<euler[t,angle]:
                            euler[t+1,angle] = euler[t+1,angle] + 2*np.pi
                        else:
                            euler[t+1,angle] = euler[t+1,angle] - 2*np.pi
                euler_smooth[:,angle] = median_filter(euler[:,angle], smooth_sigma//2)
                euler_smooth[:,angle] = gaussian_filter1d(euler_smooth[:,angle].copy(), smooth_sigma) 
    
            # generalized position
            trans = data['trans'][::sampling_freq]
            for i in range(3):
                trans[:, i] = median_filter(trans[:, i], smooth_sigma_va//2)
                trans[:, i] = gaussian_filter1d(trans[:, i], smooth_sigma_va)
                
            q = np.concatenate([trans, euler_smooth], axis=1)[15:-15]
            q[:,3:] = np.remainder(q[:,3:], 2*np.pi)
    
            q_pose = q[:,3:].copy()
            q_pose[q_pose>np.pi] = q_pose[q_pose>np.pi] - 2*np.pi
            q[:,3:] = q_pose

            # shape and gender
            shape = np.repeat(data['betas'][:10][np.newaxis], q.shape[0], axis=0)
            gender_id = np.repeat(np.array([[gender_id]]), q.shape[0], axis=0)
            
            data_i = np.concatenate([q, shape, gender_id], axis=1)

            vid_name = f'{seq_name}_{subject}_{action[:-4]}'

            n_frames = shape.shape[0]
            for frame in range(0, n_frames-seqlen, int(seqlen * (1-overlap))):
                trunk_path_f = trunk_datafolder + '/%s_vid_%d' % (vid_name, frame)
                np.save(trunk_path_f, data_i[frame:frame+seqlen])
                trunk_path.append(trunk_path_f)
    return


# %%
import pickle

amass_dir = './dataset/data_raw/AMASS'
save_dir = './dataset/data_processed'
seqlen = 50

smooth_sigma = 6
smooth_sigma_va = 8
    
for split in ['test', 'train']:
    if split == 'train':
        overlap = 9/10
    else:
        overlap = 0
    trunk_datafolder = osp.join(amass_dir, split, str(seqlen))
    if not osp.exists(trunk_datafolder):
        os.makedirs(trunk_datafolder)
        
    trunk_path = []
    data = read_data(trunk_datafolder, trunk_path, amass_dir, sequences=amass_splits[split], seqlen=seqlen, overlap=overlap)

    data_file = osp.join(save_dir, f'amass_{split}_%d.pkl' % seqlen)
    
    print(f'Saving AMASS dataset to {data_file} with total number of data %d' % len(data))
    pickle.dump(data, open(data_file, 'wb'))


