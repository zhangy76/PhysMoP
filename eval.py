import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import os 
import argparse
from tqdm import tqdm

from dataset.base_dataset_test import BaseDataset_test

from models.evaluator import evaluator
from models.PhysMoP import PhysMoP
from models.humanmodel import SMPL, SMPLH

from utils.utils import compute_errors, batch_roteulerSMPL, remove_singlular_batch

import config
import constants

def run_evaluation(evaluator_ode, checkpoint, restart, batch_size, seqlen=16):
    """Run evaluation on the datasets and metrics we report in the paper. """

    logpath = os.path.join(checkpoint.split('checkpoints')[0])
    logpath = os.path.join(logpath, 'eval')
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # Load SMPL model
    smpl = SMPL(device=device)
    smplh_m = SMPLH(gender='male', device=device)
    smplh_f = SMPLH(gender='female', device=device)

    datasetname = [i for i in config.DATASET_FOLDERS_TEST.keys()]
    evaluation = {}

    for ds in datasetname:
        print('Evaluating: ' + ds)
        filepath = os.path.join(logpath, ds)
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # Create dataloader for the dataset
        data_loader = DataLoader(dataset=BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8)
        N_dataset = len(BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length))
        N = N_dataset 

        mpjpe_data = np.zeros([N, config.total_length])
        mpjpe_physics = np.zeros([N, config.total_length])
        mpjpe_fusion = np.zeros([N, config.total_length])

        accl_data = []
        accl_physics = []
        accl_fusion = []

        time_idx = [1, 3, 7, 9, 13, 17, 21, 24]

        img_paths_all = []
        process_size = batch_size * seqlen
        with torch.no_grad():
            im = 0
            for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
                # Get ground truth annotations from the batch
                input_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                batch_size_s = input_batch['q'].shape[0]
                process_size_s = batch_size_s * seqlen

                _, eval_metrics = evaluator_ode.forward(input_batch, 'torch')
                error_test_data, error_test_physics, error_test_fusion, error_accel_data, error_accel_physics, error_accel_fusion, _, _, _ = eval_metrics

                mpjpe_data[step * batch_size :step * batch_size + batch_size_s] = error_test_data
                mpjpe_physics[step * batch_size :step * batch_size + batch_size_s] = error_test_physics
                mpjpe_fusion[step * batch_size :step * batch_size + batch_size_s] = error_test_fusion

                accl_data.append(error_accel_data)
                accl_physics.append(error_accel_physics)
                accl_fusion.append(error_accel_fusion)

        print('*** Final Results ***')
        print()
        print(ds)
        for idx in time_idx:
            print('Evaluation at Time %dms' % ((idx+1)*40))
            print('MPJPE_data: %.1fmm' % mpjpe_data.mean(axis=0)[idx+config.hist_length])
            print('MPJPE_physics: %.1fmm' % mpjpe_physics.mean(axis=0)[idx+config.hist_length])
            print('MPJPE_fusion: %.1fmm' % mpjpe_fusion.mean(axis=0)[idx+config.hist_length])
            print()
        accl_data = np.concatenate(error_accel_data, axis=0)[config.hist_length:]
        accl_physics = np.concatenate(error_accel_physics, axis=0)[config.hist_length:]
        accl_fusion = np.concatenate(error_accel_fusion, axis=0)[config.hist_length:]
        print('ACCEL_data: %.1f' % np.mean(accl_data))
        print('ACCEL_physics: %.1f' % np.mean(accl_physics))
        print('ACCEL_fusion: %.1f' % np.mean(accl_fusion))
        print()
        np.savez(os.path.join(filepath, 'results.npz'), mpjpe_data=mpjpe_data)

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')

if __name__ == '__main__':

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    evaluator_ode = evaluator(checkpoint_path=args.checkpoint)

    # Run evaluation
    run_evaluation(evaluator_ode, args.checkpoint, False, batch_size=224, seqlen=config.total_length)

