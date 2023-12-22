import os 
import numpy as np
import torch

if __name__ == '__main__':
    from train.train_options import TrainOptions

    options = TrainOptions().parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    seed = 1234567890
    np.random.seed(seed)
    torch.manual_seed(seed)

    from train.trainer import Trainer

    trainer = Trainer(options)
    trainer.train()
