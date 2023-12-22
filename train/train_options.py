import os
import json
import argparse
import numpy as np
from collections import namedtuple

import config

class TrainOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf, help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=12, help='Number of processes used for data loading')
        gen.add_argument('--gpu', type=str, default='0', help='GPU to be used')

        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')

        io.add_argument('--pretrained_checkpoint', default=None, help='Load a pretrained checkpoint at the beginning training')
        io.add_argument('--physics', default=False, help='Train the physics-based model')
        io.add_argument('--data', default=False, help='Train the data-driven model')
        io.add_argument('--fix_weight', default=False, help='Fix the weights of the data-driven and physics-based model')
        io.add_argument('--fusion', default=False, help='Train the fusion model')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=5, help='Total number of training epochs')
        train.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
        train.add_argument('--batch_size', type=int, default=64, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=500, help='Summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=5000, help='Checkpoint saving frequency')

        train.add_argument('--test_steps', type=int, default=5000, help='Testing frequency during training')
        train.add_argument('--test_batchsize', type=int, default=64, help='Testing batch_size')

        train.add_argument('--keypoint_loss_weight_data', default=0., type=float, help='Weight of 3D keypoint loss for training the data-driven model')
        train.add_argument('--pose_loss_weight_data', default=0., type=float, help='Weight of joint angle loss for training the data-driven model')
        train.add_argument('--keypoint_loss_weight_physics_gt', default=0., type=float, help='Weight of 3D keypoint loss for training the physics-based model')
        train.add_argument('--pose_loss_weight_physics_gt', default=0., type=float, help='Weight of joint angle loss for training the physics-based model')
        train.add_argument('--keypoint_loss_weight_fusion', default=0., type=float, help='Weight of 3D keypoint loss for training the fusion model')
        train.add_argument('--pose_loss_weight_fusion', default=0., type=float, help='Weight of joint angle loss for training the fusion model')
        train.add_argument('--fusion_weight_reg_loss_weight', default=0., type=float, help='Regularization on the fusion weight')

        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')
        shuffle_train.set_defaults(shuffle_train=True)
        return 

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
                json_args = namedtuple("json_args", json_args.keys())(**json_args)
                return json_args
        else:
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return
