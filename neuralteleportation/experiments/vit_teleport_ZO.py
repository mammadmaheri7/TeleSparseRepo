import itertools
import torch

import random
import math
import numpy as np
import os

import torch
from torch import nn
import torch.nn.functional as Fhtop

import argparse

import argparse
import ezkl

def get_args_parser(initial_args=None):
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
#     parser.add_argument('--batch_size', default=256, type=int,
#                         help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
#     parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
#                         help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    
    ########################## settings specific to this project ##########################
    
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument('--dropout', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    
    # early / late dropout and stochastic depth settings
    parser.add_argument('--drop_mode', type=str, default='standard', choices=['standard', 'early', 'late'], help='drop mode')
    parser.add_argument('--drop_schedule', type=str, default='constant', choices=['constant', 'linear'], 
                        help='drop schedule for early dropout / s.d. only')
    parser.add_argument('--cutoff_epoch', type=int, default=0, 
                        help='if drop_mode is early / late, this is the epoch where dropout ends / starts')
    
    ####################################################################################### 
    
    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=50, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
#     parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
#                         help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

#     parser.add_argument('--resume', default='',
#                         help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='convnext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")

    # arguments for pruning
    parser.add_argument("--nsamples", type=int, default=4096)
#     parser.add_argument("--sparsity", type=float, default=0.)
    parser.add_argument("--prune_metric", type=str, choices=["magnitude", "wanda"])
    parser.add_argument("--prune_granularity", type=str)
    parser.add_argument("--blocksize", type=int, default=1)

    return parser

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

class DefaultArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def get_default_args():
    parser = get_args_parser()
    default_args = {}
    for action in parser._actions:
        # Check if action is an argument
        if not action.option_strings:
            continue
        # Use the destination as the key and the default value as the value
        default_args[action.dest] = action.default
    return DefaultArgs(**default_args)


# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val_dirs')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

import timm

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path
from timm.models import create_model
import torch
import torch.distributed as dist
# from torch._six import inf

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    def __init__(self, args):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run 
        if self._wandb.run is None:
            self._wandb.init(
                project=args.project,
                config=args
            )

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/test metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)

        # Log current epoch
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'test' in k:
                self._wandb.log({f'Global Test/{k}': v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):

    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
        
    pattern = re.compile(r'^blocks\.(\d+)\.attn\.q\.weight$')
    state_dict_keys = list(state_dict.keys())
    
    for key in state_dict_keys:
        match = pattern.match(key)
        if match:
            index = int(match.group(1))
            query_key = key
            key_key = key.replace("q","k")
            value_key = key.replace("q","v")
            
            new_name = "blocks." + str(index) +".attn.qkv.weight"
            state_dict[new_name] = torch.cat([state_dict[query_key], state_dict[key_key], state_dict[value_key]], dim=0)
            
            print("index:",index,"\t new_name:",new_name)
            del state_dict[query_key], state_dict[key_key], state_dict[value_key]
            
    
    pattern = re.compile(r'^blocks\.(\d+)\.attn\.q\.bias$')
    
    for key in state_dict_keys:
        match = pattern.match(key)
        if match:
            index = int(match.group(1))
            query_key = key
            key_key = key.replace("q","k")
            value_key = key.replace("q","v")
            
            new_name = "blocks." + str(index) +".attn.qkv.bias"
            state_dict[new_name] = torch.cat([state_dict[query_key], state_dict[key_key], state_dict[value_key]], dim=0)
            
            print("index:",index,"\t new_name:",new_name)
            del state_dict[query_key], state_dict[key_key], state_dict[value_key]
                                              
    

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)
            

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        ########################################################
        ## Code I added 
        for param in parameters:
            weight_copy = param.data.abs().clone()
            mask = weight_copy.gt(0).float().cuda()
            sparsity = mask.sum() / mask.numel()
            if sparsity > 0.3:
                # non-trivial sparsity 
                param.grad.data.mul_(mask)
        ########################################################

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)

    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str): # does not support resuming with 'best', 'best-ema'
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                assert args.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint['model_ema'])
                else:
                    model_ema.ema.load_state_dict(checkpoint['model'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

def reg_scheduler(base_value, final_value, epochs, niter_per_ep, early_epochs=0, early_value=None, 
           mode='linear', early_mode='regular'):
    early_schedule = np.array([])
    early_iters = early_epochs * niter_per_ep
    if early_value is None:
        early_value = final_value
    if early_epochs > 0:
        print(f"Set early value to {early_mode} {early_value}")
        if early_mode == 'regular':
            early_schedule = np.array([early_value] * early_iters)
        elif early_mode == 'linear':
            early_schedule = np.linspace(early_value, base_value, early_iters)
        elif early_mode == 'cosine':
            early_schedule = np.array(
            [base_value + 0.5 * (early_value - base_value) * (1 + math.cos(math.pi * i / early_iters)) for i in np.arange(early_iters)])
    regular_epochs = epochs - early_epochs
    iters = np.arange(regular_epochs * niter_per_ep)
    schedule = np.linspace(base_value, final_value, len(iters))
    schedule = np.concatenate((early_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def build_model(args, pretrained=False):
    if args.model.startswith("convnext"):
        model = create_model(
            args.model,
            pretrained=pretrained,
            num_classes=args.nb_classes,
            layer_scale_init_value=args.layer_scale_init_value,
            head_init_scale=args.head_init_scale,
            drop_path_rate=args.drop_path,
            drop_rate=args.dropout,
            )
    else:
        model = create_model(
            args.model, 
            pretrained=pretrained, 
            num_classes=args.nb_classes, 
            drop_path_rate=args.drop_path,
            drop_rate =args.dropout
            )
    return model

import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        
        self.scale = qk_scale or head_dim ** -0.5
#         self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.query_linear = nn.Linear(dim, dim, bias=qkv_bias)
#         self.key_linear = nn.Linear(dim, dim, bias=qkv_bias)
#         self.value_linear = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        
#         query = self.query_linear(x).reshape(B,N,self.num_heads, C // self.num_heads)
#         key = self.key_linear(x).reshape(B,N,self.num_heads, C // self.num_heads)
#         value = self.value_linear(x).reshape(B,N,self.num_heads, C // self.num_heads)
        
        
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        qkvs = self.qkv(x)
        query = qkvs[:,:,0:self.dim]
        key = qkvs[:,:,self.dim:2*self.dim]
        value = qkvs[:,:,2*self.dim:] 
        query = query.reshape(B,N,self.num_heads, C // self.num_heads)
        key = key.reshape(B,N,self.num_heads, C // self.num_heads)
        value = value.reshape(B,N,self.num_heads, C // self.num_heads)
        

#         attn = (q @ k.transpose(-2, -1)) * self.scale     
        attn = query.transpose(1,2) @ key.transpose(1,2).transpose(2,3)
        attn = attn * self.scale


        attn = attn.softmax(dim=(-1))
        attn = self.attn_drop(attn)

        
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        y = (attn@value.transpose(1,2)).transpose(1,2).reshape(B,N,C)
        
#         x = self.proj(x)
#         x = self.proj_drop(x)
        y = self.proj(y)
        y = self.proj_drop(y)
#         return x
        return y

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
#         self.norm1 = norm_layer(self.dim)
        self.norm1 = norm_layer(self.dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
        mean = torch.mean(x, dim=2)  # Calculate mean along the last dimension
        mean = mean.unsqueeze(2).expand(-1,-1,x.shape[2])

        diff_squared = (x - mean) ** 2
        std = torch.sqrt(torch.mean(diff_squared, dim=2))
        std = std.unsqueeze(2).expand(-1,-1,x.shape[2])
        
        norm_x = (x - mean) 
        norm_x = norm_x / (std+1e-06)
        norm_x = norm_x * self.norm1.weight.unsqueeze(0).unsqueeze(0)
        norm_x = norm_x + self.norm1.bias.unsqueeze(0).unsqueeze(0)
        
        x = x + self.drop_path(self.attn(norm_x))      
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
    def half_block(self,x):
        mean = torch.mean(x, dim=2)  # Calculate mean along the last dimension
        mean = mean.unsqueeze(2).expand(-1,-1,x.shape[2])

        diff_squared = (x - mean) ** 2
        std = torch.sqrt(torch.mean(diff_squared, dim=2))
        std = std.unsqueeze(2).expand(-1,-1,x.shape[2])
        
        norm_x = (x - mean) 
        norm_x = norm_x / (std+1e-06)
        norm_x = norm_x * self.norm1.weight.unsqueeze(0).unsqueeze(0)
        norm_x = norm_x + self.norm1.bias.unsqueeze(0).unsqueeze(0)
        
        x = x + self.drop_path(self.attn(norm_x))
        
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.batch_size = BATCHS

    def forward(self, x):
#         B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
#         x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x).reshape(self.batch_size,192,196,1).transpose(1,2)
        x = x.squeeze(3)
        
        return x


# class HybridEmbed(nn.Module):
#     """ CNN Feature Map Embedding
#     Extract feature map from CNN, flatten, project to embedding dim.
#     """
#     def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
#         super().__init__()
#         assert isinstance(backbone, nn.Module)
#         img_size = to_2tuple(img_size)
#         self.img_size = img_size
#         self.backbone = backbone
#         if feature_size is None:
#             with torch.no_grad():
#                 # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
#                 # map for all networks, the feature metadata has reliable channel and stride info, but using
#                 # stride to calc feature dim requires info about padding of each stage that isn't captured.
#                 training = backbone.training
#                 if training:
#                     backbone.eval()
#                 o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
#                 feature_size = o.shape[-2:]
#                 feature_dim = o.shape[1]
#                 backbone.train(training)
#         else:
#             feature_size = to_2tuple(feature_size)
#             feature_dim = self.backbone.feature_info.channels()[-1]
#         self.num_patches = feature_size[0] * feature_size[1]
#         self.proj = nn.Linear(feature_dim, embed_dim)

#     def forward(self, x):
#         x = self.backbone(x)[-1]
#         x = x.flatten(2).transpose(1, 2)
#         x = self.proj(x)
#         return x



class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # I add these two lines
        self.drop_rate=drop_rate
        attn_drop_rate=drop_rate
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.depth = depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.batch_size = BATCHS

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
 
        x = self.patch_embed(x)
#         B = x.shape[0]
        B = BATCHS
    

#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B,-1,-1)
        
#         x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def split_convs(self, x):
        x = self.patch_embed(x)
        
#       B = x.shape[0]
        B = BATCHS

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
#         x = torch.cat((self.cls_token, x), dim=1)
    
        x = x + self.pos_embed
        
        x = self.pos_drop(x)
        
#         num_blocks = len(self.blocks) // 2
#         for blk in self.blocks[:num_blocks]:
#             x = blk(x)

#         x = self.blocks[0].half_block(x)

#         print("split_1, output.shape:",x.shape, "\t num_blocks:",num_blocks)

        return x


    def split_2(self,x):
        x = self.patch_embed(x)
        B = BATCHS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_embed  
        x = self.pos_drop(x)
        x = self.blocks[0].half_block(x)
        return x
    
    def split_n(self,x,n,half=None):
        x = self.patch_embed(x)
        B = BATCHS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_embed  
        x = self.pos_drop(x)
        
        for layer_idx in range(n):
            x = self.blocks[layer_idx](x)
            
        #n-th layer
        if half:
            x = self.blocks[n].half_block(x)
        else:
            x = self.blocks[n](x)
            # last layer of transformer
            if n == (self.depth - 1):
                x = self.norm(x)
                x = x[:, 0]
                x = self.head(x)
                
        return x
            
          
    def update_drop_path(self, drop_path_rate):
        self.drop_path = drop_path_rate
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        for i in range(self.depth):
            self.blocks[i].drop_path.drop_prob = dp_rates[i]
    
    def update_dropout(self, drop_rate):
        self.drop_rate = drop_rate
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

# @register_model
# def vit_tiny_tiny(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=48, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

@register_model
def vit_tiny(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def vit_small(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def vit_base(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def vit_large(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.layers.neuralteleportation import COBForwardMixin, FlattenCOB
from neuralteleportation.layers.neuron import LinearCOB
from neuralteleportation.layers.activation import ReLUCOB, SigmoidCOB, GELUCOB, LeakyReLUCOB
from neuralteleportation.layers.dropout import DropoutCOB
from neuralteleportation.layers.neuron import LayerNormCOB
from neuralteleportation.layers.merge import Add

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        # self.add = Add()
        self.norm2 = LayerNormCOB(192)
        self.fc1 = LinearCOB(192, 768, bias=True)
        self.act = GELUCOB()
        self.fc2 = LinearCOB(768, 192, bias=True)
        # self.drop1 = DropoutCOB(0.1)
        # self.drop2 = DropoutCOB(0.1)

    def forward(self, x):
        x1 = self.norm2(x)
        x2 = self.fc1(x1)
        x3 = self.act(x2)
        x4 = self.fc2(x3)
        # x4 = self.add(x, x3)
        
        return x4


def load_ln_weights(LN, model, block_idx):
    original_mlp = model.blocks[block_idx].mlp
    original_norm2 = model.blocks[block_idx].norm2

    combined_dict = {}
    combined_dict.update(original_mlp.state_dict())
    for k,v in original_norm2.state_dict().items():
        combined_dict["norm2." + k] = v
    LN.network.load_state_dict(combined_dict)


# define global variable to store the best loss found
global best_loss
global cor_best_pred_error
global cor_best_range
# initialize best_loss
best_loss = 1e9




import copy
import torch
import torch.nn as nn
import numpy as np
import random
import functools
import torch.multiprocessing as mp

# Activation hook for storing activation statistics
def activation_hook(name, activation_stats, activations_output=None, layer_idx=None):
    def hook(module, input, output):
        input_tensor = input[0]
        activation_stats[name] = {'min': input_tensor.min().item(), 'max': input_tensor.max().item()}
        if activations_output is not None:
            # key of the dict is the layer number (extracted from the name)
            activations_output[layer_idx] = output
    return hook

# The function that calculates loss based on COB and runs inference
def f_ack(cob, input_data=None, original_pred=None, layer_idx=None, original_loss=None, tm=None, activation_orig=None, grad_orig=None, hessian_sensitivity=False):    
    # Set up model with the new COB
    teleported_model = tm
    # Apply the COB
    teleported_model = teleported_model.teleport(cob, reset_teleportation=False)

    # Reset activation stats and run a forward pass
    activation_stats = {}
    activations_quant = {}

    hook_handles = []
    for i, layer in enumerate(teleported_model.network.children()):
        if isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.GELU, nn.LeakyReLU, ReLUCOB, SigmoidCOB, GELUCOB, LeakyReLUCOB)):
            handle = layer.register_forward_hook(activation_hook(f'relu_{i}', activation_stats=activation_stats, activations_output=activations_quant,layer_idx=layer_idx))
            hook_handles.append(handle)

    teleported_model.eval()
    with torch.no_grad():
        pred = teleported_model.network(input_data)

    for handle in hook_handles:
        handle.remove()

    # Calculate the range loss
    loss = sum([stats['max'] - stats['min'] for stats in activation_stats.values()])
    loss /= original_loss
    # Calculate the prediction error
    # pred_error = np.abs(original_pred - pred.detach().cpu().numpy()).mean()

    if hessian_sensitivity:
        pred_error = 0.0
        activation_quant = activations_quant[layer_idx]
        # assert that the activation_quant has only one key
        assert len(activations_quant.keys()) == 1
        # print("debug - key: ", activation_quant.shape)
        # Compute the difference between activations
        delta = activation_quant - activation_orig
        # Compute the squared gradients
        grad_squared = grad_orig.pow(2)
        # Compute the element-wise product
        elementwise_product = delta.pow(2) * grad_squared
        # Sum over all elements to get the loss for this layer
        # print("debug - elementwise_product: ", elementwise_product.shape)
        layer_loss = elementwise_product.sum()
        # Accumulate the total prediction error
        pred_error += layer_loss.item()
        if random.random() < 0.0005:
            print(f"debug - grad_squared_norm: {grad_squared.norm()} \t delta2_norm: {delta.pow(2).norm()} \t layer_loss: {layer_loss}")
    else:
        pred_error = np.abs(original_pred - pred).mean()
        pred_error /= np.abs(original_pred).mean()
        pred_error = pred_error.item()
    # TODO: change the 10 with args.pred_mul (adding that to the function signature)
    total_loss = loss + 10 * pred_error

    # if random.random() < 0.0005:
    #     print(f"pred_error: {pred_error} \t range_loss: {loss}")

    # Undo the teleportation
    teleported_model.undo_teleportation()
    return total_loss, loss, pred_error

# # Worker function for parallel gradient computation
# def worker_func(args):
#     idx, key, base, params_dict, step_size, func = args
#     perturbed_params_dict = copy.deepcopy(params_dict)
#     p_flat = perturbed_params_dict[key].flatten()
#     p_flat[idx] += step_size
#     directional_derivative = (func(perturbed_params_dict["cob"]) - base) / step_size
#     return idx, directional_derivative

def worker_func_batch(args):
    idx_batch, key, base, params_dict, step_size, func = args
    perturbed_params_dict = copy.deepcopy(params_dict)
    # perturbed_params_dict = params_dict
    p_flat = perturbed_params_dict[key].flatten()
    grads = []
    
    # Compute gradients for each perturbation in the batch
    for idx in idx_batch:
        p_flat[idx] += step_size
        out,_,_ = func(perturbed_params_dict["cob"])
        directional_derivative = (out - base) / step_size
        grads.append((idx, directional_derivative))
        p_flat[idx] -= step_size  # Reset the perturbation

    return grads

# # Parallelized CGE implementation
# @torch.no_grad()
# def cge_parallel(func, params_dict, mask_dict, step_size, base=None, num_workers=4):
#     if base is None:
#         base = func(params_dict["cob"])

#     grads_dict = {}
#     for key, param in params_dict.items():
#         if 'orig' in key:
#             mask_key = key.replace('orig', 'mask')
#             mask_flat = mask_dict[mask_key].flatten()
#         else:
#             mask_flat = torch.ones_like(param).flatten()
        
#         directional_derivative = torch.zeros_like(param)
#         directional_derivative_flat = directional_derivative.flatten()

#         # Prepare the tasks for each index to be processed in parallel
#         tasks = [(idx.item(), key, base, params_dict, step_size, func) for idx in mask_flat.nonzero()]

#         # Create a pool of workers and distribute tasks
#         with mp.Pool(num_workers) as pool:
#             results = pool.map(worker_func, tasks)

#         # Collect the results and fill the directional_derivative tensor
#         for idx, grad in results:
#             directional_derivative_flat[idx] = grad

#         grads_dict[key] = directional_derivative.to(param.device)
    
#     return grads_dict

# Batched CGE using multiprocessing
@torch.no_grad()
def cge_batched(func, params_dict, mask_dict, step_size, pool, base=None, num_process=4):
    if base is None:
        base,_,_ = func(params_dict["cob"])

    grads_dict = {}
    for key, param in params_dict.items():
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten()
        else:
            mask_flat = torch.ones_like(param).flatten()

        directional_derivative = torch.zeros_like(param)
        directional_derivative_flat = directional_derivative.flatten()

        # set 50 percent of the mask to zero
        mask_flat = mask_flat * torch.bernoulli(torch.full_like(mask_flat, 0.5))

        # Prepare batches of indices
        idx_list = mask_flat.nonzero().flatten().tolist()
        # batch_size = len(idx_list) // num_process
        # check whether the batch size is dividable by the number of processes to make sure that each process gets the same number of indices
        if len(idx_list) % num_process != 0:
            batch_size = len(idx_list) // num_process + 1
        else:
            batch_size = len(idx_list) // num_process
        batches = [idx_list[i:i + batch_size] for i in range(0, len(idx_list), batch_size)]
        
        # Create task arguments for each batch
        tasks = [(batch, key, base, params_dict, step_size, func) for batch in batches]

        # Use the already initialized pool to run the worker_func_batch in parallel
        results = pool.map(worker_func_batch, tasks)

        # Collect results from all workers and update the directional_derivative tensor
        for result_batch in results:
            for idx, grad in result_batch:
                directional_derivative_flat[idx] = grad

        grads_dict[key] = directional_derivative.to(param.device)

    return grads_dict



# # Training loop implementation with parallelized gradient computation
# def train_cob(input_teleported_model, original_pred, layer_idx, original_loss_idx, LN, args):
#     initial_cob_idx = torch.ones(960)  # Initial guess for COB

#     # Prepare the function for constrained gradient estimation
#     ackley = functools.partial(
#         f_ack,
#         input_data=input_teleported_model,
#         original_pred=original_pred,
#         layer_idx=layer_idx,
#         original_loss=original_loss_idx,
#         tm=LN
#     )

#     best_cob = None
#     best_loss = float('inf')

#     # Training loop to optimize COB
#     for step in range(args.steps):
#         # Get the gradient of the COB using the parallelized CGE
#         grad_cob = cge_parallel(ackley, {"cob": initial_cob_idx}, None, args.zoo_step_size, num_workers=3)

#         # Update the COB using gradient descent
#         initial_cob_idx -= args.cob_lr * grad_cob["cob"]

#         # Calculate the loss with the updated COB
#         loss = ackley(initial_cob_idx)

#         # Update the best loss and COB if the current loss is better
#         if loss < best_loss:
#             best_loss = loss
#             best_cob = initial_cob_idx.clone()  # Save the best COB

#         print(f"Step: {step} \t Loss: {loss}")

#     return best_cob

import time
# Training loop using the persistent pool
def train_cob(input_teleported_model,input_orig_model, original_pred, layer_idx, original_loss_idx, LN, args, activation_orig = None, grad_orig = None):
    initial_cob_idx = torch.ones(960)  # Initial guess for COB

    # Prepare the function for constrained gradient estimation
    ackley = functools.partial(
        f_ack,
        input_data=input_orig_model,
        original_pred=original_pred,
        layer_idx=layer_idx,
        original_loss=original_loss_idx,
        tm=LN,
        activation_orig = activation_orig,
        grad_orig = grad_orig,
        hessian_sensitivity = args.hessian_sensitivity
    )

    eval_ackley = functools.partial(
        f_ack,
        input_data=input_teleported_model,
        original_pred=original_pred,
        layer_idx=layer_idx,
        original_loss=original_loss_idx,
        tm=LN,
        hessian_sensitivity = args.hessian_sensitivity
    )

    best_cob = None
    # best_loss = float('inf')
    num_process = 2

    with mp.Manager() as manager:
        best_loss = manager.Value('d', float('inf'))  # Shared float variable for the best loss
        cor_best_pred_error = manager.Value('d', 0.0)  # Shared float variable for prediction error
        cor_best_range = manager.Value('d', 0.0)  # Shared float variable fo

        # compute the loss before the optimization
        loss,r_error,p_error = ackley(initial_cob_idx)
        print(f"Initial Loss: {loss} \t P_E: {p_error} \t R_E: {r_error}")

        # Initialize the process pool once and reuse it for all iterations
        with mp.Pool(num_process) as pool:
            # Training loop to optimize COB
            for step in range(args.steps):
                # Get the gradient of the COB using the batched CGE with persistent pool
                t0 = time.time()
                grad_cob = cge_batched(ackley, {"cob": initial_cob_idx}, None, args.zoo_step_size, pool, num_process=num_process)
                t1 = time.time()
                # Update the COB using gradient descent
                if not args.hessian_sensitivity:
                    initial_cob_idx -= args.cob_lr * grad_cob["cob"]
                else:
                    # normalize the gradient
                    update = grad_cob["cob"] / grad_cob["cob"].norm()
                    initial_cob_idx -= args.cob_lr * update
                t2 = time.time()

                # Calculate the loss with the updated COB
                loss,r_error,p_error = eval_ackley(initial_cob_idx)
                t3 = time.time()

                # Update the best loss and COB if the current loss is better
                if loss < best_loss.value:
                    best_loss.value = loss
                    best_cob = initial_cob_idx.clone()  # Save the best COB
                    cor_best_pred_error.value = p_error
                    cor_best_range.value = r_error

                    # print(f"Step: {step} \t Loss: {loss}")

                print(f"Step: {step} \t Loss: {loss} \t \t P_E: {p_error} \t R_E: {r_error} \t  \t Time: {t1-t0}")
            

        return best_cob, best_loss.value, cor_best_range.value, cor_best_pred_error.value




if __name__ == '__main__':
    # set spawn start method
    mp.set_start_method('spawn', force=True)
    args = get_default_args()
    print(args.__dict__)

    default_model = "vit_tiny"
    default_data_path = "/rds/general/user/mm6322/home/imagenet"
    default_resume = "./sparse-cap-acc-tmp/deit_tiny_patch16_224_sparsity=0.50_best.pth"
    # default_resume = "/rds/general/user/mm6322/home/verifiable_NN_ezkl/examples/notebooks/CAP_pruned_models/Checkpoints/deit_tiny_patch16_224_sparsity=0.50_best.pth"
    default_sparsity = 0.5
    default_batch_size = 1
    default_pruning_method = "CAP"
    default_prefix_dir = "sparse-cap-acc-tmp/"
    default_input_param_scale = 7
    default_log_rows = 20
    default_num_cols = 2
    default_scale_rebase_multiplier = 1
    defualt_hessian_sensitivity = False

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--model', default=default_model, type=str, help='Model type')
    # parser.add_argument('--data_path', default=default_data_path, type=str, help='Data path')
    # parser.add_argument('--resume', default=default_resume, type=str, help='Resume path')
    # parser.add_argument('--sparsity', default=default_sparsity, type=float, help='Sparsity value')
    # parser.add_argument('--batch_size', default=default_batch_size, type=int, help='Batch size')
    # parser.add_argument('--pruning_method', default=default_pruning_method, type=str, help='Pruning method')
    # parser.add_argument('--prefix_dir', default=default_prefix_dir, type=str, help='Prefix directory')
    # parser.add_argument('--input_param_scale', default=default_input_param_scale, type=int, help='Input parameter scale')
    # parser.add_argument('--log_rows', default=default_log_rows, type=int, help='Log rows')
    # parser.add_argument('--num_cols', default=default_num_cols, type=int, help='Number of columns')
    # parser.add_argument('--scale_rebase_multiplier', default=default_scale_rebase_multiplier, type=int, help='Scale rebase multiplier')
    # args = parser.parse_args(namespace=args)
    args.model = default_model
    args.data_path = default_data_path
    args.resume = default_resume
    args.sparsity = default_sparsity
    args.batch_size = default_batch_size
    args.pruning_method = default_pruning_method
    args.prefix_dir = default_prefix_dir
    args.input_param_scale = default_input_param_scale
    args.log_rows = default_log_rows
    args.num_cols = default_num_cols
    args.scale_rebase_multiplier = default_scale_rebase_multiplier
    args.hessian_sensitivity = defualt_hessian_sensitivity

    os.makedirs(args.prefix_dir, exist_ok=True)

    BATCHS = args.batch_size

    model = build_model(args, pretrained=False)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    total_batch_size = args.batch_size * args.update_freq
    # num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    # At most one of dropout and stochastic depth should be enabled.
    assert(args.dropout == 0 or args.drop_path == 0)
    # ConvNeXt does not support dropout.
    assert(args.dropout == 0 if args.model.startswith("convnext") else True)

    import re

    if "convnext" in args.model:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    elif "vit" in args.model:
        print("loading ...")
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        if args.pruning_method == "CAP":
            load_state_dict(model, checkpoint["state_dict"], prefix='', ignore_missing="relative_position_index")
        elif args.pruning_method == "DENSE":
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    elif "deit" in args.model:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint["model"])

    # compute the number of zeros in the model / total number of parameters
    total_params = 0
    zeros = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        zeros += (param.data == 0).sum().item()

    print(f"Total number of parameters: {total_params}")
    print(f"Number of zeros: {zeros}")
    print(f"Percentage of zeros: {zeros / total_params * 100:.2f}%")

    model = model.cpu()
    model.eval()

    if not args.hessian_sensitivity:
        for param in model.parameters():
            param.requires_grad_(False)

    import PIL.Image
    # load JPEG image /Users/mm6322/Phd research/nerual_transport/neuralteleportation/neuralteleportation/experiments/sparse-cap-acc-tmp/ILSVRC2012_val_00000616.JPEG
    from PIL import Image
    from torchvision import datasets, transforms

    img = Image.open("./sparse-cap-acc-tmp/ILSVRC2012_val_00000616.JPEG")
    img_name = "ILSVRC2012_val_00000616.JPEG"
    # img = Image.open("/rds/general/user/mm6322/home/imagenet/val/n12620546/ILSVRC2012_val_00011901.JPEG")

    img = img.resize((224,224))
    data = transforms.ToTensor()(img).unsqueeze(0)
    print(data.shape)
    print("data shape:",data.shape)

    # compute the output of the model
    result = model(data)
    network_input_data = copy.deepcopy(data)

    import json
    x = data.detach().clone()
    print("x.shape:",x.shape)

    onnx_path = args.prefix_dir + "network_complete.onnx" if args.pruning_method == "CAP" else args.prefix_dir + f"network_complete_{args.pruning_method}.onnx"

    # Export the model
    torch.onnx.export(    
        model,               # model being run
        x,                   # model input (or a tuple for multiple inputs)
        onnx_path,            # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=15,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                        'output': {0:'batch_size'},
        },         
    )

    inter_out = model.split_convs(data)
    print("layer_idx:\t","CONV","\t half:",str(False),"\t inter_out.shape:",inter_out.shape,"\t min:",inter_out.min(),"\t max:",inter_out.max())
    for layer_idx in range(model.depth):
        for half in [True,False]:
            inter_out = model.split_n(data,layer_idx,half)
            print("layer_idx:\t",layer_idx,"\t half:",str(half),"\t inter_out.shape:",inter_out.shape,"\t min:",inter_out.min(),"\t max:",inter_out.max())

    import onnx
    on = onnx.load(onnx_path)
    for tensor in on.graph.input:
        for dim_proto in tensor.type.tensor_type.shape.dim:
            print("dim_proto:",dim_proto)
            if dim_proto.HasField("dim_param"): # and dim_proto.dim_param == 'batch_size':
                dim_proto.Clear()
                dim_proto.dim_value = BATCHS   # fixed batch size
    for tensor in on.graph.output:
        for dim_proto in tensor.type.tensor_type.shape.dim:
            if dim_proto.HasField("dim_param"):
                dim_proto.Clear()
                dim_proto.dim_value = BATCHS   # fixed batch size
    onnx.save(on, onnx_path)
    on = onnx.load(onnx_path)
    on = onnx.shape_inference.infer_shapes(on)
    onnx.save(on, onnx_path)

    # generate data for all layers
    data_path = os.path.join(os.getcwd(),args.prefix_dir, "input_convs.json")
    data = dict(input_data = [((x).detach().numpy()).reshape([-1]).tolist()])
    json.dump( data, open(data_path, 'w' ))

    for i in range(model.depth):
        for half in [True,False]:
            inter_i = model.split_n(x,i,half=half)
            data_path = os.path.join(os.getcwd(),args.prefix_dir, f"input_{i}_{str(half)}.json")
            data = dict(input_data = [((inter_i).detach().numpy()).reshape([-1]).tolist()])
            json.dump( data, open(data_path, 'w' ))

    import onnx
    # extract all onnx files of layers

    # input_path = args.prefix_dir + "network_complete.onnx"
    input_path = onnx_path

    # Convs layer
    # output_path = args.prefix_dir + "network_split_convs.onnx"
    output_path = args.prefix_dir + "network_split_convs.onnx" if args.pruning_method == "CAP" else args.prefix_dir + f"network_split_convs_{args.pruning_method}.onnx"
    input_names = ["input"]
    output_names = ["/Add_output_0"]
    onnx.utils.extract_model(input_path, output_path, input_names, output_names, check_model=True)
    input_names = output_names

    for layer_idx in range(model.depth):
        for half in [True,False]:        
            # output_path = f"{args.prefix_dir}network_split_{layer_idx}_{str(half)}.onnx"
            output_path = args.prefix_dir + f"network_split_{layer_idx}_{str(half)}.onnx" if args.pruning_method == "CAP" else args.prefix_dir + f"network_split_{layer_idx}_{str(half)}_{args.pruning_method}.onnx"
            if half:
                output_names = [f"/blocks.{layer_idx}/Add_2_output_0"]
            else:
                output_names = [f"/blocks.{layer_idx}/Add_3_output_0"]
                if layer_idx == (model.depth - 1):
                    output_names = ["output"]
            print("layer_idx:",layer_idx,"\t half:",str(half),"\t input_names:",input_names,"\t output_names:",output_names)
            onnx.utils.extract_model(input_path, output_path, input_names, output_names,check_model=True)
            input_names = output_names


    # Define the CSV file and write the header if it doesn't exist
    import csv

    csv_file_path = args.prefix_dir + 'ZO-accuracy.csv'
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'EXPERIMENT SETTINGS',
                'Layer_Index', 'Sample_Name',
                'Activation_Loss_Org', 'Range_Error' ,'Prediction_Error',
        ])
            
    array_param_visibility = ["fixed"]
    array_input_param_scale = [16]
    array_num_cols = [64]
    array_max_log_rows = [16]
    array_scale_rebase = [1]
    array_lookup_margin = [2]

    # iterate over all the possible combinations
    combinations = list(itertools.product(array_param_visibility, array_input_param_scale, array_num_cols, array_max_log_rows, array_scale_rebase, array_lookup_margin))
    # the layer_idx which the logrows are equal to 20 are not gonna be teleported
    # list_of_no_teleportation = [1,3,4,5]
    if not args.hessian_sensitivity:
        # list_of_no_teleportation = [1,3,4,5]
        list_of_no_teleportation = None
        # select the three least values of the range_list_all
    else:
        list_of_no_teleportation = []
    # list_jpeg = ["ILSVRC2012_val_00000616.JPEG"]
    # list_jpeg is all jpeg file in the folder images
    list_jpeg = os.listdir("./sparse-cap-acc-tmp/images")
    list_jpeg = [x for x in list_jpeg if x.endswith(".JPEG")]

    # computing accuracy of the teleportation
    teleport_correct = 0
    teleport_total = 0

    # with no gradient pytorch
    # with torch.no_grad():
    if True:
        # iterate over all the possible combinations
        for p in combinations:
            param_visibility, input_param_scale, num_cols, max_log_rows, scale_rebase, lookup_margin = p
            # string experiment_settings as comma separated values
            experiment_settings = f"{param_visibility}/{input_param_scale}/{num_cols}/{max_log_rows}/{scale_rebase}/{lookup_margin}"
            print("========= START =========")
            print(f"input_param_scale: {input_param_scale}, num_cols: {num_cols}, max_log_rows: {max_log_rows}, param_visibility: {param_visibility}, lookup_margin: {lookup_margin}")

            # copy the model
            new_model = copy.deepcopy(model)
            model.eval()
            new_model.eval()
            
            # generate compression-model and setting for all images among all layers
            list_jpeg = list(reversed(list_jpeg))
            # set only one image in the list
            list_jpeg = list_jpeg[:1]

            for jpeg_path in list_jpeg:
                print("========== jpeg_path:",jpeg_path, " ==========")
                img = Image.open(f"./sparse-cap-acc-tmp/images/{jpeg_path}")
                img_name = os.path.splitext(jpeg_path)[0]
                img = img.resize((224,224))
                data = transforms.ToTensor()(img).unsqueeze(0)

                # compute the output_orig and grad_orig dictionaries
                activations_orig = None
                gradients_orig = None

                if args.hessian_sensitivity:
                    # Define a function to create a forward hook that captures the layer index
                    model.train()
                    activations_orig = {}
                    gradients_orig = {}

                    def get_forward_hook(layer_idx):
                        def forward_hook_orig(module, input, output):
                            # Ensure the output requires gradients
                            output.requires_grad_(True)
                            activations_orig[layer_idx] = output
                        return forward_hook_orig

                    # Register forward hooks on the original model's activation layers
                    hook_handles = []
                    for layer_idx, block_orig in enumerate(model.blocks):
                        mlp_orig = block_orig.mlp
                        for layer_orig in mlp_orig.children():
                            if isinstance(layer_orig, (nn.ReLU, nn.Sigmoid, nn.GELU, nn.LeakyReLU)):
                                # Register a forward hook with the layer index
                                handle_orig = layer_orig.register_forward_hook(get_forward_hook(layer_idx))
                                hook_handles.append(handle_orig)
                    # Perform a forward pass through the original model
                    result = model(data)
                    # Define the loss function (assuming you have the true labels)
                    criterion = nn.CrossEntropyLoss()
                    # TODO: # Replace 'labels' with your actual target tensor
                    targets = result.argmax(dim=1)
                    # Compute the loss using the original model's output
                    loss_cls = criterion(result, targets)

                    # Compute the gradients w.r.t. the original model's activations
                    for layer_idx in activations_orig:
                        grad = torch.autograd.grad(loss_cls, activations_orig[layer_idx], retain_graph=True)[0]
                        gradients_orig[layer_idx] = grad.detach().cpu()
                    # detach all output stored in the dictionary
                    activations_orig = {key: value.detach().cpu() for key, value in activations_orig.items()}

                    # Print the shapes of the gradients and activations for verification
                    for layer_idx in gradients_orig:
                        print(f"Layer {layer_idx}: Gradient shape {gradients_orig[layer_idx].shape}, Activation shape {activations_orig[layer_idx].shape}")
                    # Disable gradients for all parameters of the original model
                    for param in model.parameters():
                        param.requires_grad_(False)
                    # Remove the hooks to prevent side effects
                    for handle in hook_handles:
                        handle.remove()
                    # print shape of the grad and output in the dictionaries
                    for key in gradients_orig:
                        print("key:",key,"\t grad.shape:",gradients_orig[key].shape,"\t output.shape:",activations_orig[key].shape)
                    model.eval()
                else:
                    with torch.no_grad():
                        # compute the range of the activations of the model (for all layers)
                        # 1. set the hooks
                        hook_handles = []
                        activation_stats_all = {}
                        for layer_idx, block in enumerate(model.blocks):
                            mlp = block.mlp
                            for layer in mlp.children():
                                if isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.GELU, nn.LeakyReLU)):
                                    handle = layer.register_forward_hook(activation_hook(f'relu_{layer_idx}', activation_stats=activation_stats_all, layer_idx=layer_idx))
                                    hook_handles.append(handle)

                        # inference on original model
                        result = model(data)

                        # 2. finding the range based on the hooks
                        range_list_all = {i : activation_stats_all[f'relu_{i}']['max'] - activation_stats_all[f'relu_{i}']['min'] for i in range(model.depth)}
                        print("range_list_all:",range_list_all)

                        # 3. remove the hooks
                        for handle in hook_handles:
                            handle.remove()
                            
                        # 4. define list of no teleportation
                        topk = 3
                        list_of_no_teleportation = [k for k, v in sorted(range_list_all.items(), key=lambda item: item[1])[:topk]]
                        print("list_of_no_teleportation:",list_of_no_teleportation)
                       
                # inference on original model
                network_input_data = copy.deepcopy(data)

                # generate data for all layers
                data_path = os.path.join(os.getcwd(),args.prefix_dir, f"input_convs.json")
                data_dict = dict(input_data = [((data).detach().numpy()).reshape([-1]).tolist()])
                json.dump( data_dict, open(data_path, 'w' ))

                with torch.no_grad():
                    for layer_idx in [0,1,2,3,4,5,6,7,8,9,10,11]:  # Parallelize this loop since it's independent for each layer
                        args.pred_mul = 10
                        args.steps = 200
                        args.cob_lr = 0.2
                        args.zoo_step_size = 0.0005

                        # Hook for the intermediate output of the block
                        hook_handles = []
                        original_mlp_idx = model.blocks[layer_idx].mlp
                        activation_stats_idx = {}
                        for i,layer in enumerate(original_mlp_idx.children()):
                            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.GELU) or isinstance(layer, nn.LeakyReLU):
                                handle = layer.register_forward_hook(activation_hook(f'relu_{i}', activation_stats=activation_stats_idx))
                                hook_handles.append(handle)
                        # run the mlp model to find original_loss
                        # create input_convs based on 
                        input_convs = json.load(open(args.prefix_dir + "input_convs.json"))["input_data"][0]
                        _ = model.split_n(torch.tensor(input_convs).view(BATCHS,3,224,224),layer_idx,half=False)
                        # original_block_idx_pred = model(data)

                        for handle in hook_handles:
                            handle.remove()
 
                        # print activation stats
                        print(f"layer_idx: {layer_idx} , \t  activation_stats: {activation_stats_idx}")
                        original_loss_idx = sum([stats['max'] - stats['min'] for stats in activation_stats_idx.values()])
                        print("ORIGINAL LOSS:",original_loss_idx)

                        # track best loss
                        best_loss = 1e9
                        cor_best_pred_error = 1e9
                        cor_best_range = 1e9

                        # define input_teleported_model (used in ng_loss_function)
                        input_convs = json.load(open(args.prefix_dir + "input_convs.json"))["input_data"][0]
                        input_convs = torch.tensor(input_convs).view(1,3,224,224)
                        input_teleported_model = new_model.split_n(input_convs,layer_idx,half=True)

                        # save npy file using in python checking script
                        np.save(args.prefix_dir + f"input_teleported_model_{layer_idx}.npy", input_teleported_model.detach().numpy())
                        # define original_pred (used in ng_loss_function)
                        input_org = model.split_n(input_convs,layer_idx,half=True)

                        print("debug1: diffrence between input_teleported_model and input_org:",torch.norm(input_teleported_model - input_org),"\t layer_idx:",layer_idx)
                        np.save(args.prefix_dir + f"input_org_{layer_idx}.npy", input_org.detach().numpy())
                        original_pred = model.blocks[layer_idx].mlp(model.blocks[layer_idx].norm2(input_org))

                        # Apply best COB and save model weights
                        LN = LinearNet()
                        LN = NeuralTeleportationModel(LN, input_shape=(1, 197, 192))
                        load_ln_weights(LN, model, layer_idx)
                        LN.eval()

                        if layer_idx in list_of_no_teleportation:
                            print("====== NO OPTIMIZATION SINCE NO TELEPORTATION =====")
                            best_loss = torch.tensor(best_loss).detach().cpu()
                            LN = LN.teleport(torch.ones_like(torch.ones(960)), reset_teleportation=True)
                            # torch.save(LN.network.state_dict(), args.prefix_dir + f'block{layer_idx}_cob_activation_norm_teleported.pth')
                            save_path = args.prefix_dir + f'block{layer_idx}_cob_activation_norm_teleported.pth' if args.pruning_method == "CAP" else args.prefix_dir + f'block{layer_idx}_cob_activation_norm_teleported_{args.pruning_method}.pth'
                            torch.save(LN.network.state_dict(), save_path)
                            cor_best_range = 1
                            cor_best_pred_error = 0
                        # check whether the teleportation .pth already exists
                        # elif os.path.exists(args.prefix_dir + f'block{layer_idx}_cob_activation_norm_teleported.pth'):
                        #     print(f"block{layer_idx}_cob_activation_norm_teleported.pth already exists.")
                        #     LN.network.load_state_dict(torch.load(args.prefix_dir + f'block{layer_idx}_cob_activation_norm_teleported.pth'))
                        #     best_loss = torch.tensor(best_loss).detach().cpu()
                        else:
                            act_idx = activations_orig[layer_idx] if activations_orig is not None else None
                            grad_idx = gradients_orig[layer_idx] if gradients_orig is not None else None
                            # best_cob,best_loss,cor_best_range,cor_best_pred_error = train_cob(input_teleported_model, original_pred, layer_idx, original_loss_idx, LN, args, activation_orig=act_idx, grad_orig=grad_idx)
                            # change the input of the train_cob to input_orig in order to run the all blocks teleportation in parallel
                            best_cob,best_loss,cor_best_range,cor_best_pred_error = train_cob(input_teleported_model, input_org, original_pred, layer_idx, original_loss_idx, LN, args, activation_orig=act_idx, grad_orig=grad_idx)
                            print("BEST LOSS:",best_loss)
                            LN = LN.teleport(best_cob, reset_teleportation=True)
                            # save the .pth of the teleported model
                            # torch.save(LN.network.state_dict(), args.prefix_dir + f'block{layer_idx}_cob_activation_norm_teleported.pth')
                            save_path = args.prefix_dir + f'block{layer_idx}_cob_activation_norm_teleported.pth' if args.pruning_method == "CAP" else args.prefix_dir + f'block{layer_idx}_cob_activation_norm_teleported_{args.pruning_method}.pth'
                            torch.save(LN.network.state_dict(), save_path)
                        
                        # Apply the teleportation to the new_model (Using for computing the next layer inputs)
                        sd = LN.network.state_dict()
                        sd = {k: v for k, v in sd.items() if 'norm2' not in k}
                        new_model.blocks[layer_idx].mlp.load_state_dict(sd)
                        sd = LN.network.state_dict()
                        sd = {k.replace('norm2.',''): v for k, v in sd.items() if 'norm2' in k}
                        new_model.blocks[layer_idx].norm2.load_state_dict(sd)

                        # TODO: uncomment if your plan didn't work
                        # # Export the optimized model to ONNX
                        # onnx_path = args.prefix_dir + f'block{layer_idx}_cob_activation_norm_teleported.onnx' if args.pruning_method == "CAP" else args.prefix_dir + f'block{layer_idx}_cob_activation_norm_teleported_{args.pruning_method}.onnx'
                        # torch.onnx.export(LN.network, input_teleported_model, onnx_path, verbose=False, export_params=True, opset_version=15, do_constant_folding=True, input_names=['input_0'], output_names=['output'])

                        # TODO: uncomment if your plan didn't work
                        # # check the validation of the teleportation
                        # # 1.extract onnx corrosponding to the teleported model (in original onnx)
                        # input_path = args.prefix_dir + f"network_split_{layer_idx}_False.onnx" if args.pruning_method == "CAP" else args.prefix_dir + f"network_split_{layer_idx}_False_{args.pruning_method}.onnx"
                        # output_path = args.prefix_dir + f"block{layer_idx}_cob_activation_norm.onnx" if args.pruning_method == "CAP" else args.prefix_dir + f"block{layer_idx}_cob_activation_norm_{args.pruning_method}.onnx"
                        # input_names = [f"/blocks.{layer_idx}/Add_2_output_0"]
                        # output_names = [f"/blocks.{layer_idx}/mlp/fc2/Add_output_0"]
                        # onnx.utils.extract_model(input_path, output_path, input_names, output_names, check_model=True)
                    
                        # write the results to the csv file
                        with open(csv_file_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                experiment_settings,
                                layer_idx, img_name,
                                original_loss_idx, cor_best_range, cor_best_pred_error,
                            ])
                        
                        print("\n\n")

                    # compute the new prediction of model (after all layers are teleported)
                    new_pred = new_model(network_input_data)
                    # check whether the teleportation is successful
                    print("MAX ARGUMENT NEW PREDICTION:", new_pred.argmax())
                    print("ORIGINAL PREDICTION:", result.argmax())
                    if new_pred.argmax() == result.argmax():
                        teleport_correct += 1
                    teleport_total += 1
                    norm1 = torch.norm(new_pred - result)
                    print("ACCURACY OF TELEPORTATION:", teleport_correct/teleport_total, "L1-DIFFERENCE:", norm1)
                    # log on the file txt (append the accuracy of teleportation + number of corrot and total)
                    with open(args.prefix_dir + "accuracy_teleportation.txt", "a") as f:
                        f.write(f"ACCURACY OF TELEPORTATION: {teleport_correct/teleport_total} \t CORRECT: {teleport_correct} \t TOTAL: {teleport_total} \t NORM1: {norm1}\n")
                    print("==========================")
                    time.sleep(2)

                    # export onnx for each split layer of the model
                    # Convs layer
                    output_path = args.prefix_dir + "network_split_convs.onnx" if args.pruning_method == "CAP" else args.prefix_dir + f"network_split_convs_{args.pruning_method}.onnx"
                    input_names = ["input"]
                    output_names = ["/Add_output_0"]
                    onnx.utils.extract_model(input_path, output_path, input_names, output_names, check_model=True)
                    input_names = output_names
                    print("====== EXPORT ONNX OF SPLITED NOT TELEPORTED LAYERS ======")
                    for layer_idx in range(model.depth):
                        for half in [True,False]:        
                            # output_path = f"{args.prefix_dir}network_split_{layer_idx}_{str(half)}.onnx"
                            output_path = args.prefix_dir + f"network_split_{layer_idx}_{str(half)}.onnx" if args.pruning_method == "CAP" else args.prefix_dir + f"network_split_{layer_idx}_{str(half)}_{args.pruning_method}.onnx"
                            if half:
                                output_names = [f"/blocks.{layer_idx}/Add_2_output_0"]
                            else:
                                output_names = [f"/blocks.{layer_idx}/Add_3_output_0"]
                                if layer_idx == (model.depth - 1):
                                    output_names = ["output"]
                            print("layer_idx:",layer_idx,"\t half:",str(half),"\t input_names:",input_names,"\t output_names:",output_names)
                            onnx.utils.extract_model(input_path, output_path, input_names, output_names,check_model=True)
                            input_names = output_names

                    # export onnx for each split layer of the new_model (teleported model)
                    # Convs layer
                    output_path = args.prefix_dir + "network_split_convs_teleported.onnx" if args.pruning_method == "CAP" else args.prefix_dir + f"network_split_convs_teleported_{args.pruning_method}.onnx"
                    input_names = ["input"]
                    output_names = ["/Add_output_0"]
                    onnx.utils.extract_model(input_path, output_path, input_names, output_names, check_model=True)
                    input_names = output_names
                    print("====== EXPORT ONNX OF SPLITED TELEPORTED LAYERS ======")
                    for layer_idx in range(model.depth):
                        # Apply the teleportation to the new_model (Using for computing the next layer inputs)
                        for half in [True,False]:
                            output_path = args.prefix_dir + f"network_split_{layer_idx}_{str(half)}_teleported.onnx" if args.pruning_method == "CAP" else args.prefix_dir + f"network_split_{layer_idx}_{str(half)}_teleported_{args.pruning_method}.onnx"
                            if half:
                                output_names = [f"/blocks.{layer_idx}/Add_2_output_0"]
                            else:
                                output_names = [f"/blocks.{layer_idx}/Add_3_output_0"]
                                if layer_idx == (model.depth - 1):
                                    output_names = ["output"]
                            print("layer_idx:",layer_idx,"\t half:",str(half),"\t input_names:",input_names,"\t output_names:",output_names)
                            onnx.utils.extract_model(input_path, output_path, input_names, output_names,check_model=True)
                            input_names = output_names
                    