# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python (fmri)
#     language: python
#     name: fmri
# ---

# +
import argparse
import os
import copy
import time
from enum import Enum
import pickle
import importlib

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.data import Subset
import attention
from sklearn.linear_model import LogisticRegression


import datetime
import utils
import numpy as np
import math
import einops
import random
import pandas as pd

import wandb 
import sys 
import glob
from collections import defaultdict

# +
parser = argparse.ArgumentParser(description='GMM L2L Training with Sequence Model')
parser.add_argument('--data', metavar='DIR', nargs='?', default='./data',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--cache', default='./cache',
                    help='path to cached files (e.g. for previous random weights)')
parser.add_argument(
    "--wandb_log",action=argparse.BooleanOptionalAction,default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--wandb_project",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--wandb_group_name",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--train_seed', default=None, type=int,
                    help='seed for training dataset.')
parser.add_argument('--probe_seed', default=None, type=int,
                    help='seed for probe dataset.')
parser.add_argument('--num_iter_per_epoch', default=100, type=int,  
                    help='number of iters per epoch')
parser.add_argument('--num_iters', default=5e5, type=int,  
                    help='number of iters')
parser.add_argument('--num_iters_per_epoch', default=100, type=int,  
                    help='number of iters per epoch')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')                         
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--optimizer', default='SGD', type=str, 
                    choices = ['SGD', 'Adam'],
                    help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--arch', '-a', metavar='ARCH', default='mlp',
                    help='model architecture (default: mlp)')
parser.add_argument('--gpt_bias', default="True", type=str,
                    help='whether to include bias in GPT')
parser.add_argument('--is_initialize_attention_weights_to_zero', default="False", type=str,
                    help='whether to initialize attention weights to zero')
parser.add_argument('--num_hidden_features', default=1, type=int,
                    help='num_hidden_features')
parser.add_argument('--num_hidden_features_mlp', default=None, type=int,
                    help='MLP width for OneLayerAttention (defaults to model dimension)')
parser.add_argument('--num_layers', default=1, type=int,
                    help='num_layers in transformer')
parser.add_argument('--num_heads', default=1, type=int,
                    help='num_heads in transformer')
parser.add_argument('--num_mlp_layers', default=3, type=int,
                    help='num_mlp_layers in transformer')
parser.add_argument('--vocab_size', default=2, type=int,
                    help='vocabulary size for OneLayerAttention')
parser.add_argument('--len_context', default=1, type=int,
                    help='number of in-context images in sequence')
parser.add_argument('--SLURM_ARRAY_TASK_ID', default=1, type=int,
                    help='SLURM_ARRAY_TASK_ID')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')  
 
parser.add_argument('--K', default=1, type=int, 
                    help='number of tasks')
parser.add_argument('--sequence_sampling_distribution', type=str,
                    default="uniform", 
                    choices = ["uniform", "zipf"],
                    help='sequence_sampling_distribution')
parser.add_argument('--is_resample_tasks', default="False", type=str,
                    help='whether to resample tasks')
parser.add_argument('--is_probe', default="True", type=str,
                    help='whether to probe the model')
parser.add_argument('--num_probe_sequences', default=1000, type=int,
                    help='number of probe sequences')
parser.add_argument('--probe_every_iter', default=5, type=int,
                    help='how many iters to probe the model')
parser.add_argument(
    '--savedir', 
            default="",
            type=str, 
            action='store'
)
parser.add_argument(
            '--fileprefix', 
            default="",
            type=str, 
            action='store') 
parser.add_argument('--resume', type=str, default=None,
                    help='resume from checkpoint')
    

# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    arch = "pytorch_transformer"
    # arch = "transformer"
    SLURM_ARRAY_TASK_ID = 0 
    optimizer = "SGD"
    lr = 1e-3
    num_iters = int(5e5 )
    gpt_bias = "True"
    len_context = 100
    K = 1000
    sequence_sampling_distribution = "uniform"
    jupyter_args = f"--data ./cache --fileprefix transformer --SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} --batch-size 256 --optimizer ${optimizer} --lr ${lr} --wd 0.0  --num_iters ${num_iters} --arch gpt --gpt_bias ${gpt_bias} --num_hidden_features 8 --num_layers 4 --len_context ${len_context} --K ${K} --sequence_sampling_distribution ${sequence_sampling_distribution} --no-wandb_log --wandb_project l2l --wandb_group_name t"
    # replace $ with '' in jupyter_args 
    jupyter_args = jupyter_args.replace("$","")
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    # %load_ext autoreload 
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    # %autoreload 2 

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()
 
    
def set_zipf_exp_one_layer_attention(args, lr, num_hidden_features, num_heads, vocab_size):
    args.arch = "OneLayerAttention" 
    # args.arch = "gpt"
    args.len_context = 100
    args.num_hidden_features = num_hidden_features
    args.num_heads = num_heads
    args.vocab_size = vocab_size
    
    args.is_resample_tasks = "False"
    args.num_iters = 2000
    args.sequence_sampling_distribution = "zipf"
    args.K = 1000
    
    args.lr = lr
    
    # how many iters to probe the model
    args.probe_every_iter = 1
    
    
    
    
if args.wandb_group_name == "memo_apr3_zipf_onelayerattention_lr_1e3_swapmlp_eval_fork_progress":
    # num_hidden_features is log spaced from 2 to 512
    args.train_seed = 42
    args.probe_seed = 43
    
    num_hidden_features_list = 2 ** np.linspace(0, 9, 10)
    num_hidden_features = int(num_hidden_features_list[args.SLURM_ARRAY_TASK_ID % len(num_hidden_features_list)])
    num_heads_list = [1] + list(np.arange(2, 24, 4)) # length: 7
    num_heads = int(num_heads_list[args.SLURM_ARRAY_TASK_ID // len(num_hidden_features_list)])
    set_zipf_exp_one_layer_attention(args, 1e-3, num_hidden_features, num_heads, int(2))  
    args.K = 100000 
elif args.wandb_group_name == "memo_jan4_fork_progress_set_attn_weights_to_zero_K_100000":
    # num_hidden_features is log spaced from 2 to 512
    args.train_seed = 42
    args.probe_seed = 43
    
    num_hidden_features_list = 2 ** np.linspace(0, 9, 5)
    num_hidden_features = int(num_hidden_features_list[args.SLURM_ARRAY_TASK_ID % len(num_hidden_features_list)])
    num_heads_list = [1] + list(np.arange(2, 24, 8)) # length: 4
    num_heads = int(num_heads_list[args.SLURM_ARRAY_TASK_ID // len(num_hidden_features_list)])
    set_zipf_exp_one_layer_attention(args, 1e-3, num_hidden_features, num_heads, int(2))  
    args.K = 100000
    args.is_initialize_attention_weights_to_zero = "zero_all_attn_weights"
    args.num_iters = 40000
    args.num_iters_per_epoch = 400 
elif args.wandb_group_name == "memo_jan4_fork_progress_set_zero_all_attn_allow_learning_K_100000":
    # num_hidden_features is log spaced from 2 to 512
    args.train_seed = 42
    args.probe_seed = 43
    
    num_hidden_features_list = 2 ** np.linspace(0, 9, 5)
    num_hidden_features = int(num_hidden_features_list[args.SLURM_ARRAY_TASK_ID % len(num_hidden_features_list)])
    num_heads_list = [1] + list(np.arange(2, 24, 8)) # length: 4
    num_heads = int(num_heads_list[args.SLURM_ARRAY_TASK_ID // len(num_hidden_features_list)])
    set_zipf_exp_one_layer_attention(args, 1e-3, num_hidden_features, num_heads, int(2))  
    args.K = 100000
    args.is_initialize_attention_weights_to_zero = "zero_all_attn_allow_learning"
    args.num_iters = 40000
    args.num_iters_per_epoch = 400
else:
    raise ValueError(f"wandb_group_name is not valid: {args.wandb_group_name}")

if args.arch == "OneLayerAttention" and args.num_hidden_features_mlp is None:
    args.num_hidden_features_mlp = args.num_heads * args.num_hidden_features

# assert args.K % args.L == 0, "K must be divisible by L"
if args.seed is None:
    args.seed = np.random.randint(0, 10000000)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local Rank for distributed training
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)
print("args:\n",vars(args))
# setup weights and biases (optional)
if local_rank==0 and args.wandb_log: # only use main process for wandb logging
    print(f"wandb {args.wandb_project} run")
    wandb_model_name = f"{args.fileprefix}"
    wandb_config = vars(args)
    os.environ["WANDB_MODE"] = "offline" # turn off wandb logging for now
    print("wandb_id:",wandb_model_name)
    wandb.login(relogin=True, key = os.environ["WANDB_API_KEY"])
    wandb.init(
        project=args.wandb_project,
        name=wandb_model_name,
        config=wandb_config,
        resume="allow",
        group=args.wandb_group_name
    )
    wandb.config.local_file_dir = wandb.run.dir 
else:
    record = {
        "args": vars(args),
        "logs": []
    }
if args.resume is not None:
    with open(args.resume, 'rb') as f:
        record = pickle.load(f)
    args = record['args']
    

# +

class Sequence(torch.utils.data.Dataset):
    def __init__(self, K,   
                 len_context = 1,
                len_data = 60000, skip=False,
                sequence_sampling_distribution = "uniform",
                seed = None):

        # if K < 40000:
        self.len_context = len_context
        self.K = K 
        self.len_data = len_data
        self.sequence_sampling_distribution = sequence_sampling_distribution
        self.seed = seed
        if skip == False:
            self.generate_sequences()
        if args.sequence_sampling_distribution == "zipf":
            self.p = 1.0 / np.arange(1, self.K + 1)
            self.p /= np.sum(self.p)
        else:
            assert self.sequence_sampling_distribution in ["uniform", "probe_uniform"], f"sequence_sampling_distribution must be uniform or probe_uniform, got {self.sequence_sampling_distribution}"
        self.skip = skip
        
    def generate_sequences(self):
        """
        Generate sequences from the dataset.
        If seed is not None, use the seed to generate the sequences.
        Otherwise, use the default random generator.
        """
        if self.seed is not None:
            rng = torch.Generator()
            rng.manual_seed(self.seed)
            self.sequences = torch.randint(0, 2, (self.K, self.len_context), generator=rng) 
            print ("sequences", self.sequence_sampling_distribution, self.sequences[:10])
        else:
            self.sequences = torch.randint(0, 2, (self.K, self.len_context)) 
        
    def __len__(self):
        if self.skip:
            return len(self.sequences)
        return self.len_data

    def __getitem__(self, task: int):
        if self.skip:
            i = task % len(self.sequences)
        elif self.sequence_sampling_distribution == "uniform":
            i = task % self.K
        elif self.sequence_sampling_distribution == "probe_uniform":
            i = np.random.choice(self.K)
        elif self.sequence_sampling_distribution == "zipf":
            i = np.random.choice(self.K, p= self.p)
        samples = self.sequences[i]
        # samples = torch.randint(0, 2, (self.len_context,))
        return samples.type(torch.long), i 


# -

# importlib.reload(gpt)
import gpt
criterion = nn.NLLLoss(reduction="none")
# define the model, optimizer, and scheduler, and criterion
if args.arch == "gpt":
    import gpt 
    config = gpt.GPTConfig(
        block_size = args.len_context,
        n_embd=args.num_heads * args.num_hidden_features,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        bias = args.gpt_bias == "True",
        is_initialize_attention_weights_to_zero = args.is_initialize_attention_weights_to_zero 
    )
    get_model = lambda: gpt.GPT(config, criterion).to(device)
    model = get_model()
elif args.arch == "OneLayerAttention":
    import gpt
    # gpt.py
    # arguments: len_context, num_heads, num_hidden_features, vocab_size, num_mlp_layers)
    get_model = lambda: gpt.OneLayerAttention(args.len_context, 
                                  args.num_heads, 
                                  args.num_hidden_features, 
                                  args.vocab_size, 
                                  args.num_mlp_layers,
                                  is_initialize_attention_weights_to_zero=args.is_initialize_attention_weights_to_zero,
                                  ).to(device)
    model = get_model()
if args.optimizer == 'SGD': 
    optimizer = torch.optim.SGD(model.parameters(),  
                            lr=args.lr, 
                            weight_decay=args.weight_decay
                            )
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),  
                            lr=args.lr, 
                            weight_decay=args.weight_decay
                            )
else:
    raise ValueError("optimizer not recognized")
# optimizer = get_optimizer(model.parameters())
iters_per_epoch = 1000
# scheduler = StepLR(optimizer, step_size=50, gamma=0.7)
# scheduler = OneCycleLR(optimizer, max_lr=args.lr, 
#                        total_steps=args.epochs * iters_per_epoch, 
#                        pct_start=0.5,
#                        steps_per_epoch=iters_per_epoch, epochs=args.epochs)

# +
# define the dataset
train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.batch_size}
use_cuda = not args.no_cuda and torch.cuda.is_available()
if use_cuda:
    cuda_kwargs = {'num_workers': args.workers,
                    "shuffle": True,
                    'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
train_dataset = Sequence(K=args.K, 
                         len_data = 1000 * args.num_iter_per_epoch,
                         len_context=args.len_context,
                         sequence_sampling_distribution = args.sequence_sampling_distribution,
                         seed = args.train_seed
                         )
# IWL dataset is for testing the model on every 10th sequence from train_dataset
iwl_dataset = Sequence(K=args.K, len_context=args.len_context,  skip=True, 
                       len_data = args.K // 10,
                       sequence_sampling_distribution = "uniform")
iwl_dataset.sequences = train_dataset.sequences[::10] # take every 10th sequence from train_dataset
print ("sequences.sequence_sampling_distribution", iwl_dataset.sequence_sampling_distribution)

# Create probe sequences independent of train_dataset
probe_dataset = Sequence(K=args.num_probe_sequences, len_context=args.len_context,  skip=False, 
                       len_data = 2 ** 20, # infinite data for now, break when threshold accuracy is reached
                       sequence_sampling_distribution = "probe_uniform",
                       seed = args.probe_seed)

train_sampler = None
val_sampler = None 
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            sampler=train_sampler, 
                                            **train_kwargs)  
# At test time we don't shuffle and sample uniformly
iwl_test_loader = torch.utils.data.DataLoader(iwl_dataset,
                                            sampler=val_sampler,
                                            **{'batch_size': args.batch_size, 'num_workers': args.workers,
                                        "shuffle": False,
                                        'pin_memory': True})

print ("sequences.sequence_sampling_distribution", probe_dataset.sequence_sampling_distribution)
probe_sampler = None
probe_loader = torch.utils.data.DataLoader(probe_dataset, 
                                            sampler=probe_sampler, 
                                            **{'batch_size': args.batch_size, 'num_workers': args.workers,
                                               "shuffle": True,
                                               'pin_memory': True})


# +
# # test dataset construction 
# ihistogram = []
# for _, (seq, i) in enumerate(train_loader):
#     ihistogram.extend(i.tolist()) 
#     if _ > 100: break 
# # print ("train_dataset", train_dataset.p, train_dataset.K)
# print ("ihistogram",ihistogram)``
# import matplotlib.pyplot as plt
# # plot histogram, should be zipf distribution
# # plt.hist(ihistogram, bins = 20)
# plt.bar(np.arange(args.K), np.bincount(ihistogram, minlength=args.K))
# plt.plot(np.arange(args.K), train_dataset.p * len(ihistogram), "r")
# # plt.xlim(0, 30)
# plt.semilogy()
# plt.title("Histogram of tasks in training set, should be zipf distribution")
# plt.legend(["expected: p(zipf) * n_samps", "empirical"])
# plt.xlabel("seq rank")
# plt.ylabel("freq")
# plt.show()
# -

# * Train our one-layer attention model on 100K normal sequences with a Zipf distribution
#     * Have a set of 1000 probe sequences 
# * For a regular # of epochs we fork the model and train it to some threshold accuracy on only the probe sequences (which are presented uniformly at random) 
#     * During the probing iterations, the attention weights are fixed and only the MLP weights are allowed to change
#     * I use the same set of training and probe sequences for all models with different # of heads
# * I record how many iterations it takes to train the model to 95% accuracy and the accuracy for each iteration of the probe training
#     * I made sure that models with different # of heads are all trained on the same set of training and probe sequences
# * The models are initialized independently
#
#
#
#
#
#
#
#
#

# +

def validate_gradient_descent(epoch, val_loader, model, args, criterion, device):
    test_losses = [utils.AverageMeter('Loss', ':.4e') for _ in range(args.len_context)]
    test_top1 = [utils.AverageMeter('Acc@1', ':6.2f') for _ in range(args.len_context)]
    model.eval()

    with torch.no_grad():
        for i, (seq, task) in enumerate(val_loader):
            seq, task = seq.to(device), task.to(device)
            output = model(seq, task)
            preds = output
            _, _, D = output.shape

            for i_seq in range(args.len_context - 1):
                preds_query = preds[:, i_seq, :].reshape(-1, D)
                seqs_query = seq[:, (i_seq + 1)].reshape(-1)
                logsoftmax = F.log_softmax(preds_query, dim=1)
                losstest = criterion(logsoftmax, seqs_query)
                acc1 = utils.accuracy(preds_query, seqs_query, topk=[1])
                test_losses[i_seq].update(losstest.item(), seq.size(0))
                test_top1[i_seq].update(acc1[0].item(), seq.size(0))

            if i > 10:
                break

    return test_losses, test_top1


def _fit_linear_probe(train_features, train_labels, test_features, test_labels):
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(train_features, train_labels)
    return logistic_regression.score(test_features, test_labels)


def _collect_mlp_probe_metrics(mlp_reps_per_layer, num_layers):
    mlp_probe_metrics = {
        "layer": [],
        "accuracy": [],
        "num_train": [],
        "num_test": [],
    }

    labels = np.array(mlp_reps_per_layer["label"])
    if labels.size == 0:
        return mlp_probe_metrics

    for layer_idx in range(num_layers):
        if len(mlp_reps_per_layer[layer_idx]) == 0:
            continue

        layer_reps = np.concatenate(mlp_reps_per_layer[layer_idx], axis=0)
        num_examples = layer_reps.shape[0]
        num_train = int(0.8 * num_examples)
        if num_train == 0 or num_train == num_examples:
            continue

        train_ids = np.random.choice(np.arange(num_examples), size=num_train, replace=False).astype(int)
        test_ids = np.setdiff1d(np.arange(num_examples), train_ids).astype(int)
        accuracy = _fit_linear_probe(
            layer_reps[train_ids],
            labels[train_ids],
            layer_reps[test_ids],
            labels[test_ids],
        )
        mlp_probe_metrics["layer"].append(layer_idx)
        mlp_probe_metrics["accuracy"].append(accuracy)
        mlp_probe_metrics["num_train"].append(len(train_ids))
        mlp_probe_metrics["num_test"].append(len(test_ids))

    for key, values in mlp_probe_metrics.items():
        mlp_probe_metrics[key] = np.array(values)
    return mlp_probe_metrics


def validate_gradient_descent_zipf(epoch, val_loader, model, args, criterion, device):
    test_metrics = defaultdict(list)
    model.eval()

    num_eval_sequences = len(val_loader.dataset.sequences)
    sequence_ranks = torch.zeros((num_eval_sequences, args.len_context - 1), dtype=torch.long).to(device)
    lengths = torch.zeros((num_eval_sequences, args.len_context - 1), dtype=torch.long).to(device)
    logsoftmaxlosses = torch.zeros((num_eval_sequences, args.len_context - 1), dtype=torch.float).to(device)
    accuracys = torch.zeros((num_eval_sequences, args.len_context - 1), dtype=torch.float).to(device)

    analysis_batches = 15
    analyzed_batches = 0
    vertical_bars_metric = None
    decorrelation_metric = None
    mlp_reps_per_layer = defaultdict(list)
    num_mlp_layers = None

    with torch.no_grad():
        for i, (seq, task) in enumerate(val_loader):
            seq, task = seq.to(device), task.to(device)
            output = model(seq, task)
            preds = output
            _, _, D = output.shape

            if i < analysis_batches and hasattr(model, "get_attention_weights"):
                _, attn = model.get_attention_weights(seq, task)
                batch_vertical_bars = utils.corr_attn_prefix_rows(attn)
                batch_decorrelation = utils.corr_heads_at_ts(attn)
                if vertical_bars_metric is None:
                    vertical_bars_metric = batch_vertical_bars
                    decorrelation_metric = batch_decorrelation
                else:
                    vertical_bars_metric += batch_vertical_bars
                    decorrelation_metric += batch_decorrelation
                analyzed_batches += 1

            if i < analysis_batches and hasattr(model, "forward_mlp"):
                _, mlp_reps_batch = model.forward_mlp(seq, 1)
                num_mlp_layers = len(mlp_reps_batch)
                T = mlp_reps_batch[0].shape[1]
                mlp_reps_per_layer["label"].extend(np.repeat(task.detach().cpu().numpy(), T))
                for layer_idx, mlp_rep in enumerate(mlp_reps_batch):
                    mlp_reps_per_layer[layer_idx].append(
                        mlp_rep.detach().cpu().numpy().reshape(-1, mlp_rep.shape[-1])
                    )

            batch_start = i * args.batch_size
            batch_end = batch_start + len(task)
            for i_seq in range(args.len_context - 1):
                preds_query = preds[:, i_seq, :].reshape(-1, D)
                seqs_query = seq[:, (i_seq + 1)].reshape(-1)
                is_correct = preds_query.argmax(dim=-1) == seqs_query
                logsoftmax = F.log_softmax(preds_query, dim=1)
                logsoftmaxloss = F.nll_loss(logsoftmax, seqs_query, reduction="none")

                sequence_ranks[batch_start:batch_end, i_seq] = task
                lengths[batch_start:batch_end, i_seq] = i_seq + 1
                logsoftmaxlosses[batch_start:batch_end, i_seq] = logsoftmaxloss
                accuracys[batch_start:batch_end, i_seq] = is_correct.float()

    test_metrics["sequence_rank"] = sequence_ranks[:, 0].detach().cpu().numpy()
    test_metrics["length"] = 50
    test_metrics["logsoftmaxloss"] = logsoftmaxlosses.mean(dim=1).detach().cpu().numpy()
    test_metrics["accuracy"] = accuracys.mean(dim=1).detach().cpu().numpy()

    if analyzed_batches > 0 and vertical_bars_metric is not None and decorrelation_metric is not None:
        test_metrics["vertical_bars_metric"] = (
            vertical_bars_metric / analyzed_batches
        ).detach().cpu().numpy().mean(axis=0)
        test_metrics["decorrelation_metric"] = (
            decorrelation_metric / analyzed_batches
        ).detach().cpu().numpy().mean(axis=0)

    if num_mlp_layers is not None:
        test_metrics["mlp_probe_metrics"] = _collect_mlp_probe_metrics(mlp_reps_per_layer, num_mlp_layers)

    return test_metrics

def probe_gradient_descent_uniform(epoch, val_loader, model, args, criterion, device, get_model):
    # model.train()
    # model_fork = copy.deepcopy(model).to(device)
    model_fork = get_model()
    model_fork.load_state_dict(copy.deepcopy(model.state_dict()))
    model_fork.train() 
    model_fork.require_grad_attention_weights(require_grad=False) # freeze the model attention parameters, so we only train the forked model
    # model_fork_optimizer = get_optimizer(model_fork.parameters()) # initialize new optimizer for the forked model
    if args.optimizer == 'Adam':
        print("args.lr", args.lr)
        model_fork_optimizer = torch.optim.Adam(model_fork.parameters(),  
                            lr=args.lr, 
                            weight_decay=args.weight_decay
                            )
    elif args.optimizer == 'SGD':
        model_fork_optimizer = torch.optim.SGD(model_fork.parameters(),  
                            lr=args.lr, 
                            weight_decay=args.weight_decay
                            )
    else:
        raise ValueError("optimizer not recognized")
    
    test_metrics = defaultdict(list) 
    accuracys = []
    for num_iters, (seq, task) in enumerate(val_loader):
        model_fork_optimizer.zero_grad()
        seq, task = seq.to(device), task.to(device) 

        # print("seq", seq.shape, "task", task)
        output = model_fork(seq, task) # shape: B, N, D
        B, N, D = output.shape
        preds = output # shape: (B, L, 2)

        # loss function: cross-entropy loss
        # output at position i should be the input at position i+1
        
        # Write a function to compute the binary cross-entropy for each position in the sequence
        # But don't compute the mean, keep the vector dimension
        logsoftmax = F.log_softmax(preds[:,:-1,:], dim=-1).reshape(B * (N-1), D)
        logsoftmaxloss = criterion(logsoftmax, seq[:,1:].reshape(B * (N-1)))
        # logsoftmaxloss = logsoftmaxloss.reshape(B, N-1).mean(dim=-1) # shape: (B,)
        loss = logsoftmaxloss.mean()
        
        loss.backward()
        model_fork_optimizer.step() 
        # print("task", task[:10], "preds", preds.shape, "model_state_dict", model_fork.state_dict()[list (model_fork.state_dict().keys())[0]].flatten()[:50])
        # preds contains the predictions for the next position, so we need to compare it with the actual next position
        is_correct = (preds.argmax(dim=-1)[:,:-1] == seq[:,1:]) 
        accuracy = is_correct.float().mean()
        accuracys.append(accuracy.detach().cpu().numpy())
        # compute the average accuracy of the last 10 items in accuracys
        running_average_accuracy = np.mean(accuracys[-10:])
        # if running_average_accuracy > 0.95 then stop trainin
        if running_average_accuracy > 0.95: 
            break 
        elif num_iters > 1000:
            break 
        # print ("loss", loss, "accuracy", accuracy)

    test_metrics["accuracy"] = accuracys 
    test_metrics["num_iters_required"] = num_iters
     
    return test_metrics
 


# +
import json
import pickle
# import matplotlib.pyplot as plt
if os.path.exists(f"{args.savedir}/cache/{args.wandb_group_name}") == False:
    os.makedirs(f"{args.savedir}/cache/{args.wandb_group_name}", exist_ok=True)
fileprefix = f"{args.wandb_group_name}_{args.fileprefix}_K_{args.K}_L_{args.len_context}_hidden_{args.num_hidden_features}_nheads_{args.num_heads}_nlayers_{args.num_layers}_{time.time()}"
dirprefix = f"{args.savedir}/cache/{args.wandb_group_name}/{fileprefix}"
if os.path.exists(dirprefix) == False:
    os.makedirs(dirprefix, exist_ok=True)
exp_name = f"{dirprefix}/{fileprefix}.pkl"
print("Saving to", exp_name)
# fileprefix = f"{args.wandb_group_name}_{args.fileprefix}_K_{args.K}_L_{args.len_context}_hidden_{args.num_hidden_features}_nheads_{args.num_heads}_nlayers_{args.num_layers}_{time.time()}"
# dirprefix = f"./cache/{args.wandb_group_name}/{fileprefix}"
# if os.path.exists(dirprefix) == False:
#     os.makedirs(dirprefix, exist_ok=True)
# exp_name = f"{dirprefix}/{fileprefix}.pkl"
# print("Saving to", exp_name)
num_apppearances = np.zeros(args.K)
test_every = 1 # np.log10(args.K).astype(int) * 2
all_sequences_across_switches = {
    "sequences": [],
    "switch_start_iter": [],
}
# save all sequences in pickle file
all_sequences_across_switches["sequences"].append(copy.deepcopy(train_dataset.sequences))
all_sequences_across_switches["switch_start_iter"].append([0] * len(train_dataset.sequences))
with open(f"{dirprefix}/{fileprefix}_all_sequences.pkl", "wb") as f:
    pickle.dump(all_sequences_across_switches, f)
for iter in range(args.num_iters // args.num_iters_per_epoch):
    # Switch the sequences half way through the training
    if iter == args.num_iters // args.num_iters_per_epoch / 2 and args.is_resample_tasks == "True": # resample the tasks
        train_dataset.generate_sequences()
        all_sequences_across_switches["sequences"].append(copy.deepcopy(train_dataset.sequences))
        all_sequences_across_switches["switch_start_iter"].append([iter] * len(train_dataset.sequences))
        with open(f"{dirprefix}/{fileprefix}_all_sequences.pkl", "wb") as f:
            pickle.dump(all_sequences_across_switches, f)
        
        if args.sequence_sampling_distribution == "zipf":
            iwl_dataset.sequences = train_dataset.sequences[::10] # take every 10th sequence from train_dataset
            iwl_test_loader = torch.utils.data.DataLoader(iwl_dataset,
                                            sampler=val_sampler,
                                            **{'batch_size': args.batch_size, 'num_workers': args.workers,
                                        "shuffle": False,
                                        'pin_memory': True})
        num_apppearances = np.zeros(args.K)
        
    # Switch the sequences several times throughout the training
    # Save the sequences and the start iter of the switch, so that
    # we can plot forgetting curves across different switches
    elif args.is_resample_tasks == "Forget" and iter % 100 == 0:
        train_dataset.generate_sequences()
        all_sequences_across_switches["sequences"].append(copy.deepcopy(train_dataset.sequences[::50]))
        all_sequences_across_switches["switch_start_iter"].append([iter] * len(train_dataset.sequences[::50]))
        num_apppearances = np.zeros(args.K)
        if args.sequence_sampling_distribution == "zipf":
            iwl_dataset.sequences = torch.cat(all_sequences_across_switches["sequences"], dim=0)
            iwl_dataset.len_data = len(iwl_dataset.sequences)
            # print("iter", iter, "len(iwl_dataset.sequences)", iwl_dataset.sequences.shape)
            iwl_test_loader = torch.utils.data.DataLoader(iwl_dataset,
                                            sampler=val_sampler,
                                            **{'batch_size': args.batch_size, 'num_workers': args.workers,
                                        "shuffle": False,
                                        'pin_memory': True})
    logs = {
        "num_apppearances": copy.deepcopy(num_apppearances),
    }
    if args.sequence_sampling_distribution == "zipf"  :
        if (iter % test_every == 0):
            test_metrics = validate_gradient_descent_zipf(iter, iwl_test_loader, model, args, criterion, device)
        else:
            test_metrics = {}
    else:
        test_losses, test_top1 = validate_gradient_descent(iter, train_loader, model, args, criterion, device)
        
    if args.is_probe == "True" and iter % args.probe_every_iter == 0:
        probe_metrics = probe_gradient_descent_uniform(iter, probe_loader, model, args, criterion, device, get_model) 
        logs["probe_metrics"] = probe_metrics
        
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    
    
    model.train() # switch to train mode
    # train iteration
    appearances = []
    loss_per_appearance = []
    for i, (seq, task) in enumerate(train_loader):
        optimizer.zero_grad()
        seq, task = seq.to(device), task.to(device)
        batch_num_appearances = torch.bincount(task, minlength=args.K).detach().cpu().numpy()
        num_apppearances += batch_num_appearances # update the number of appearances of each task
        
        # print ("seq", seq.shape, "task", task.shape, batch_num_appearances[:20])
        output = model(seq, task)
        B, N, D = output.shape
        preds = output # shape: (B, L, 2)
        # loss function: cross-entropy loss
        # output at position i should be the input at position i+1
        
        # Write a function to compute the binary cross-entropy for each position in the sequence
        # But don't compute the mean, keep the vector dimension
        # loss = criterion(preds[:,:-1,:].reshape(B * (N-1), D), seq[:,1:].reshape(B * (N-1)))
        logsoftmax = F.log_softmax(preds[:,:-1,:], dim=-1).reshape(B * (N-1), D)
        logsoftmaxloss = criterion(logsoftmax, seq[:,1:].reshape(B * (N-1)))
        logsoftmaxloss = logsoftmaxloss.reshape(B, N-1).mean(dim=-1) # shape: (B,)
        appearances.append(task.detach().cpu().numpy())
        loss_per_appearance.append(logsoftmaxloss.detach().cpu().numpy())
        loss = logsoftmaxloss.mean()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), seq.size(0)) 
    logs.update({
            "train_loss": losses.avg,
            "epoch": iter,
            "lr": optimizer.param_groups[0]['lr'],  
            "loss_per_appearance": (loss_per_appearance),
            "appearances": (appearances),
        })
    # print("iter", iter, "loss", losses.avg)
    if iter == args.num_iters - 1: 
        if args.sequence_sampling_distribution == "zipf":
            test_metrics = validate_gradient_descent_zipf(iter, iwl_test_loader, model, args, criterion, device)
        else:
            test_losses, test_top1 = validate_gradient_descent(iter, train_loader, model, args, criterion, device)            
    if args.sequence_sampling_distribution == "zipf":
        logs["test_metrics"] = test_metrics
    elif args.sequence_sampling_distribution == "uniform":
        for i in range(args.len_context):
            logs[f"test_loss_{i}"] = test_losses[i].avg
            logs[f"test_top1_{i}"] = test_top1[i].avg
    
    if args.wandb_log:
        wandb.log(logs)
    else:
        # wandb.log(logs)
        record["logs"].append(logs)
    
 
    # save phi_xt_list_epoch 

    if iter % 50 == 5 and args.wandb_log != True:
        # record["model"] = model.state_dict()
        with open(exp_name, "wb") as f:
            pickle.dump(record, f)
            
        # save model state_dict
        # with open(f"{dirprefix}/{fileprefix}_model_iter_{iter}.pkl", "wb") as f:
            # pickle.dump(model.state_dict(), f)
        # use json
        # with open(exp_name, "w") as f:
            # json.dump(record, f)
  
if args.wandb_log != True:
    with open(exp_name, "wb") as f:
        pickle.dump(record, f)
        
    with open(f"{dirprefix}/{fileprefix}_all_sequences.pkl", "wb") as f:
        pickle.dump(all_sequences_across_switches, f)
        
    # save model state_dict
    # with open(f"{dirprefix}/{fileprefix}_model_iter_{iter}.pkl", "wb") as f:
        # pickle.dump(model.state_dict(), f)
print("Finished training")
sys.exit(0)
