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
# import webdataset as wds

import datetime
import utils
import numpy as np
import math
import einops
import random

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
parser.add_argument('--num_iter_per_epoch', default=100, type=int,  
                    help='number of iters per epoch')
parser.add_argument('--num_iters', default=5e5, type=int,  
                    help='number of iters')
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
parser.add_argument('--num_layers', default=1, type=int,
                    help='num_layers in transformer')
parser.add_argument('--num_heads', default=1, type=int,
                    help='num_heads in transformer')
parser.add_argument('--num_mlp_layers', default=3, type=int,
                    help='num_mlp_layers in transformer')
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
parser.add_argument('--num_iters_per_epoch', default=50, type=int,
                    help='number of iters per epoch')
parser.add_argument(
            '--fileprefix', 
            default="",
            type=str, 
            action='store') 
parser.add_argument(
    '--savedir', 
            default="",
            type=str, 
            action='store'
)
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
    args.len_context = 100
    args.num_hidden_features = num_hidden_features
    args.num_heads = num_heads
    args.vocab_size = vocab_size
    
    args.is_resample_tasks = "True"
    args.num_iters = 100000
    args.sequence_sampling_distribution = "zipf"
    args.K = 100000
    
    args.lr = lr
    
def set_zipf_exp_heads_mlp_gpt(args, lr, num_hidden_features, num_heads, vocab_size):
    args.arch = "HeadsMLPGPT"
    args.len_context = 100
    args.num_hidden_features = num_hidden_features
    args.num_heads = num_heads
    args.vocab_size = vocab_size
    
    args.is_resample_tasks = "True"
    args.num_iters = 500000
    args.sequence_sampling_distribution = "zipf"
    args.K = 100000
    args.num_iters_per_epoch = int(10000)
    args.lr = lr
    
    

if args.wandb_group_name == "memo_apr3_zipf_onelayerattention_lr_1e-3_swapmlp_eval":
    # num_hidden_features is log spaced from 2 to 512
    num_hidden_features_list = 2 ** np.linspace(0, 9, 10)
    num_hidden_features = int(num_hidden_features_list[args.SLURM_ARRAY_TASK_ID % len(num_hidden_features_list)])
    num_heads_list = [1] + list(np.arange(2, 22, 4)) # length: 7
    num_heads = int(num_heads_list[args.SLURM_ARRAY_TASK_ID // len(num_hidden_features_list)])
    set_zipf_exp_one_layer_attention(args, 1e-3, num_hidden_features, num_heads, int(2))  
    args.num_iters_per_epoch = int(1000)
elif args.wandb_group_name == "memo_aug15_zipf_onelayerattention_lr_1e-3":
    num_hidden_features = 64
    num_heads = 18
    set_zipf_exp_one_layer_attention(args, 1e-3, num_hidden_features, num_heads, int(2))  
elif args.wandb_group_name in ["memo_nov24_zipf_heads_mlp_gpt_lr_1e_3"]:
    num_hidden_features_list = 2 ** np.linspace(0, 9, 10)
    num_hidden_features = int(num_hidden_features_list[args.SLURM_ARRAY_TASK_ID % len(num_hidden_features_list)])
    num_heads_list = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    num_heads = int(num_heads_list[args.SLURM_ARRAY_TASK_ID // len(num_hidden_features_list)])
    set_zipf_exp_heads_mlp_gpt(args, 1e-3, num_hidden_features, num_heads, int(2)) 
elif args.wandb_group_name in ["memo_dec8_zipf_heads_mlp_gpt_lr_1e_3"]:
    num_hidden_features_list = 2 ** np.linspace(0, 9, 10)
    num_hidden_features = int(num_hidden_features_list[args.SLURM_ARRAY_TASK_ID % len(num_hidden_features_list)])
    num_heads_list = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    num_heads = int(num_heads_list[args.SLURM_ARRAY_TASK_ID // len(num_hidden_features_list)])
    set_zipf_exp_heads_mlp_gpt(args, 1e-3, num_hidden_features, num_heads, int(2))  
elif args.wandb_group_name == "memo_jan23_zipf_onelayerattention_lr_1e-3":
    num_hidden_features = 64
    num_heads = 18
    set_zipf_exp_one_layer_attention(args, 1e-3, num_hidden_features, num_heads, int(2))   
    args.is_resample_tasks = "False"
else:
    raise ValueError(f"wandb_group_name {args.wandb_group_name} not found")

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
        "logs": [],
        "mlp_state_dict": {},
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
                sequence_sampling_distribution = "uniform"):

        # if K < 40000:
        self.len_context = len_context
        self.K = K 
        self.len_data = len_data
        self.sequence_sampling_distribution = sequence_sampling_distribution
        if skip == False:
            self.generate_sequences()
        if args.sequence_sampling_distribution == "zipf":
            self.p = 1.0 / np.arange(1, self.K + 1)
            self.p /= np.sum(self.p)
        else:
            assert self.sequence_sampling_distribution == "uniform", f"sequence_sampling_distribution must be uniform or zipf, got {self.sequence_sampling_distribution}"
        self.skip = skip
    def generate_sequences(self):
        self.sequences = torch.randint(0, 2, (self.K, self.len_context)) 
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, task: int):
        if (self.sequence_sampling_distribution == "uniform") or (self.skip == True):
            i = task % self.K
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
elif args.arch == "HeadsMLPGPT":
    import gpt 
    get_model = lambda: gpt.HeadsMLPGPT(len_context=args.len_context, 
                                  num_heads=args.num_heads, 
                                  num_hidden_attn_features=16, 
                                  num_hidden_mlp_features=args.num_hidden_features, 
                                  num_input_mlp_features=64, 
                                  vocab_size=args.vocab_size, 
                                  num_layers=6).to(device)
    model = get_model()
elif args.arch == "OneLayerAttention":
    import gpt
    # gpt.py
    # arguments: len_context, num_heads, num_hidden_features, vocab_size, num_mlp_layers)
    get_model = lambda: gpt.OneLayerAttention(args.len_context, 
                                  args.num_heads, 
                                  args.num_hidden_features, 
                                  args.vocab_size, 
                                  args.num_mlp_layers).to(device)
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
                         sequence_sampling_distribution = args.sequence_sampling_distribution
                         )
iwl_dataset = Sequence(K=args.K, len_context=args.len_context,  skip=True, 
                       len_data = args.K // 10,
                       sequence_sampling_distribution = "uniform")
iwl_dataset.sequences = train_dataset.sequences[::10] # take every 10th sequence from train_dataset
print ("sequences.sequence_sampling_distribution", iwl_dataset.sequence_sampling_distribution)
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


# +

def validate_gradient_descent(epoch, val_loader, model, args, criterion, device):
    # seq_lens = list(range(1, args.len_context+1, 5)) 
   
    test_losses = [utils.AverageMeter('Loss', ':.4e') for _ in range(args.len_context)]
    test_top1 = [utils.AverageMeter('Acc@1', ':6.2f') for _ in range(args.len_context)]
    model.eval() # switch to eval mode
    
    with torch.no_grad():
        for i, (seq, task) in enumerate(val_loader):
            seq, task = seq.to(device), task.to(device) 
            
            
            output = model(seq, task) # shape: B, N, D
            # print ("seq", seq.shape, "task", task.shape, "output", output.shape )
            preds = output 
            B, N, D = output.shape
             
            
            # print ("test preds", preds.shape, "task", task.shape)
            
            # [test_losses[_].update(loss[_].item(), target.size(0)) for _ in range(N)]
            
            for i_seq in range(args.len_context-1): 
                preds_query = preds[:, i_seq,:].reshape(-1, D)
                seqs_query = seq[:,(i_seq+1)].reshape(-1)
                losstest = criterion(preds_query, seqs_query)
                acc1 = utils.accuracy(preds_query, seqs_query, topk=[1])
                print("preds_query", preds_query.shape, "seqs_query", seqs_query.shape)
                test_losses[i_seq].update(losstest.item(), seq.size(0))
                test_top1[i_seq].update(acc1[0].item(), seq.size(0))
                
            if i > 10: break
            # print ("acc1", acc1, "loss", loss)
            # test_top1[seq_len].update(acc1[0], target.size(0))
            # acc1 = torch.mean(((output.squeeze(1) * (seq_target*2-1)) > 0).float()).item()
            # test_top1[seq_len].update(acc1, target.size(0))

    return test_losses, test_top1


def validate_gradient_descent_zipf(epoch, val_loader, model, args, criterion, device):
    # seq_lens = list(range(1, args.len_context+1, 5)) 
   
    sequence_rank = []
    test_losses = [] 
    test_top1 = []
    test_metrics = defaultdict(list)
    model.eval() # switch to eval mode
    sequence_ranks = torch.zeros((len(val_loader.dataset.sequences),args.len_context-1), dtype=torch.long).to(device)
    lengths = torch.zeros((len(val_loader.dataset.sequences),args.len_context-1), dtype=torch.long).to(device)
    logsoftmaxlosses = torch.zeros((len(val_loader.dataset.sequences),args.len_context-1), dtype=torch.float).to(device) 
    accuracys = torch.zeros((len(val_loader.dataset.sequences),args.len_context-1), dtype=torch.float).to(device)
    with torch.no_grad():
        for i, (seq, task) in enumerate(val_loader):
            seq, task = seq.to(device), task.to(device) 
            
            # print("seq", seq.shape, "task", task)
            output = model(seq, task) # shape: B, N, D
            # print ("seq", seq.shape, "task", task.shape, "output", output.shape )
            preds = output 
            B, N, D = output.shape
             
            for i_seq in range(args.len_context-1): 
                preds_query = preds[:, i_seq,:].reshape(-1, D)
                seqs_query = seq[:,(i_seq+1)].reshape(-1)
                # losstest = criterion(preds_query, seqs_query)
                # acc1 = utils.accuracy(preds_query, seqs_query, topk=[1]) 
                # print ("preds_max", preds_query.argmax(dim=-1).shape, "seqs_query", seqs_query.shape)
                is_correct = (preds_query.argmax(dim=-1) == seqs_query)
                logsoftmax = F.log_softmax(preds_query, dim=1)
                logsoftmaxloss = F.nll_loss(logsoftmax, seqs_query, reduction="none") 
                
                # if args.K < 10000: # for small K, we can save the whole tensor 
                # test_metrics["sequence_rank"].append(task.detach().cpu().numpy())
                sequence_ranks[i*args.batch_size:i*args.batch_size+len(task), i_seq] = task 
                lengths[i*args.batch_size:i*args.batch_size+len(task), i_seq] = i_seq + 1 
                logsoftmaxlosses[i*args.batch_size:i*args.batch_size+len(task), i_seq] = logsoftmaxloss 
                accuracys[i*args.batch_size:i*args.batch_size+len(task), i_seq] = is_correct.float() 
    test_metrics["sequence_rank"] = sequence_ranks[:,0].detach().cpu().numpy() # average over positions
    test_metrics["length"] = 50
    test_metrics["logsoftmaxloss"] = logsoftmaxlosses.mean(dim=1).detach().cpu().numpy()
    test_metrics["accuracy"] = accuracys.mean(dim=1).detach().cpu().numpy()
    # make sure we can print long arrays 
    print("test_metrics accuracy with length", len(test_metrics["accuracy"]), test_metrics["accuracy"][:1000], test_metrics["accuracy"][10000:11000], "\n\n", )
     
    return test_metrics

# def probe_gradient_descent_swapped_mlp(iter, iwl_test_loader, model, args, criterion, device, record):
#     probed_model = get_model()
#     # load the state_dict of the model (this loads the attention weights)
#     probed_model.load_state_dict(model.state_dict()) 
#     print ("probed_model.linear_modules parameters before swap", list(probed_model.linear_modules.parameters())[0])
#     print ("probed_model.c_attn parameters before swap", list(probed_model.c_attn.parameters())[0])
#     # swap the mlp of the probed model with the original mlp
#     probed_model.linear_modules.load_state_dict(record["mlp_state_dict"]["pre_switch"]["linear_modules"])
#     probed_model.lm_head.load_state_dict(record["mlp_state_dict"]["pre_switch"]["lm_head"])
#     print ("probed_model.linear_modules parameters after swap", list(probed_model.linear_modules.parameters())[0])
#     print ("probed_model.c_attn parameters after swap", list(probed_model.c_attn.parameters())[0])
#     # probe the model on original sequences
#     probed_model.eval() # switch to eval mode
#     sequence_ranks = torch.zeros((args.K,args.len_context-1), dtype=torch.long).to(device)
#     lengths = torch.zeros((args.K,args.len_context-1), dtype=torch.long).to(device)
#     logsoftmaxlosses = torch.zeros((args.K,args.len_context-1), dtype=torch.float).to(device) 
#     accuracys = torch.zeros((args.K,args.len_context-1), dtype=torch.float).to(device)
#     probe_metrics = defaultdict(list)
#     with torch.no_grad():
#         for i, (seq, task) in enumerate(iwl_test_loader):
#             seq, task = seq.to(device), task.to(device) 
#             output = probed_model(seq, task) # shape: B, N, D
#             preds = output 
#             B, N, D = output.shape
            
#             for i_seq in range(args.len_context-1): 
#                 preds_query = preds[:, i_seq,:].reshape(-1, D)
#                 seqs_query = seq[:,(i_seq+1)].reshape(-1)
#                 # losstest = criterion(preds_query, seqs_query)
#                 # acc1 = utils.accuracy(preds_query, seqs_query, topk=[1]) 
#                 # print ("preds_max", preds_query.argmax(dim=-1).shape, "seqs_query", seqs_query.shape)
#                 is_correct = (preds_query.argmax(dim=-1) == seqs_query)
#                 logsoftmax = F.log_softmax(preds_query, dim=1)
#                 logsoftmaxloss = F.nll_loss(logsoftmax, seqs_query, reduction="none") 
                
#                 # if args.K < 10000: # for small K, we can save the whole tensor 
#                 # test_metrics["sequence_rank"].append(task.detach().cpu().numpy())
#                 sequence_ranks[task, i_seq] = task 
#                 lengths[task, i_seq] = i_seq + 1 
#                 logsoftmaxlosses[task, i_seq] = logsoftmaxloss 
#                 accuracys[task, i_seq] = is_correct.float() 
                
#     probe_metrics["sequence_rank"] = sequence_ranks[:,0].detach().cpu().numpy() # average over positions
#     probe_metrics["length"] = 50
#     probe_metrics["logsoftmaxloss"] = logsoftmaxlosses.mean(dim=1).detach().cpu().numpy()
#     probe_metrics["accuracy"] = accuracys.mean(dim=1).detach().cpu().numpy()
    
#     return probe_metrics
            
 
# -

import json
import pickle
if os.path.exists(f"{args.savedir}/cache/{args.wandb_group_name}") == False:
    os.makedirs(f"{args.savedir}/cache/{args.wandb_group_name}", exist_ok=True)
fileprefix = f"{args.wandb_group_name}_{args.fileprefix}_K_{args.K}_L_{args.len_context}_hidden_{args.num_hidden_features}_nheads_{args.num_heads}_nlayers_{args.num_layers}_{time.time()}"
dirprefix = f"{args.savedir}/cache/{args.wandb_group_name}/{fileprefix}"
if os.path.exists(dirprefix) == False:
    os.makedirs(dirprefix, exist_ok=True)
exp_name = f"{dirprefix}/{fileprefix}.pkl"
print("Saving to", exp_name)
num_iters_per_epoch = args.num_iters_per_epoch
num_apppearances = np.zeros(args.K)
test_every = 1 # np.log10(args.K).astype(int) * 2
all_sequences_across_switches = {
    "sequences": [],
    "switch_start_iter": [],
} 
print("num_iters_per_epoch", num_iters_per_epoch, "num_iters", args.num_iters)
# save all sequences in pickle file
all_sequences_across_switches["sequences"].append(copy.deepcopy(train_dataset.sequences))
all_sequences_across_switches["switch_start_iter"].append([0] * len(train_dataset.sequences))
with open(f"{dirprefix}/{fileprefix}_all_sequences.pkl", "wb") as f:
    pickle.dump(all_sequences_across_switches, f)
for iter in range(int(args.num_iters // num_iters_per_epoch)):
    # record[f"model_{iter}"] = copy.deepcopy(model.state_dict())
    # Switch the sequences half way through the training
    if iter == int(args.num_iters // num_iters_per_epoch / 2) and args.is_resample_tasks == "True": # resample the tasks
        # save the model linear modules for c_attn 
        record["mlp_state_dict"]["pre_switch"] = {
            "linear_modules": copy.deepcopy(model.linear_modules.state_dict()),
            "lm_head": copy.deepcopy(model.lm_head.state_dict()),
        } 
        # generate new sequences
        train_dataset.generate_sequences()
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            sampler=train_sampler, 
                                            **train_kwargs)  
        # save the sequences to pickle file
        all_sequences_across_switches["sequences"].append(copy.deepcopy(train_dataset.sequences))
        all_sequences_across_switches["switch_start_iter"].append([iter] * len(train_dataset.sequences))
        
        
        with open(f"{dirprefix}/{fileprefix}_all_sequences.pkl", "wb") as f:
            pickle.dump(all_sequences_across_switches, f)
        
        if args.sequence_sampling_distribution == "zipf":
            iwl_dataset.sequences = torch.cat(all_sequences_across_switches["sequences"], dim=0)[::10]
            # test on every 10th sequence in the train set
            iwl_test_loader = torch.utils.data.DataLoader(iwl_dataset,
                                            sampler=val_sampler,
                                            **{'batch_size': args.batch_size, 'num_workers': args.workers,
                                        "shuffle": False,
                                        'pin_memory': True})
        num_apppearances = np.zeros(args.K)
         
    logs = {
        "num_apppearances": copy.deepcopy(num_apppearances),
    }

    if (iter % test_every == 0):
        print("iter", iter, "testing", "length of test set", len(iwl_dataset.sequences))
        test_metrics = validate_gradient_descent_zipf(iter, iwl_test_loader, model, args, criterion, device)
    else:
        test_metrics = {}
     
        
    
    losses = utils.AverageMeter('Loss', ':.4e')
    ridge_losses = utils.AverageMeter('Ridge Loss', ':.4e')
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
        logsoftmax = F.log_softmax(preds[:,:-1,:], dim=-1).reshape(B * (N-1), D)
        logsoftmaxloss = criterion(logsoftmax, seq[:,1:].reshape(B * (N-1)))
        accuracy = (logsoftmax.argmax(dim=-1) == seq[:,1:].reshape(B * (N-1))).float().mean() 
        
        logsoftmaxloss = logsoftmaxloss.reshape(B, N-1).mean(dim=-1) # shape: (B,) 
        appearances.append(task.detach().cpu().numpy())
        loss_per_appearance.append(logsoftmaxloss.detach().cpu().numpy())
        loss = logsoftmaxloss.mean()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), seq.size(0)) 
        
    # scheduler.step()
    # print ("loss" , loss, "preds", preds.shape, "target", seq.shape)
    
    logs.update({
            "train_loss": losses.avg,
            "epoch": iter,
            "lr": optimizer.param_groups[0]['lr'],  
            "loss_per_appearance": (loss_per_appearance),
            "appearances": (appearances),
        })
    print("iter", iter, "loss", losses.avg)
    if iter == (args.num_iters // num_iters_per_epoch) - 1: 
        if args.sequence_sampling_distribution == "zipf":
            test_metrics = validate_gradient_descent_zipf(iter, iwl_test_loader, model, args, criterion, device)
        else:
            test_losses, test_top1 = validate_gradient_descent(iter, train_loader, model, args, criterion, device)            
    if args.sequence_sampling_distribution == "zipf":
        logs["test_metrics"] = test_metrics
    # elif args.sequence_sampling_distribution == "uniform":
    #     for i in range(args.len_context):
    #         logs[f"test_loss_{i}"] = test_losses[i].avg
    #         logs[f"test_top1_{i}"] = test_top1[i].avg
    
    if args.wandb_log:
        wandb.log(logs)
    else:
        # wandb.log(logs)
        record["logs"].append(logs)
    
 
    # save phi_xt_list_epoch 

    if iter % 50 == 0 and args.wandb_log != True:
        
        with open(exp_name, "wb") as f:
            pickle.dump(record, f)
            
        # save model state_dict
        # with open(f"{dirprefix}/{fileprefix}_model_iter_{iter}.pkl", "wb") as f:
            # pickle.dump(model.state_dict(), f)
        # use json
        # with open(exp_name, "w") as f:
            # json.dump(record, f)
# probe the swapped mlp model on original sequences
# probe_metrics = probe_gradient_descent_swapped_mlp(iter, iwl_test_loader, model, args, criterion, device, record)
# record["probe_metrics"] = probe_metrics
record["model_state_dict"] = copy.deepcopy(model.state_dict())
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
