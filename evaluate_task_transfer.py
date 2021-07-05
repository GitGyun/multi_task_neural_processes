import os
import argparse

import torch
import torch.nn.functional as F

from data import load_test_data_task_transfer, preprocess_mtp_data
from model import get_model
from trainer import infer_mtp
from generate_data import colors, g_list
import itertools
from tqdm import tqdm

torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/multi_data_fr.pth')
parser.add_argument('--eval_dir', type=str, default='runs_mtp2')
parser.add_argument('--eval_name', type=str, default='')
parser.add_argument('--tasks', nargs='+', default=[0, 1, 2, 3])

parser.add_argument('--n_complete_data', type = int, default = 1)
parser.add_argument('--source_tasks', nargs = '+', default = [0])
parser.add_argument('--target_tasks', nargs = '+', default = [1,2,3])

parser.add_argument('--n_targets', '-nt', type=int, default=1000)
parser.add_argument('--n_samples', '-nsp', type=int, default = 5)
parser.add_argument('--n_seeds', '-nsd', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)

args_meta = parser.parse_args()
args_meta.eval_dir = os.path.join('experiments', args_meta.eval_dir)
device = torch.device('cuda')
test_domains = range(950, 1000)

ckpt_path = os.path.join(args_meta.eval_dir, args_meta.eval_name, 'checkpoints', 'best.pth')
if not os.path.exists(ckpt_path):
    import sys
    sys.exit(-1)

ckpt = torch.load(ckpt_path, map_location=device)
args = ckpt['args']

model = get_model('mtp', args, device)
model.load_state_dict(ckpt['model'])

n_complete_data = 1
source_tasks = args_meta.source_tasks
target_tasks = args_meta.target_tasks

save_dict = {}

args_meta.target_tasks = [int(i) for i in args_meta.target_tasks]
args_meta.source_tasks = [int(i) for i in args_meta.source_tasks]


for n_incomplete_data in tqdm(range(0, 60)):
    for seed in range(args_meta.n_seeds):
        *data, domain_info = load_test_data_task_transfer(args_meta, test_domains, n_incomplete_data)
        
        X_context_batch, Y_context_batch, X_target_batch, Y_target_batch = data
        
        X_context_batch = X_context_batch.to(device).float()
        Y_context_batch = Y_context_batch.to(device).float()
        X_target_batch = X_target_batch.to(device).float()
        Y_target_batch = Y_target_batch.to(device).float()
        
        Y_preds_mean = infer_mtp(model, X_context_batch, Y_context_batch, X_target_batch, args_meta.n_samples, mean_z = False, mean_v = False, var = False)
        Y_preds_mean = torch.stack([torch.stack([p.cpu() for p in pred]) for pred in Y_preds_mean])
        Y_preds_mean = Y_preds_mean.permute(1,2,3,0,4).squeeze(-1)
        
        test_mses = []
        domain_info_c = []
        domain_info_a = []
        domain_info_w = []
        domain_info_b = []
        for d_idx in range(len(test_domains)):
            c = domain_info[test_domains[d_idx]]['c']
            a = domain_info[test_domains[d_idx]]['a']
            w = domain_info[test_domains[d_idx]]['w']
            b = domain_info[test_domains[d_idx]]['b']
            
            domain_info_c.append(c)
            domain_info_a.append(a)
            domain_info_w.append(w)
            domain_info_b.append(b)
            
            test_mse = F.mse_loss((Y_preds_mean.mean(0)[d_idx] - c) / a,
                                  (Y_target_batch[d_idx].cpu() - c) / a, reduction='none').mean(0)
            test_mses.append(test_mse)
        
        save_dict[n_incomplete_data] = {'pred_yt': Y_preds_mean, # N_samples, N_domains, N_target, NT 
                                        'gt_xc' : X_context_batch, # N_domains, N_context, NT
                                        'gt_yc' : Y_context_batch, # N_domains, N_context, NT
                                        'gt_xt' : X_target_batch, # N_domains, N_target, NT
                                        'gt_yt' : Y_target_batch, # N_domains, N_target, NT
                                        'test_mses' : test_mses, # N_domains, NT
                                        'domain_c' : domain_info_c, # N_domains
                                        'domain_a' : domain_info_a, # N_domains
                                        'domain_w' : domain_info_w,
                                        'domain_b' : domain_info_b
                                       }

os.makedirs(os.path.join(args_meta.eval_dir, args.eval_name, 'mtp_task-transfer'), exist_ok = True)
torch.save(save_dict
            , os.path.join(args_meta.eval_dir, args.eval_name, 'mtp_task-transfer','eval_srctsk-{}_tartsk-{}-ncomp-{}'.format(args_meta.source_tasks, args_meta.target_tasks, args_meta.n_complete_data)))