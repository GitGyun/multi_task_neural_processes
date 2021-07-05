import os
import argparse

import torch
import torch.nn.functional as F

from data import load_test_data, preprocess_mtp_data
from model import get_model
from trainer import infer_mtp
from generate_data import colors, g_list
import itertools

torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/multi_data_fr.pth')
parser.add_argument('--eval_dir', type=str, default='runs_mtp2')
parser.add_argument('--eval_name', type=str, default='')
parser.add_argument('--target_tasks', nargs='+', default=[0, 1, 2, 3])
parser.add_argument('--n_contexts', '-nc', type=int, default=200)
parser.add_argument('--n_targets', '-nt', type=int, default=1000)
parser.add_argument('--n_samples', '-nsp', type=int, default=100)
parser.add_argument('--n_seeds', '-nsd', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)

args_meta = parser.parse_args()


args_meta.eval_dir = os.path.join('experiments', args_meta.eval_dir)
device = torch.device('cuda')
test_domains = range(950, 1000)

if args_meta.eval_name == '':
    eval_list = os.listdir(args_meta.eval_dir)
else:
    eval_list = [args_meta.eval_name]
    

target_task = 0    
source_tasks = [0,1,2,3]

for exp_name in eval_list:
    ckpt_path = os.path.join(args_meta.eval_dir, exp_name, 'checkpoints', 'best.pth')
    if not os.path.exists(ckpt_path):
            continue
        
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt['args']
    
    model = get_model('mtp', args, device)
    model.load_state_dict(ckpt['model'])
    
    print('evaluating {}'.format(exp_name))
    
    args_meta.tasks = args.tasks
    args_meta.target_tasks = args.target_tasks

    for seed in range(args_meta.n_seeds):
        *test_data, domain_info = load_test_data(args_meta, test_domains, droprate=0., seed=args_meta.seed)
        X_context_test, Y_context_test, X_target_test, Y_target_test = test_data

        random_idx = torch.randperm(args_meta.n_contexts)[:20]
 
        X_context_batch = X_context_test[:,random_idx,:].clone()
        Y_context_batch = Y_context_test[:,random_idx,:].clone()
        X_target_batch = X_target_test
        Y_target_batch = Y_target_test
        Y_context_batch[:, 3:, 0] = float('nan')
        Y_context_batch[:, 3:, 1] = float('nan')
        Y_context_batch[:, 3:, 2] = float('nan')
        Y_context_batch[:, 3:, 3] = float('nan')
        
        X_context_batch = X_context_batch.to(device)
        Y_context_batch = Y_context_batch.to(device)
        X_target_batch = X_target_batch.to(device)
        Y_target_batch = Y_target_batch.to(device)
        
        Y_preds_mean, Y_preds_var = infer_mtp(model, X_context_batch, Y_context_batch, X_target_batch, 1, mean_z = True, mean_v = True, var = True)
        Y_preds_mean = torch.stack([torch.stack([p.cpu() for p in pred]) for pred in Y_preds_mean])
        Y_preds_mean = Y_preds_mean.permute(1,2,3,0,4).squeeze(-1)
        Y_preds_var = torch.stack([torch.stack([p.cpu() for p in pred]) for pred in Y_preds_var])
        Y_preds_var = Y_preds_var.permute(1,2,3,0,4).squeeze(-1)

        Z_preds_mean = infer_mtp(model, X_context_batch, Y_context_batch, X_target_batch, args_meta.n_samples, mean_z = False, mean_v = True)
        Z_preds_mean = torch.stack([torch.stack([p.cpu() for p in pred]) for pred in Z_preds_mean])
        Z_preds_mean = Z_preds_mean.permute(1,2,3,0,4).squeeze(-1)

        V_preds_mean = infer_mtp(model, X_context_batch, Y_context_batch, X_target_batch, args_meta.n_samples, mean_z = True, mean_v = False)
        V_preds_mean = torch.stack([torch.stack([p.cpu() for p in pred]) for pred in V_preds_mean])
        V_preds_mean = V_preds_mean.permute(1,2,3,0,4).squeeze(-1)
        
        Z_preds_var = torch.std(Z_preds_mean, dim = 0).unsqueeze(0)
        V_preds_var = torch.std(V_preds_mean, dim = 0).unsqueeze(0)

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
        
        os.makedirs(os.path.join(args_meta.eval_dir, exp_name, 'mtp_task-effect-5'), exist_ok = True)
        torch.save({'pred_yt': Y_preds_mean, # N_samples, N_domains, N_target, NT 
                    'pred_yt_var' : Y_preds_var,
                    'z_pred_yt': Z_preds_mean,
                    'z_pred_yt_var': Z_preds_var,
                    'v_pred_yt': V_preds_mean,
                    'v_pred_yt_var': V_preds_var,
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
                    , os.path.join(args_meta.eval_dir, exp_name, 'mtp_task-effect-5','eval_seed{}_additionaltask-original'.format(seed)))
        
        for source_task in source_tasks:
            for variance_type in ["y", "z", "v"]:
                if variance_type == "y":
                    calc_max_variance_point = torch.argmax(Y_preds_var.squeeze(0), 1)
                elif variance_type == "z":
                    calc_max_variance_point = torch.argmax(Z_preds_var.squeeze(0), 1)
                elif variance_type == "v":
                    calc_max_variance_point = torch.argmax(V_preds_var.squeeze(0), 1)
                
                calc_max_variance_point = calc_max_variance_point[...,target_task]
                
                additional_xs = [X_target_batch[dom, calc_max_variance_point[dom],0].item() for dom in range(50)]
                additional_ys = torch.stack([
                        torch.Tensor([domain_info[test_domains[dom]]['a']*g(domain_info[test_domains[dom]]['w']*additional_xs[dom] + domain_info[test_domains[dom]]['b']) + domain_info[test_domains[dom]]['c'] for g in g_list])
                        for dom in range(50)])
                additional_xs = torch.Tensor(additional_xs).to(device).unsqueeze(-1).unsqueeze(-1)
                X_context_batch_task = torch.cat((X_context_batch, additional_xs), dim = 1)
                additional_ys = additional_ys.unsqueeze(1)
                Y_context_batch_mask = additional_ys.clone().to(device)

                for t in [0,1,2,3]:
                    """
                    only mask target
                    """
                    if t!= source_task:
                        Y_context_batch_mask[...,t] = float('nan')

                Y_context_batch_task = torch.cat((Y_context_batch, Y_context_batch_mask), dim = 1)
            
                Y_preds_mean_task, Y_preds_var_task = infer_mtp(model, X_context_batch_task, Y_context_batch_task, X_target_batch, 1, mean_z = True, mean_v = True, var = True)
                Y_preds_mean_task = torch.stack([torch.stack([p.cpu() for p in pred]) for pred in Y_preds_mean_task])
                Y_preds_mean_task = Y_preds_mean_task.permute(1,2,3,0,4).squeeze(-1)
                Y_preds_var_task = torch.stack([torch.stack([p.cpu() for p in pred]) for pred in Y_preds_var_task])
                Y_preds_var_task = Y_preds_var_task.permute(1,2,3,0,4).squeeze(-1)

                Z_preds_mean_task = infer_mtp(model, X_context_batch_task, Y_context_batch_task, X_target_batch, args_meta.n_samples, mean_z = False, mean_v = True)
                Z_preds_mean_task = torch.stack([torch.stack([p.cpu() for p in pred]) for pred in Z_preds_mean_task])
                Z_preds_mean_task = Z_preds_mean_task.permute(1,2,3,0,4).squeeze(-1)

                V_preds_mean_task = infer_mtp(model, X_context_batch_task, Y_context_batch_task, X_target_batch, args_meta.n_samples, mean_z = True, mean_v = False)
                V_preds_mean_task = torch.stack([torch.stack([p.cpu() for p in pred]) for pred in V_preds_mean_task])
                V_preds_mean_task = V_preds_mean_task.permute(1,2,3,0,4).squeeze(-1)
        
                Z_preds_var_task = torch.std(Z_preds_mean_task, dim = 0).unsqueeze(0)
                V_preds_var_task = torch.std(V_preds_mean_task, dim = 0).unsqueeze(0)

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

                    test_mse = F.mse_loss((Y_preds_mean_task.mean(0)[d_idx] - c) / a,
                              (Y_target_batch[d_idx].cpu() - c) / a, reduction='none').mean(0)
                    test_mses.append(test_mse)
                    
                torch.save({'pred_yt': Y_preds_mean_task, # N_samples, N_domains, N_target, NT
                            'pred_yt_var' : Y_preds_var_task,
                            'z_pred_yt': Z_preds_mean_task,
                            'z_pred_yt_var': Z_preds_var_task,
                            'v_pred_yt': V_preds_mean_task,
                            'v_pred_yt_var': V_preds_var_task,
                            'gt_xc' : X_context_batch_task, # N_domains, N_context, NT
                            'gt_yc' : Y_context_batch_task, # N_domains, N_context, NT
                            'gt_xt' : X_target_batch, # N_domains, N_target, NT
                            'gt_yt' : Y_target_batch, # N_domains, N_target, NT
                            'test_mses' : test_mses, # N_domains, NT
                            'domain_c' : domain_info_c, # N_domains
                            'domain_a' : domain_info_a, # N_domains
                            'domain_w' : domain_info_w,
                            'domain_b' : domain_info_b
                           }
                         , os.path.join(args_meta.eval_dir, exp_name, 'mtp_task-effect-5','eval_seed{}_additionaltask-{}-variancetype-{}'.format(seed,source_task, variance_type)))
