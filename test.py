import os
import argparse
import tqdm
import yaml
from easydict import EasyDict

import torch

from dataset import load_data, to_device
from model import get_model
from train import evaluate


# ENVIRONMENTAL SETTINGS
# to prevent over-threading
torch.set_num_threads(1)

DATASETS = ['synthetic', 'weather']
CHECKPOINTS = ['best_nll', 'best_error', 'last']
SPLITS = ['test', 'valid']

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', choices=DATASETS)
parser.add_argument('--eval_root', type=str, default='experiments')
parser.add_argument('--eval_dir', type=str, default='')
parser.add_argument('--eval_name', type=str, default='')
parser.add_argument('--eval_ckpt', type=str, default='best_error', choices=CHECKPOINTS)
parser.add_argument('--split', type=str, default='test', choices=SPLITS)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--reset', default=False, action='store_true')
parser.add_argument('--verbose', '-v', default=False, action='store_true')
parser.add_argument('--use_valid_imputer', '-uvi', default=False, action='store_true')
parser.add_argument('--use_homogeneous_imputer', '-uhi', default=False, action='store_true')

parser.add_argument('--cs', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--global_batch_size', type=int, default=16)

args = parser.parse_args()


# load test config
with open(os.path.join('configs', args.data, 'config_test.yaml')) as f:
    config_test = EasyDict(yaml.safe_load(f))
config_test[f'cs_{args.split}'] = args.cs
config_test[f'gamma_{args.split}'] = args.gamma
config_test.seed = args.seed
if args.eval_dir != '':
    config_test.eval_dir = args.eval_dir

# set device and evaluation directory
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device('cuda')
config_test.eval_dir = os.path.join(args.eval_root, config_test.eval_dir)
if args.eval_name == '':
    eval_list = os.listdir(config_test.eval_dir)
else:
    eval_list = [args.eval_name]
    
# load test dataloader
test_loader = load_data(config_test, device, split=args.split)

# test models in eval_list
for exp_name in eval_list:
    # skip if checkpoint not exists or still running
    eval_path = os.path.join(config_test.eval_dir, exp_name, 'checkpoints', f'{args.eval_ckpt}.pth')
    # last_path = os.path.join(config_test.eval_dir, exp_name, 'checkpoints', 'last.pth')
    last_path = eval_path
    if not (os.path.exists(eval_path) and os.path.exists(last_path)):
        if args.verbose:
            print(f'checkpoint of {exp_name} does not exist or still running - skip...')
        continue
    
    # skip if already tested
    result_dir = os.path.join(config_test.eval_dir, exp_name, 'results')
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f'result_cs{args.cs}_gamma{args.gamma}_seed{args.seed}_{args.split}_from{args.eval_ckpt}.pth')
    if os.path.exists(result_path) and not args.reset:
        if args.verbose:
            print(f'result of {exp_name} already exists - skip...')
        continue
    
    # load model and config
    ckpt = torch.load(eval_path, map_location=device)
    config = ckpt['config']
    params = ckpt['model']
    
    model = get_model(config, device)
    model.load_state_dict_(params)
    
    # load imputer
    if config.model == 'jtp' and config_test[f'gamma_{args.split}'] > 0:
        if args.use_valid_imputer:
            imputer_path = config.imputer_path
        elif args.use_homogeneous_imputer:
            imputer_path = eval_path.replace('jtp', 'stp')
        else:
            imputer_path = config_test.imputer_path
        
        assert os.path.exists(imputer_path)
        ckpt_imputer = torch.load(imputer_path)
        config_imputer = ckpt_imputer['config']
        params_imputer = ckpt_imputer['model']
        
        imputer = get_model(config_imputer, device)
        imputer.load_state_dict_(params_imputer)
    else:
        imputer = config_imputer = None
    
    if args.verbose:
        print('evaluating {} with test seed {} and gamma {} on {} data'.format(exp_name, args.seed, args.gamma, args.split))
    
    # evaluate and save results
    nlls, errors = evaluate(model, test_loader, device, config_test, imputer=imputer, config_imputer=config_imputer)
    if args.verbose:
        print(f'nll: {nlls}\nmse:{errors}')
    torch.save({'nlls': nlls, 'errors': errors, 'global_step': ckpt['global_step']}, result_path)
