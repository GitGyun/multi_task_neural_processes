import os
import argparse
import tqdm
import yaml
from easydict import EasyDict

import torch

from data import load_data, to_device
from model import get_model
from train import evaluate


### ENVIRONMENTAL SETTINGS
# to prevent over-threading
torch.set_num_threads(1)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', choices=['synthetic', 'synthetic_noised', 'synthetic_tasknoised', 'celeba'])
parser.add_argument('--eval_dir', type=str, default='')
parser.add_argument('--eval_name', type=str, default='')
parser.add_argument('--split', type=str, default='test', choices=['test', 'valid', 'subtrain'])
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--reset', default=False, action='store_true')
parser.add_argument('--verbose', '-v', default=False, action='store_true')

parser.add_argument('--M', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--global_batch_size', type=int, default=-1)

args = parser.parse_args()

# load test config
with open(os.path.join('configs', args.data, 'config_test.yaml')) as f:
    config_test = EasyDict(yaml.safe_load(f))
config_test.M = args.M
config_test.gamma_test = args.gamma
config_test.seed = args.seed
if args.global_batch_size > 0:
    config_test.global_batch_size = args.global_batch_size
if args.eval_dir != '':
    config_test.eval_dir = args.eval_dir
if args.data == 'synthetic_noised' or args.data == 'synthetic_tasknoised':
    config_test.noised = True
else:
    config_test.noised = False
if args.data == 'synthetic_tasknoised':
    config_test.tasknoised = True
else:
    config_test.tasknoised = False

# set device and evaluation directory
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device('cuda')
config_test.eval_dir = os.path.join('experiments', config_test.eval_dir)
if args.eval_name == '':
    eval_list = os.listdir(config_test.eval_dir)
else:
    eval_list = [os.path.join(config_test.eval_dir, args.eval_name)]
# load test dataloader
test_loader = load_data(config_test, device, split=args.split)

# test models in eval_list
for exp_name in eval_list:
    # skip if checkpoint not exists or still running
    best_path = os.path.join(config_test.eval_dir, exp_name, 'checkpoints', 'best.pth')
    last_path = os.path.join(config_test.eval_dir, exp_name, 'checkpoints', 'last.pth')
    if not (os.path.exists(best_path) and os.path.exists(last_path)):
        if args.verbose:
            print('checkpoint of {} does not exist or still running - skip...'.format(exp_name))
        continue
    
    # skip if already tested
    result_dir = os.path.join(config_test.eval_dir, exp_name, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, 'result_M{}_gamma{}_seed{}_{}.pth'.format(args.M, args.gamma, args.seed, args.split))
    if os.path.exists(result_path) and not args.reset:
        continue
    
    # load model and config
    ckpt = torch.load(best_path, map_location=device)
    config = ckpt['config']
    
    # for legacy
    if 'stochastic_path' not in config:
        config.stochastic_path = True
    if 'deterministic_path' not in config:
        config.deterministic_path = True
    if 'implicit_global_latent' not in config:
        config.implicit_global_latent = False
    if 'global_latent_only' not in config:
        config.global_latent_only = False
    if 'deterministic_path2' not in config:
        config.deterministic_path2 = False
    if 'context_posterior' not in config:
        config.context_posterior = False
    if 'epsilon' not in config:
        config.epsilon = 0.1
    model = get_model(config, device)
    model.load_state_dict(ckpt['model'])
    if args.verbose:
        print('evaluating {} with test seed {} and gamma {} on {} data'.format(exp_name, args.seed, args.gamma, args.split))
    
    # evaluate and save results
    errors = evaluate(model, test_loader, device, config_test)
    if args.verbose:
        print(errors)
    torch.save(errors, result_path)
