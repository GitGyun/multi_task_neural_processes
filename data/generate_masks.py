import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--n_datasets', type=int, default=50)
parser.add_argument('--n_tasks', type=int, default=4)
parser.add_argument('--M', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='mask_indices')
parser.add_argument('--split', type=str, default='test')
args = parser.parse_args()

torch.manual_seed(args.seed)

mask = torch.zeros(args.n_datasets, args.M, args.n_tasks)
for i in range(args.n_tasks):
    dropidxs = torch.randperm(args.M)[:int(args.gamma * args.M)]
    mask[:, dropidxs, i] = 1

mask = mask.bool()
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
torch.save(mask, os.path.join(args.save_dir, 'mask_M{}_gamma{}_seed{}_{}'.format(args.M, args.gamma, args.seed, args.split)))