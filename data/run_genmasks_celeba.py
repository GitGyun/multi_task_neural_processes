import itertools
import subprocess

# Ms = [10, 100, 512]
Ms = [100]
gammas = [0, 0.25, 0.5, 0.75]
seeds = [0, 1, 2, 3, 4]
splits = ['test', 'valid']

prod = list(itertools.product(Ms, gammas, seeds, splits))
for i, p in enumerate(prod):
    command = 'python generate_masks.py --n_datasets 1500 --M {} --gamma {} --seed {} --split {} --postfix _celeba'.format(*p)
    print(command)
    subprocess.call(command.split())
