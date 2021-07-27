import itertools
import subprocess

Ms = [5, 10, 20]
gammas = [0, 0.25, 0.5, 0.75]
seeds = [0, 1, 2, 3, 4]
# splits = ['test', 'valid', 'subtrain']
splits = ['subtrain']

prod = list(itertools.product(Ms, gammas, seeds, splits))
for i, p in enumerate(prod):
    command = 'python generate_masks.py --M {} --gamma {} --seed {} --split {}'.format(*p)
    print(command)
    subprocess.call(command.split())
