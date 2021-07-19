import argparse
import subprocess
import itertools
import multiprocessing as mp
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', choices=['synthetic', 'celeba'])
parser.add_argument('--eval_dir', type=str, default='')
parser.add_argument('--eval_name', type=str, default='')
parser.add_argument('--split', type=str, default='test', choices=['test', 'valid'])
parser.add_argument('--reset', default=False, action='store_true')
parser.add_argument('--devices', nargs='+', default=['0', '1', '2', '3'])
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--verbose', '-v', default=False, action='store_true')
args = parser.parse_args()

if args.data == 'synthetic':
    Ms = [5, 10, 20]
elif args.data == 'celeba':
    Ms = [10, 100, 512]
gammas = [0, 0.25, 0.5, 0.75]
seeds = [0, 1, 2, 3, 4]

prod = list(itertools.product(Ms, gammas, seeds))

def run_thread(ip):
    i, p = ip
    device = args.devices[i % len(args.devices)]
    command = 'python test.py --device {} --data {} --split {} --M {} --gamma {} --seed {}'.format(device, args.data, args.split, *p)
    if args.eval_dir != '':
        command += ' --eval_dir {}'.format(args.eval_dir)
    if args.eval_name != '':
        command += ' --eval_name {}'.format(args.eval_name)
    if args.reset:
        command += ' --reset'
    if args.verbose:
        command += ' --verbose'
    subprocess.call(command.split())
    
with mp.Pool(args.n_threads) as pool:
    pbar = tqdm.tqdm(total=len(prod), initial=0,
                     bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    for _ in pool.imap(run_thread, enumerate(prod), chunksize=1):
        pbar.update()
    pbar.close()