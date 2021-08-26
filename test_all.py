import argparse
import subprocess
import itertools
import multiprocessing as mp
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', choices=['template', 'synthetic', 'synthetic_noised', 'synthetic_tasknoised', 'celeba'])
parser.add_argument('--eval_dir', type=str, default='')
parser.add_argument('--eval_name', type=str, default='')
parser.add_argument('--eval_ckpt', type=str, default='best_error', choices=['best_nll', 'best_error', 'last'])
parser.add_argument('--split', type=str, default='test', choices=['test', 'valid', 'subtrain'])
parser.add_argument('--global_batch_size', type=int, default=16)
parser.add_argument('--reset', default=False, action='store_true')
parser.add_argument('--devices', nargs='+', default=['0', '1', '2', '3'])
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--verbose', '-v', default=False, action='store_true')
args = parser.parse_args()

if args.data == 'template' or args.data == 'synthetic' or args.data == 'synthetic_noised' or args.data == 'synthetic_tasknoised':
    cs_tests = [5, 10, 20]
elif args.data == 'celeba':
    cs_tests = [10, 100, 512]
gammas = [0, 0.25, 0.5, 0.75]
seeds = [0, 1, 2, 3, 4]

prod = list(itertools.product(cs_tests, gammas, seeds))

def run_thread(ip):
    i, p = ip
    device = args.devices[i % len(args.devices)]
    command = f'python test.py --device {device} --data {args.data} --split {args.split} --eval_ckpt {args.eval_ckpt} --global_batch_size {args.global_batch_size}' + ' --cs {} --gamma {} --seed {}'.format(*p)
    if args.eval_dir != '':
        command += f' --eval_dir {args.eval_dir}'
    if args.eval_name != '':
        command += f' --eval_name {args.eval_name}'
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