import argparse
import subprocess
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--eval_dir', type=str, default='runs_mtp2')
parser.add_argument('--eval_name', type=str, default='')
parser.add_argument('--n_procs', type=int, default=16)
parser.add_argument('--pid', type = int, default = 0)
args = parser.parse_args()

# n_complete_datas = [1,2,3]
n_complete_datas = [1]

source_target_pairs = [
    ([0],[1,2,3]),
    ([1],[0,2,3]),
    ([2],[0,1,3]),
    ([3],[0,1,2]),
    ([0,1],[2,3]),
    ([0,2],[1,3]),
    ([0,3],[1,2]),
    ([1,2],[0,3]),
    ([1,3],[0,2]),
    ([2,3],[0,1]),
    ([1,2,3],[0]),
    ([0,2,3],[1]),
    ([0,1,3],[2]),
    ([0,1,2],[3]),
]

prod = itertools.product(n_complete_datas, source_target_pairs)

for i, p in enumerate(prod):
    if i % args.n_procs == args.pid:
        n_complete_data, (source_tasks, target_tasks) = p
        source_tasks = [str(i) for i in source_tasks]
        target_tasks = [str(i) for i in target_tasks]
        source_tasks = " ".join(source_tasks)
        target_tasks = " ".join(target_tasks)
        command = 'python evaluate_task_transfer.py --eval_dir {} --eval_name {} --n_complete_data {} --source_tasks {} --target_tasks {}'.format(args.eval_dir,args.eval_name,n_complete_data, source_tasks, target_tasks)
        print(command)

        if args.eval_name != '':
            command += ' --eval_name {}'.format(args.eval_name)
        subprocess.call(command.split())
