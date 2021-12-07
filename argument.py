import argparse


def str2bool(v):
    if v == 'True' or v == 'true': return True
    elif v == 'False' or v == 'false': return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
DATASETS = ['synthetic', 'weather']
MODELS = ['stp', 'jtp', 'mtp', 'mtp_s']
LR_SCHEDULES = ['constant', 'sqroot', 'cos', 'poly']
BETA_SCHEDULES = ['constant', 'linear_warmup']

# argument parser
parser = argparse.ArgumentParser()

# basic arguments
parser.add_argument('--data', type=str, default='synthetic', choices=DATASETS)
parser.add_argument('--model', type=str, default='mtp_s', choices=MODELS)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name', type=str, default='')
parser.add_argument('--log_root', type=str, default='experiments')
parser.add_argument('--imputer_path', type=str, default='')
parser.add_argument('--name_postfix', '-ptf', type=str, default='')
parser.add_argument('--debug_mode', '-debug', default=False, action='store_true')

# model-specific arguments
parser.add_argument('--dim_hidden', type=int, default=-1)
parser.add_argument('--pma', type=str2bool, default=True)

# training arguments
parser.add_argument('--n_steps', type=int, default=-1)
parser.add_argument('--global_batch_size', type=int, default=-1)
parser.add_argument('--lr', type=float, default=-1.)
parser.add_argument('--lr_schedule', '-lrs', type=str, default='', choices=LR_SCHEDULES)
parser.add_argument('--beta_T_schedule', '-bts', type=str, default='', choices=BETA_SCHEDULES)
parser.add_argument('--beta_G_schedule', '-bgs', type=str, default='', choices=BETA_SCHEDULES)
parser.add_argument('--gamma_train', '-gtr', type=float, default=-1)
parser.add_argument('--gamma_valid', '-gvl', type=float, default=-1)
parser.add_argument('--cs_range_train', '-csr', nargs='+', default=[])
parser.add_argument('--ts_train', '-ts', type=int, default=-1)

args = parser.parse_args()