import random
import os
import torch


tasks = ['sine', 'tanh', 'sigmoid', 'gaussian']
activations = {
    'sine': lambda x: torch.sin(x),
    'tanh': lambda x: torch.tanh(x),
    'sigmoid': lambda x: torch.sigmoid(x),
    'gaussian': lambda x: torch.exp(-x.pow(2))
}
colors = {
    'sine': 'r',
    'tanh': 'g',
    'sigmoid': 'b', 
    'gaussian': 'c'
}

def generate_data(n_datasets, n_examples, task_noise=False, independent=False):
    meta_info = {}
    X = []
    Y = {task: [] for task in tasks}
    for dataset in range(n_datasets):
        meta_info[dataset] = {}

        x = 5*(torch.rand(n_examples, 1)*2 - 1) # -5 to +5
        X.append(x)

        if not independent:
            a = torch.exp(torch.rand(1, 1) - 0.5) # e^-0.5 to e^0.5
            w = torch.exp(torch.rand(1, 1) - 0.5) # e^-0.5 to e^0.5
            b = 4*torch.rand(1, 1) - 2 # -2 to 2
            c = 4*torch.rand(1, 1) - 2 # -2 to 2
        
            if not task_noise:
                meta_info[dataset]['a'] = a
                meta_info[dataset]['w'] = w
                meta_info[dataset]['b'] = b
                meta_info[dataset]['c'] = c
        
        for task in tasks:
            if task_noise:
                a_ = a * torch.exp(torch.randn(1, 1) * 0.1)
                w_ = w * torch.exp(torch.randn(1, 1) * 0.1)
                b_ = b + torch.randn(1, 1) * 0.2
                c_ = c + torch.randn(1, 1) * 0.2
                meta_info[dataset][task] = {'a': a_, 'w': w_, 'b': b_, 'c': c_}
            elif independent:
                a_ = torch.exp(torch.rand(1, 1) - 0.5) # e^-0.5 to e^0.5
                w_ = torch.exp(torch.rand(1, 1) - 0.5) # e^-0.5 to e^0.5
                b_ = 4*torch.rand(1, 1) - 2 # -2 to 2
                c_ = 4*torch.rand(1, 1) - 2 # -2 to 2
                meta_info[dataset][task] = {'a': a_, 'w': w_, 'b': b_, 'c': c_}
            else:
                a_, w_, b_, c_ = a, w, b, c
                
            y = a_ * activations[task](w_ * x + b_) + c_
            Y[task].append(y)

    for dataset in range(n_datasets):
        ids = torch.randperm(len(X[dataset]))
        X[dataset] = X[dataset][ids]
        for task in tasks:
            Y[task][dataset] = Y[task][dataset][ids]

    X = torch.stack(X)
    Y = {task: torch.stack(Y[task]) for task in tasks}
    
    return X, Y, meta_info


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_datasets', type=int, default=1000)
    parser.add_argument('--n_examples', type=int, default=200)
    parser.add_argument('--task_noise', '-tn', default=False, action='store_true')
    parser.add_argument('--independent', '-ind', default=False, action='store_true')
    args = parser.parse_args()

    X, Y, meta_info = generate_data(args.n_datasets, args.n_examples, args.task_noise, args.independent)

    if args.task_noise:
        data_dir = 'synthetic_tn'
    elif args.independent:
        data_dir = 'synthetic_ind'
    else:
        data_dir = 'synthetic'
    name = ','.join(tasks) + f'_N{args.n_datasets}_n{args.n_examples}'
    os.makedirs(data_dir, exist_ok=True)
    torch.save((X, Y, meta_info), os.path.join(data_dir, f'{name}.pth'))
    
    plt.figure(figsize=(40, 12))
    for dataset in range(10):
        plt.subplot(2, 5, dataset+1)
        x = torch.linspace(-5, 5, args.n_examples).unsqueeze(1)
        
        for task in Y:
            if args.task_noise or args.independent:
                a = meta_info[dataset][task]['a']
                w = meta_info[dataset][task]['w']
                b = meta_info[dataset][task]['b']
                c = meta_info[dataset][task]['c']
            else:
                a = meta_info[dataset]['a']
                w = meta_info[dataset]['w']
                b = meta_info[dataset]['b']
                c = meta_info[dataset]['c']
                
            y = a * activations[task](w * x + b) + c
            plt.plot(x, y, color=colors[task])

    plt.savefig(os.path.join(data_dir, f'{name}.png'))
