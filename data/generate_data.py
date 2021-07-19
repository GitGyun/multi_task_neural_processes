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

def generate_data(n_datasets, n_examples, random_range=True, noised=False):
    meta_info = {}
    X = []
    Y = {task: [] for task in tasks}
    for dataset in range(n_datasets):
        meta_info[dataset] = {}

        center = False
        if random_range:
            center = 3*(random.random()*2 - 1) # -3 to 3
        meta_info[dataset]['center'] = center

        x = 5*(torch.rand(n_examples, 1)*2 - 1) + center # center - 5 to center + 5
        X.append(x)

        a = (2*(random.random() > 0.5) - 1)*torch.exp(torch.rand(1, 1) - 0.5) # +- e^-0.5 to e^0.5
        w = (2*(random.random() > 0.5) - 1)*torch.exp(torch.rand(1, 1) - 0.5) # +- e^-0.5 to e^0.5
        b = 4*torch.rand(1, 1) - 2 # -2 to 2
        c = 4*torch.rand(1, 1) - 2 # -2 to 2
        
        meta_info[dataset]['a'] = a
        meta_info[dataset]['w'] = w
        meta_info[dataset]['b'] = b
        meta_info[dataset]['c'] = c
        
        for task in tasks:
            if noised:
                noise = a*0.05*torch.randn(x.size())
            else:
                noise = 0
                
            y = a*activations[task](w*x + b) + c + noise
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
    parser.add_argument('--random_range', '-rr', default=False, action='store_true')
    parser.add_argument('--noised', default=False, action='store_true')
    parser.add_argument('--postfix', '-ptf', type=str, default='')
    parser.add_argument('--name', type=str, default='multi_data')
    args = parser.parse_args()

    X, Y, meta_info = generate_data(args.n_datasets, args.n_examples, args.random_range, args.noised)
    if args.noised:
        args.postfix += '_noised'
    args.name += args.postfix
    torch.save((X, Y, meta_info), '{}.pth'.format(args.name))

    plt.figure(figsize=(40, 12))
    for dataset in range(10):
        plt.subplot(2, 5, dataset+1)
        interval = torch.linspace(meta_info[dataset]['center'] - 5, meta_info[dataset]['center'] + 5, args.n_examples).unsqueeze(1)
        for task in Y:
            values = meta_info[dataset]['a']*activations[task](meta_info[dataset]['w']*interval + meta_info[dataset]['b']) + meta_info[dataset]['c']
            if args.noised:
                values_noised = values + 0.05*meta_info[dataset]['a']*torch.randn(interval.size())
                plt.scatter(interval, values_noised, color=colors[task], s=3, alpha=0.3)
            plt.plot(interval, values, color=colors[task])

    plt.savefig('{}.png'.format(args.name))
