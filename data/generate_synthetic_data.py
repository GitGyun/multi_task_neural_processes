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

def generate_data(n_datasets, n_examples):
    meta_info = {}
    X = []
    Y = {task: [] for task in tasks}
    for dataset in range(n_datasets):
        meta_info[dataset] = {}

        x = 5*(torch.rand(n_examples, 1)*2 - 1) # -5 to +5
        X.append(x)

        a = torch.exp(torch.rand(1, 1) - 0.5) # e^-0.5 to e^0.5
        w = torch.exp(torch.rand(1, 1) - 0.5) # e^-0.5 to e^0.5
        b = 4*torch.rand(1, 1) - 2 # -2 to 2
        c = 4*torch.rand(1, 1) - 2 # -2 to 2
        
        meta_info[dataset]['a'] = a
        meta_info[dataset]['w'] = w
        meta_info[dataset]['b'] = b
        meta_info[dataset]['c'] = c
        
        for task in tasks:        
            y = a*activations[task](w*x + b) + c
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
    args = parser.parse_args()

    X, Y, meta_info = generate_data(args.n_datasets, args.n_examples)

    name = ','.join(tasks) + f'_N{args.n_datasets}_n{args.n_examples}'
    os.makedirs('synthetic', exist_ok=True)
    torch.save((X, Y, meta_info), os.path.join('synthetic', f'{name}.pth'))
    
    plt.figure(figsize=(40, 12))
    for dataset in range(10):
        plt.subplot(2, 5, dataset+1)
        x = torch.linspace(-5, 5, args.n_examples).unsqueeze(1)
        
        a = meta_info[dataset]['a']
        w = meta_info[dataset]['w']
        b = meta_info[dataset]['b']
        c = meta_info[dataset]['c']
        for task in Y:
            y = a*activations[task](w*x + b) + c
            plt.plot(x, y, color=colors[task])

    plt.savefig(os.path.join('synthetic', f'{name}.png'))
