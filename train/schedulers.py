import math


class LRScheduler(object):
    """Learning Rate Scheduler
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    """
    def __init__(self, optimizer, mode, base_lr, num_iters, warmup_iters=1000,
                 from_iter=0, decay_degree=0.9, decay_steps=5000):
        self.optimizer = optimizer
        self.mode = mode
        # print('Using {} LR Scheduler!'.format(self.mode))
        self.base_lr = base_lr
        self.lr = base_lr
        self.iter = from_iter
        self.N = num_iters + 1 
        self.warmup_iters = warmup_iters
        self.decay_degree = decay_degree
        self.decay_steps = decay_steps

    def step(self):
        self.iter += 1
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * self.iter / self.N * math.pi))
        elif self.mode == 'poly':
            if self.iter == self.N:
                lr = 0.0
            else:
                lr = self.lr * pow((1 - 1.0 * self.iter / self.N), self.decay_degree)
        elif self.mode == 'step':
            lr = self.lr * (0.1**(self.decay_steps // self.iter))
        elif self.mode == 'constant':
            lr = self.lr
        elif self.mode == 'sqroot':
            lr = self.lr * self.warmup_iters**0.5 * min(self.iter * self.warmup_iters**-1.5, self.iter**-0.5)
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and self.iter < self.warmup_iters:
            lr = lr * 1.0 * self.iter / self.warmup_iters
        assert lr >= 0
        self._adjust_learning_rate(self.optimizer, lr)
        
        return lr

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

    def reset(self):
        self.lr = self.base_lr
        self.iter = 0
        self._adjust_learning_rate(self.optimizer, self.lr)
        
        
class BetaScheduler:
    def __init__(self, mode, base_beta, n_steps, warmup_steps=10000):
        self.mode = mode
        self.base_beta = base_beta
        self.warmup_steps = warmup_steps
        self.n_steps = n_steps + 1
        self.iter = 0
        
    def step(self):
        self.iter += 1
        if self.mode == 'constant':
            return self.base_beta
        elif self.mode == 'linear_warmup':
            return min(1, (self.iter / self.warmup_steps)) * self.base_beta
#         elif self.mode == 'linear':
#             return (self.iter / self.n_steps) * (self.beta_last - self.beta_init) + self.beta_init
#         elif self.mode == 'inverse-linear':
#             return (1 - self.iter / self.n_steps) * (self.beta_last - self.beta_init) + self.beta_init
#         elif self.mode == 'cyclic':
#             cycles = 10
#             period = (self.n_steps - 1) // cycles
#             iter_p = (self.iter - 1) % period
#             return float(iter_p) / max(1, float(period - 1)) * (self.beta_last - self.beta_init) + self.beta_init
        
        
class GammaScheduler:
    def __init__(self, n_steps):
        self.n_steps = n_steps + 1
        self.iter = 0
        
    def step(self):
        self.iter += 1
        return (self.iter / self.n_steps)
