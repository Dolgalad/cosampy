import numpy as np

class HammingBall:
    def __init__(self, x0, r=1):
        self.x0 = x0
    def __len__(self):
        return self.x0.shape[0]
    def __getitem__(self, i):
        x = np.copy(self.x0)
        x[i] = not x[i]
        return x
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
    def sample(self, weights=None):
        i = np.random.choice(range(self.__len__()), p=weights)
        return self.__getitem__(i)

def gibbs(f, x0, max_iter=100, neighborhood=HammingBall):
    for it in range(max_iter):
        # sample a neighbor
        Nx0 = neighborhood(x0)
        w = np.array([np.exp(.5 * (f(x)-f(x0))) for x in Nx0])
        w /= np.sum(w)
        x = Nx0.sample(weights=w)
        # compute acceptance rate
        Zx0 = sum(np.exp(.5 * (f(y)-f(x0))) for y in Nx0)
        Zx = sum(np.exp(.5 * (f(y)-f(x))) for y in neighborhood(x))
        accept_rate = min(1.0, Zx0/Zx)
        if np.random.rand() < accept_rate:
            yield x
            x0 = x
        else:
            yield x0


def primal_dual_sampling(f, grad_f, x0, y0, tau=0.0001, max_iter=100, neighborhood=HammingBall):
    d = y0.shape[0]
    for it in range(max_iter):
        # primal sampling step
        # sample a neighbor
        Nx0 = neighborhood(x0)
        w = np.array([np.exp(.5 * (f(x,y0)-f(x0,y0))) for x in Nx0])
        w /= np.sum(w)
        x = Nx0.sample(weights=w)
        # compute acceptance rate
        Zx0 = sum(np.exp(.5 * (f(xp,y0)-f(x0,y0))) for xp in Nx0)
        Zx = sum(np.exp(.5 * (f(xp,y0)-f(x,y0))) for xp in neighborhood(x))
        accept_rate = min(1.0, Zx0/Zx)
        if np.random.rand() < accept_rate:
            x0 = x

        # dual step
        noise = np.random.normal(0,1,d)
        y0 = y0 - tau * grad_f(x0,y0) + np.sqrt(2*tau) * noise
        y0 = - tau * grad_f(x0,y0)

        yield x0,y0

