
def langevin(grad_f, x0, tau=0.01, max_iter=100):
    d = x0.shape[0]
    for it in range(max_iter):
        noise = np.random.normal(0, 1, d)
        x0 = x0 + tau * grad_f(x0) + np.sqrt(2*tau) * noise
        yield x0

def primal_dual_sampling(f, grad_f, x0, y0, tau=0.001, max_iter=100):
    d = y0.shape[0]
    for it in range(max_iter):
        if it % 100:
            y0 = np.zeros(d)
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
        yield x0,y0

