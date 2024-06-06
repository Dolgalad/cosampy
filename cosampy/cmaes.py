""" Implementation of the CMAES optimization strategy
"""

import numpy as np

def cmaes(x0, func, n_samples=10, max_iter=100, noise_func=lambda n: np.random.normal(0,0.1,n), patience=10):
    history = [np.copy(x0)]
    xdim = np.shape(x0)
    patience_counter = 0
    it = 0
    best_val = 1e30
    while it < max_iter and patience_counter < patience:
        # sample points around current candidate
        x = x0 + noise_func((n_samples, *xdim))
        x = np.concatenate((x, [x0]))
        # evaluate func
        val = func(x)
        # set current candidate to argmin
        if np.min(val) >= best_val:
            patience_counter += 1
        else:
            best_val = np.min(val)
            patience_counter = 0
        x0 = x[np.argmin(val)]
        history.append(np.copy(x0))
        it += 1
    return np.array(history)

def cmaes2(x0, func, patience=100, sigma=0.3):
    history = [np.copy(x0)]
    N = x0.shape[0]
    it = 0
    patience_count = 0
    last_val = 1e30
    stopfitness = -1e30
    stopeval = 1e3 * N**2

    # Strategy parameter setting: Selection
    lmbda = 4 + int(np.floor(3 * np.log(N)))
    mu = lmbda / 2
    print(mu)
    weights = np.log(mu + .5) - np.log(np.arange(1, np.floor(mu+1)))
    mu = int(np.floor(mu))
    print(mu)
    weights = weights / np.sum(weights)
    mueff = np.sum(weights)**2 / np.sum(weights**2)

    # Strategy parameter setting: Adaptation
    cc = (4.0+mueff/N) / (N+4.0 + 2*mueff/N)
    cs = (mueff+2) / (N+mueff+5)
    c1 = 2.0 / ((N+1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff-2+1/mueff) / ((N+2)**2 + mueff))
    damps = 1.0 + 2*max(0, np.sqrt((mueff-1)/(N+1))-1) + cs

    # initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros(N)
    ps = np.zeros(N)
    B = np.eye(N)
    D = np.ones(N)
    C = np.matmul(np.matmul(B, np.diag(1.0 / D)), B.T)
    invsqrtC = np.matmul(np.matmul(B, np.diag(1.0 / D)), B.T)
    eigeneval = 0
    chiN = N**0.5 * (1 - 1/(4*N)+1/(21*N**2))
    counteval = 0

    while counteval < stopeval:
        # generate and evaluate offspring
        arx = np.zeros((lmbda, N))
        for k in range(lmbda):
            arx[k,:] = x0 + sigma * np.matmul(B, D * np.random.normal(0,1,N))
        arfitness = func(arx)
        counteval += lmbda

        # sort by fitness and compute weighted mean
        arindex = np.argsort(arfitness)
        arfitness = arfitness[arindex]
        xold = x0
        x0 = np.average(arx[arindex[0:mu],:], weights=weights, axis=0)
        history.append(np.copy(x0)) # update history

        # Cumulation: Update evolution paths
        ps = (1.0-cs)*ps + np.sqrt(cs*(2.0-cs)*mueff) * np.matmul(invsqrtC, x0-xold) / sigma
        hsig = np.linalg.norm(ps) / np.sqrt(1.0 - (1.0-cs)**(2*counteval/lmbda)) / chiN < 1.4 + 2.0/(N+1)
        pc = (1.0-cc)*pc + hsig * np.sqrt(cc * (2.0-cc) * mueff) * (x0 - xold) / sigma
        
        # Adapt covariance matrix C
        artmp = (1.0/sigma) * (arx[arindex[0:mu],:] - np.tile(xold, (mu, 1)))
        C = (1.0-c1-cmu) * C + c1 * ( np.matmul(pc.reshape(N,1), pc.reshape(1,N)) + (1.0-hsig) * cc*(2.0-cc) * C) + cmu * np.matmul(np.matmul(artmp.T, np.diag(weights)), artmp)

        # Adapt step size sigma
        sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1.0))

        # Decomposition of C into B*diag(D**2)*B.T (diagonalization)
        if counteval - eigeneval > lmbda/(c1+cmu)/N/10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C,1).T
            D,B = np.linalg.eig(C)
            D = np.sqrt(D)
            invsqrtC = np.matmul(B, np.matmul(np.diag(1.0 / D), B.T))

        # Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable
        if arfitness[0] < last_val:
            patience_count = 0
        else:
            patience_count += 1

        last_val = arfitness[0]
        it += 1
        print(it, counteval, stopeval, arfitness[0], np.max(D)/np.min(D), sigma, patience_count)
        if arfitness[0] <= stopfitness or np.max(D) > 1e7 * np.min(D) or patience_count>=patience:
            print("Stop condition : ", (arfitness[0]<=stopfitness), (np.max(D) > 1e7 * np.min(D)), (patience_count>=patience))
            break
    return np.array(history)

if __name__=="__main__":
    # test our implementation of cmaes
    #def f(x, A=10):
    #    if np.ndim(x) <= 1:
    #        n = 1
    #        r = A + x**2 - A * np.cos(2 * np.pi * x)
    #    else:
    #        n = np.prod(x.shape[1:])
    #        r = A*n + np.sum( x**2 - A * np.cos(2 * np.pi * x) , axis=1)
    #    return r

    #hist = cmaes(np.random.rand(), f, max_iter=100, n_samples=10)

    import matplotlib.pyplot as plt
    #fig,ax=plt.subplots(1,1)
    ##x = np.linspace(-5.12, 5.12, 1000)
    ##y = f(x)
    ##ax[0].plot(x, y)
    #vals = f(hist)
    #ax.plot(vals)

    # test CMAES for solving a ETSP with steps taken in the euclidean space
    from sampco.tsp import *
    N = 200
    pb = random_euclidean_tsp(N)
    sol0 = nearest_neighbor_route(pb)
    cost_sol0 = solution_cost(sol0, pb)
    def g(c):
        #print(c.shape)
        vals = []
        for i in range(c.shape[0]):
            pb1 = ETSP(c[i].reshape(N,2))
            sol = nearest_neighbor_route(pb1)
            #print(sol.sequence)
            vals.append(solution_cost(sol, pb))
        #return np.array(vals)
        return (np.array(vals) - cost_sol0) / cost_sol0
    hist = cmaes2(pb.pos.reshape(2*N), g, sigma=0.00000001, patience=1000)
    #print(hist)
    fig,ax=plt.subplots(1,1)
    vals = g(hist)
    ax.plot(vals)

    sol0 = nearest_neighbor_route(pb)
    print("Original cost : ", solution_cost(sol0, pb))

    plt.show()

    import os
    dirname = f"etsp_cmaes_test_{N}"
    if os.path.isdir(dirname):
        os.system(f"rm {dirname}/*")
    os.makedirs(dirname, exist_ok=True)

    last_val = 1e30
    for j,v in zip(range(len(hist)), vals):
        if v == last_val:
            continue
        else:
            last_val = v
        fig, ax = plt.subplots()
        x, y = pb.pos[:,0], pb.pos[:,1]
        n = np.arange(pb.number_of_nodes())
        ax.scatter(x, y)
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))
        
        pb1 = ETSP(hist[j].reshape(N,2))
        sol = nearest_neighbor_route(pb1)
        for i in range(1, len(sol.sequence)):
            plt.arrow(*pb.pos[sol.sequence[i-1]], *(pb.pos[sol.sequence[i]] - pb.pos[sol.sequence[i-1]]), head_width=.01, fc='k', ec='k', length_includes_head=True)
        plt.arrow(*pb.pos[sol.sequence[-1]], *(pb.pos[sol.sequence[0]] - pb.pos[sol.sequence[-1]]), head_width=.01, fc='k', ec='k', length_includes_head=True)
        plt.savefig(os.path.join(dirname, str(j)+".png"))
        plt.close(fig)



