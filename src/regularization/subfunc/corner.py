import numpy as np
import matplotlib.pyplot as plt

def corner(rho, eta, fig=0):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    n = len(rho)
    if n != len(eta):
        raise ValueError('Vectors rho and eta must have the same length')
    if n < 3:
        raise ValueError('Vectors rho and eta must have at least 3 elements')
    
    rho = np.asarray(rho).reshape(-1, 1)
    eta = np.asarray(eta).reshape(-1, 1)
    
    if fig < 0:
        fig = 0
    
    info = 0
    
    fin = np.isfinite(rho + eta)
    nzr = rho * eta != 0
    kept = np.where(fin & nzr)[0]
    
    if len(kept) == 0:
        raise ValueError('Too many Inf/NaN/zeros found in data')
    
    if len(kept) < n:
        info += 1
        print('Bad data - Inf, NaN or zeros found in data\n'
              '         Continuing with the remaining data')
    
    rho = rho[kept]
    eta = eta[kept]
    
    if np.any(rho[:-1] < rho[1:]) or np.any(eta[:-1] > eta[1:]):
        info += 10
        print('Lack of monotonicity')
    
    nP = len(rho)
    P = np.column_stack((np.log10(rho), np.log10(eta)))
    V = P[1:] - P[:-1]
    v = np.sqrt(np.sum(V ** 2, axis=1))
    W = V / v.reshape(-1, 1)
    clist = []
    p = min(5, nP - 1)
    convex = False
    
    Y, I = np.sort(v), np.argsort(v)[::-1]
    
    while p < (nP - 1) * 2:
        elmts = np.sort(I[:min(p, nP - 1)])
        
        candidate = Angles(W[elmts], elmts)
        if candidate > 0:
            convex = True
        if candidate and candidate not in clist:
            clist.append(candidate)
        
        candidate = Global_Behavior(P, W[elmts], elmts)
        if candidate and candidate not in clist:
            clist.append(candidate)
        
        p *= 2
    
    if not convex:
        k_corner = None
        info += 100
        print('Lack of convexity')
        return k_corner, info
    
    clist.sort()
    if 1 not in clist:
        clist.insert(0, 1)
    
    vz = np.where(np.diff(P[clist, 1]) >= np.abs(np.diff(P[clist, 0])))[0]
    if len(vz) > 1:
        if vz[0] == 0:
            vz = vz[1:]
    elif len(vz) == 1:
        if vz[0] == 0:
            vz = []
    
    if len(vz) == 0:
        index = clist[-1]
    else:
        vects = np.column_stack((P[clist[1:], 0] - P[clist[:-1], 0], P[clist[1:], 1] - P[clist[:-1], 1]))
        vects /= np.sqrt(np.sum(vects ** 2, axis=1)).reshape(-1, 1)
        delta = vects[:-1, 0] * vects[1:, 1] - vects[1:, 0] * vects[:-1, 1]
        vv = np.where(delta[vz - 1] <= 0)[0]
        if len(vv) == 0:
            index = clist[vz[-1]]
        else:
            index = clist[vz[vv[0]]]
    
    k_corner = kept[index - 1]
    
    if fig:
        plt.figure(fig)
        plt.clf()
        diffrho2 = (np.max(P[:, 0]) - np.min(P[:, 0])) / 2
        diffeta2 = (np.max(P[:, 1]) - np.min(P[:, 1])) / 2
        plt.loglog(rho, eta, 'k--o')
        plt.axis('square')
        plt.loglog([np.min(rho) / 100, rho[index - 1]], [eta[index - 1], eta[index - 1]], ':r',
                   [rho[index - 1], rho[index - 1]], [np.min(eta) / 100, eta[index - 1]], ':r')
        
        if np.abs(diffrho2) > np.abs(diffeta2):
            ax = [np.min(P[:, 0]), np.max(P[:, 0]), np.min(P[:, 1]) + diffrho2, np.max(P[:, 1]) + diffrho2]
        else:
            ax = [np.min(P[:, 0]) + diffeta2, np.max(P[:, 0]) + diffeta2, np.min(P[:, 1]), np.max(P[:, 1])]
        
        ax = np.power(10, ax)
        ax[0] /= 2
        plt.axis(ax)
        plt.xlabel('residual norm || A x - b ||')
        plt.ylabel('solution (semi)norm || L x ||')
        plt.title(f'Discrete L-curve, corner at {k_corner}')
    
    return k_corner, info


def Angles(W, kv):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    delta = W[:-1, 0] * W[1:, 1] - W[1:, 0] * W[:-1, 1]
    mm, kk = np.min(delta), np.argmin(delta)
    if mm < 0:
        index = kv[kk] + 1
    else:
        index = 0
    return index


def Global_Behavior(P, vects, elmts):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    hwedge = np.abs(vects[:, 1])
    An, In = np.sort(hwedge), np.argsort(hwedge)
    
    count = 1
    ln = len(In)
    mn = In[0]
    mx = In[ln - 1]
    
    while mn >= mx:
        mx = max([mx, In[ln - count]])
        count += 1
        mn = min([mn, In[count - 1]])
    
    if count > 1:
        I = 0
        J = 0
        for i in range(count):
            for j in range(ln, ln - count, -1):
                if In[i] < In[j]:
                    I = In[i]
                    J = In[j]
                    break
            if I > 0:
                break
    else:
        I = In[0]
        J = In[ln - 1]
    
    x3 = P[elmts[J] + 1, 0] + (P[elmts[I], 1] - P[elmts[J] + 1, 1]) / (P[elmts[J] + 1, 1] - P[elmts[J], 1]) * (
                P[elmts[J] + 1, 0] - P[elmts[J], 0])
    origin = np.array([x3, P[elmts[I], 1]])
    
    dists = (origin[0] - P[:, 0]) ** 2 + (origin[1] - P[:, 1]) ** 2
    index = np.argmin(dists)
    
    return index
