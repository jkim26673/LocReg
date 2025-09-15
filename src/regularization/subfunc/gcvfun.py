from numpy.linalg import norm
def gcvfun(lamb_da, s2, beta, delta0, mn, dsvd, nargin):
    
    if (nargin == 5):
        f = (lamb_da **2)/(s2 + lamb_da**2)
    else:
        f = lamb_da/(s2 + lamb_da)
    G = (norm(f*beta)**2 + delta0)/(mn + sum(f))**2

    return G
