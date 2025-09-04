from NN_2D.STOUpozo import STOU
import numpy as np
import jax.numpy as jnp



def test_covs():
    Z = STOU()

    assert 1==1


p = 2
h_t = 0.5
h_s = 0.5
c=1
distances_XY = []
for t in reversed(range(p)): # t in {p, p-1, p-2, ..., 1, 0}
    b = np.floor(c*(t+1)*h_t/h_s)
    distances_XY.append(jnp.array([[v, -h_t*(t+1)] for v in jnp.arange(-b*h_s, (b+1)*h_s, h_s)]))
distances_XY = jnp.concat(distances_XY, axis=0)
#print(distances_XY)
assert(jnp.abs(distances_XY - jnp.array([[-1.,  -1. ],
    [-0.5, -1. ],
    [ 0.,  -1. ],
    [ 0.5, -1. ],
    [ 1.,  -1. ],
    [-0.5, -0.5],
    [ 0.,  -0.5],
    [ 0.5, -0.5]])).sum()<10e-10)
# assert(jnp.abs(distances_XY - jnp.array([[-1.  -1. ],
#     [-0.5 -1. ],
#     [ 0.  -1. ],
#     [ 0.5 -1. ],
#     [ 1.  -1. ],
#     [-0.5 -0.5],
#     [ 0.  -0.5],
#     [ 0.5 -0.5]]).sum()<0.0000001))


A_= 3
VarLevySeed_ = 0.5
r=4

def truncated_cov(u, tau, r):
    """
    returns Cov(Z_t(x)^(r), Z_{t+tau}(x+u)^(r)) = Var(Lambda') exp(-Au) int_{A_0(0)\V_{(0,0)}^r \cap A_{tau}(u)\V__{(tau,u)}^r} exp(2As) ds
    """
    # the formula below works for tau<=0, u in |R. If tau>0, we have to set tau=-tau, u=-u, as Cov(Z_tau(u)^r, Z_0(0)^r) = Cov(Z_0(0)^r, Z_{-tau}(-u)^r) because of stationarity
    if tau > 0:
         tau = -tau
         u = -u
    if tau <= -r:
            return 0
    #r = a-p
    int = c/A_ * (-np.exp(-2*A_*r)*(tau+r+1/(2*A_)) + np.exp(2*A_*tau)/(2*A_))

    return VarLevySeed_ * np.exp(-A_*u) * int

covs_XY = jnp.array([truncated_cov(u=dist[0], tau=dist[1], r=r) for dist in distances_XY])
#print(covs_XY)
#print(truncated_cov(u=1,tau=1,r=5))
#print(truncated_cov(u=1,tau=-1,r=5))
#print(truncated_cov(u=-1,tau=-1,r=5))

c_= c
p = 1
h_t = 1
h_s = 0.5
c=1

distances_XX = []
for t in range(p,0,-1): # t in {p, p-1, p-2, ..., 1}
    bt = int(np.floor(c_*t*h_t/h_s)) # bt:= argmax {a: a*h_s <= (t+1)*c*h_t}
    for pixel1 in jnp.arange(-bt*h_s,(bt+1)*h_s, h_s):
        dist_row = []
        for s in range(p,0,-1):
            bs = int(np.floor(c_*s*h_t/h_s))
            for pixel2 in jnp.arange(-bs*h_s, (bs+1)*h_s, h_s):
                 dist_row.append([float(pixel1-pixel2), -h_t*(t-s)])
        distances_XX.append(dist_row)
#dist_XX = jnp.array(dist_XX)
#for row in dist_XX:
#     print(row)

#print(dist_XX)

c_= 1
p = 1
h_t = 0.5
h_s = 1

p = 1
h_t =1
h_s = 1
c=1

distances_XX = []
covs_XX = []
for t in range(p,0,-1): # t in {p, p-1, p-2, ..., 1}
    bt = int(np.floor(c_*t*h_t/h_s)) # bt:= argmax {a: a*h_s <= (t+1)*c*h_t}
    for pixel1 in jnp.arange(-bt*h_s,(bt+1)*h_s, h_s):
        dist_row = []
        cov_row = []
        for s in range(p,0,-1):
            bs = int(np.floor(c_*s*h_t/h_s))
            for pixel2 in jnp.arange(-bs*h_s, (bs+1)*h_s, h_s):
                 dist_row.append([float(pixel1-pixel2), -h_t*(t-s)])
                 cov_row.append(truncated_cov(u = float(pixel1-pixel2), tau = -h_t*(t-s), r = r))
        distances_XX.append(dist_row)
        covs_XX.append(cov_row)
covs_XX = jnp.array(covs_XX)
#dist_XX = jnp.array(dist_XX)
# for row in dist_XX:
#      print(row)

distances_XY = []
for t in reversed(range(p)): # t in {p, p-1, p-2, ..., 1, 0}
    b = np.floor(c*(t+1)*h_t/h_s)
    distances_XY.append(jnp.array([[v, -h_t*(t+1)] for v in jnp.arange(-b*h_s, (b+1)*h_s, h_s)]))
distances_XY = jnp.concat(distances_XY, axis=0)
covs_XY = jnp.array([truncated_cov(u=dist[0], tau=dist[1], r=r) for dist in distances_XY])
#print(distances_XY)
#print(covs_XY)
print("\n")
# for dist in distances_XX:
#      print(dist)
# for cov in covs_XX:
#      print(cov)