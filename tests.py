from NN_2D.STOUpozo import STOU
import numpy as np
import jax.numpy as jnp



def test_covs():
    Z = STOU()

    assert 1==1


p = 2
h_t = 1
h_s = 1
c=1
distances_XY = []
for t in reversed(range(p)): # t in {p, p-1, p-2, ..., 1, 0}
    b = np.floor(c*(t+1)*h_t/h_s)
    distances_XY.append(jnp.array([[v, -h_t*(t+1)] for v in jnp.arange(-b*h_s, (b+1)*h_s, h_s)]))
distances_XY = jnp.concat(distances_XY, axis=0)
#print(distances_XY)

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
distances_XX = [] #die nächsten zeilen nur kopiert
covs_XX = []
for t in range(p,0,-1): # t in {p, p-1, p-2, ..., 1}
    bt = int(np.floor(c_*t*h_t/h_s)) # bt:= argmax {a: a*h_s <= (t+1)*c*h_t}
    dist_row = []
    cov_row = []
    for s in range(t,0,-1): # s in {t, t-1, t-2, ..., 1}
        bs = int(np.floor(c_*s*h_t/h_s))
        #dist_block = jnp.zeros((2*bt+1, 2*bs+1))
        cov_block = jnp.zeros((2*bt+1, 2*bs+1))
        dist_block = [[[pixel1-pixel2, t-s] for pixel2 in range(-bs, bs+1)] for pixel1 in range(-bt,bt+1)]
        #for pixel1 in range(-bt,bt+1):
        #    for pixel2 in range(-bs, bs+1):
        #        dist_block.append([pixel1-pixel2, t-s]) #[spatial pos, temporal pos]
        #        cov_block.append(truncated_cov(u = dist_block[-1][0], tau = dist_block[-1][1], r = r))
        dist_row.append(dist_block)
        cov_row.append(cov_block)
    distances_XX.append(dist_row)
    covs_XX.append(cov_row)
#distances_XX = jnp.array(distances_XX)
#covs_XX = jnp.array(covs_XX)
for row in distances_XX:
    for block in row:
        for blockrow in block:
            print(blockrow)
        print("\n")
    