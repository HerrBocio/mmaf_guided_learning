import time
from scipy.optimize import minimize, basinhopping
from scipy.special import gamma




def variogram(data):
    n_x = len(data[0, :, 0])
    n_y = len(data[0, 0, :])

    x_coords = np.arange(n_x)
    y_coords = np.arange(n_y)

    h_estimation = jnp.array([])
    M_variogram = jnp.array([])

    # DATASET PER CALCOLARE IL VARIOGRAMMA
    dataV = data[0, :, :]
    # print(dataV)

    # for per creare i diversi lag
    for i in range(1, len(y_coords), 1):
        h_estimation = jnp.append(h_estimation,
                                 jnp.linalg.norm(jnp.array([x_coords[i]-x_coords[0], y_coords[0]-y_coords[i]])))

    # for per calcolare il variogramma di Matheron per qui lag
    # Posso usare entrambe le direzioni perche ho isotropia
    # altrimenti usare solo una delle due direzioni
    for i in range(len(h_estimation)):
        val = []
        # print('------------------SW -> NE---------------------')
        # print('h_i=', i)
        # Uso i valori del variogramma in direzione SW -> NE
        for row in range(i+1, len(h_estimation)+1, 1):
            # print('row=', row)
            for col in range(0, len(h_estimation)-i, 1):
                # print('col=', col)
                val = np.append(
                    val, (dataV[row, col]-dataV[row-i-1, col+i+1])**2)
                # print('data 1 =', dataV[row, col])
                # print('data 2 =', dataV[row-i-1, col+i+1])
                # print('val=', val)
        # print('------------------SE -> NW---------------------')
        # Uso i valori del variogramma in direzione SE -> NW
        for row in range(0, len(h_estimation)-i, 1):
            # print('row=', row)
            for col in range(0, len(h_estimation)-i, 1):
                # print('col=', col)
                val = np.append(
                    val, (dataV[row, col]-dataV[row+i+1, col+i+1])**2)
                # print('data 3 =', dataV[row, col])
                # print('data 4 =', dataV[row+i+1, col+i+1])
                # print('val=', val)
        # print('val=', val)
        n = len(val)
        M_variogram = np.append(M_variogram, 1/n*np.sum(val))
        # print('M_variogram=', M_variogram)

    return M_variogram, h_estimation


# MODELLO TEORICO STOU
# theta = [lambda, c, Var]
def modello_teoricoSTOU(h, theta):
    return 2*jnp.pi*theta[2]*theta[1]**2 *(1 - jnp.exp(-theta[0]*h))
# MODELLO TEORICO MSTOU
# theta = [alpha, c, Var, beta, a]
def modello_teoricoMSTOU(h, theta):
    return (jnp.pi*theta[2]*theta[1]**2*gamma(theta[0]-3)*theta[3]**3)/(4*gamma(theta[0]-1))*(1 - (1 + theta[4] * h)**(theta[0]-3))


# ESTIMATION
# y=[Z_ti(xi) - Z_tj(xj)] dove ti(xi) e tj(xj) hanno distanza h
def ols_objective(theta, h, y, modello):
    residuals = y - modello(h, theta)
    return jnp.sum(residuals**2)


def wls1_objective(theta, h, y, modello):
    W = jnp.array([2*i**2 for i in reversed(range(1, len(h)+1, 1))])
    residuals = W*(y - modello(h, theta))
    return jnp.sum(residuals**2)


def wls2_objective(theta, h, y, modello):
    W = jnp.array([2*i**2 for i in reversed(range(1, len(h)+1, 1))]
                 )/(jnp.mean(h)**2)
    residuals = W*(y - modello(h, theta))
    return jnp.sum(residuals**2)


def estimation(model, LSEtype, M_variogram, h_estimation, initial_guess, bounds):

    if model == "STOU":
        def f_model(h, theta):
            return modello_teoricoSTOU(h, theta)
    else:
        def f_model(h, theta):
            return modello_teoricoMSTOU(h, theta)

    if LSEtype == "OLS":
        def objective(theta, h, y, model):
            return ols_objective(theta, h, y, model)
    elif LSEtype == "WLS1":
        def objective(theta, h, y, model):
            return wls1_objective(theta, h, y, model)
    else:
        def objective(theta, h, y, model):
            return wls2_objective(theta, h, y, model)

    # Perform the minimization
    '''
    popt = minimize(ols_objective,  initial_guess,
                  args=(h_estimation, M_variogram), bounds=bounds_STOU)
    '''
    popt = basinhopping(objective, initial_guess, minimizer_kwargs={
        "args": (h_estimation, M_variogram, f_model), "bounds": bounds}, seed=2)

    return popt

