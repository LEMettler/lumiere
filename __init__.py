# __init__.py
#
# Python 3.11.5
# 
# 2023-12-02
# Lu Me

import numpy as np
from scipy import odr
from inspect import signature



def fit(function, x, y, xerr=None, yerr=None, p0=None, **kwargs):
    '''
    ODR and OLS

    ## parameters
    - function
    - x, y: data arrays
    - xerr, yerr: optional absolute x,y ucertainties
    - p0: initial parameter guess (list type)
    - **kwargs: of fitter

    ## returns
    - popt: optimized parameters in array
    - perr: popt 1 sigma uncertainty
    '''

    #inital guess
    if type(p0) == type(None):
        p0 = np.ones(len(signature(function).parameters)-1)

    model = odr.Model(lambda p, x: function(x, *p))
    data = odr.RealData(x, y, sx=xerr, sy=yerr)
    fitter = odr.ODR(data, model, beta0=p0, **kwargs)

    # use ordinary least squares if possible
    if type(xerr) == type(None):
        fitter.set_job(fit_type=2)

    out = fitter.run()
    return out.beta, out.sd_beta




# for mathematical details see e.g.: https://www2.stat.duke.edu/~tjl13/s101/slides/unit6lec3H.pdf
def predictionband(xfit, x0, y0, popt, func):
    '''
    1 sigma probability of this interval containing a randomly drawn datapoint.

    ## parameter
    - xfit: x-positions of fit
    - x0: x-datapoints
    - y0: y-datapoints
    - popt: fit parameter
    - func: function

    ## returns
    - dy: prediction interval array


    '''
    N = len(x0)     
    dof = len(popt)  # degree of freedom
    yhat = func(x0, *popt)

    # singe measurement std
    se = np.sqrt(np.sum((y0 - yhat)**2) / (N - dof))
    # prediction band
    dy = se * np.sqrt(1 + 1/N + ( (xfit - x0.mean())**2 /np.sum((x0 - x0.mean())**2) ))

    return dy


def confidenceband(xfit, x0, y0, popt, func):
    '''
    1 sigma probability of the true function being contained in this interval.
    
    ## parameter
    - xfit: x-positions of fit
    - x0: x-datapoints
    - y0: y-datapoints
    - popt: fit parameter
    - func: function

    ## returns
    - dy: confidence interval array


    '''
    N = len(x0)     
    dof = len(popt)  # degree of freedom
    yhat = func(x0, *popt)

    # singe measurement std
    se = np.sqrt(np.sum((y0 - yhat)**2) / (N - dof))
    # prediction band
    dy = se * np.sqrt(1/N + ( (xfit - x0.mean())**2 /np.sum((x0 - x0.mean())**2) ))

    return dy

