# __init__.py
#
# Python 3.11.5
# 
# Creation: 2023-12-02
# Lu Me

import numpy as np
from scipy import odr
from inspect import signature
import matplotlib.pyplot as plt
from cycler import cycler
import uncertainties.unumpy as unp

def fit(function, x, y, xerr=None, yerr=None, p0=None, return_out=False, **kwargs):
    '''
    ODR and OLS

    ## parameters
    - function
    - x, y: data arrays
    - xerr, yerr: optional absolute x,y ucertainties
    - p0: initial parameter guess (list type)
    - return_out: returns Output object
    - **kwargs: of fitter

    ## returns
    - popt: optimized parameters in array
    - perr: popt 1 sigma uncertainty
    ---
    Output object
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
    

    if return_out:
        return out.beta, out.sd_beta, out
    else:    
        return out.beta, out.sd_beta

def fitinfo(variables, out, function_name=None,rel_x=0.02, rel_y=0.98, ax=None):
    uvals = unp.uarray(out.beta, out.sd_beta)
    if type(variables) is str:
        if ',' in variables:
            variables = variables.split(', ')
        else:
            variables = variables.split(' ')


    title = f'$\chi^2 /\,$dof$\,=\,{out.res_var:.3f}$'

    if function_name is not None:
        title = function_name + f'\n' + title

    for nval, uval in zip(variables, uvals):
        title += f'\n${nval}\,=\,{uval:L}$'
    
    if ax is None:
        plt.text(rel_x, rel_y, title, size=12, ha='left', va='top', transform=plt.gca().transAxes)
    else:
        plt.text(rel_x, rel_y, title, size=12, ha='left', va='top', transform=ax.transAxes)

    return title




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


#----------------------------------------------------------
def labels(xlabel='x', ylabel='y', grid=True,  ax=None):
    if ax is None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(grid)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(grid)


def loadstyle(kit_colors=False):
    if kit_colors:
        cvals = ['#008e7b', #green
                '#eb4a00', #orange
                '#128edb', #light blue
                '#88c906', #light green
                '#d10434', #red-purple
                '#4664aa', #blue
                '#7f7f7f', #grey
                '#fce500', #yellow
                '#a7822e', #brown
                '#a3107c', #purple    red:#a22223
                        ]
        print('Using KIT colors.')
    else:
        cvals = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        print('Using standard colors.')

    custom_rcParams = {
        'axes.labelsize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.grid': True,
        'figure.figsize': (12,6),
        'axes.prop_cycle': cycler('color', cvals),
        }
    plt.rcParams.update(custom_rcParams)
    print('Custom style set.')
