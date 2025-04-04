# __init__.py
#
# Python 3.11.5
# 
# Creation: 2023-12-02
# Lukas Mettler

import numpy as np
from scipy import odr
from inspect import signature
import matplotlib.pyplot as plt
from cycler import cycler
import uncertainties.unumpy as unp 

def fit(function, x, y, xerr=None, yerr=None, p0=None, return_out=False, **kwargs):
    '''
    Perform ODR (Orthogonal Distance Regression) and OLS (Ordinary Least Squares) fitting.

    Parameters:
    function (callable): The function to be fitted.
    x (array-like): The independent variable data.
    y (array-like): The dependent variable data.
    xerr (array-like, optional): The absolute uncertainties in the x-values.
    yerr (array-like, optional): The absolute uncertainties in the y-values.
    p0 (list, optional): The initial parameter guess for the fit.
    return_out (bool, optional): If True, the Output object from the fitting routine is returned.
    **kwargs: Additional keyword arguments passed to the fitter.

    Returns:
    popt (array): The optimized fit parameters.
    perr (array): The 1-sigma uncertainties of the fit parameters.
    out (scipy.odr.Output, optional): The Output object from the fitting routine (if return_out is True).
    '''

    # Set initial guess if not provided
    if type(p0) == type(None):
        p0 = np.ones(len(signature(function).parameters)-1)

    # Create ODR model and data
    model = odr.Model(lambda p, x: function(x, *p))
    data = odr.RealData(x, y, sx=xerr, sy=yerr)
    fitter = odr.ODR(data, model, beta0=p0, **kwargs)

    # Use OLS if no x-uncertainties
    if type(xerr) == type(None):
        fitter.set_job(fit_type=2)

    # Run the fit
    out = fitter.run()

    # Return the fit parameters and uncertainties, and the Output object if requested
    if return_out:
        return out.beta, out.sd_beta, out
    else:
        return out.beta, out.sd_beta

def fitinfo(variables, out, function_name=None, rel_x=0.02, rel_y=0.98, ax=None):
    '''
    Generate a formatted string with fit information to be displayed on a plot.

    Parameters:
    variables (str or list): The names of the fit parameters.
    out (scipy.odr.Output): The Output object from the fitting routine.
    function_name (str, optional): The name of the function being fitted.
    rel_x (float, optional): The relative x-position of the text on the plot (0 to 1).
    rel_y (float, optional): The relative y-position of the text on the plot (0 to 1).
    ax (matplotlib.axes.Axes, optional): The axes object to plot the text on.

    Returns:
    title (str): The formatted string with the fit information.
    '''
    uvals = unp.uarray(out.beta, out.sd_beta)

    # Handle single or multiple variable names
    if type(variables) is str:
        if ',' in variables:
            variables = variables.split(', ')
        else:
            variables = variables.split(' ')

    # Construct the title string
    title = f'$\chi^2 /\,$dof$\,=\,{out.res_var:.3f}$'
    if function_name is not None:
        title = function_name + f'\n' + title
    for nval, uval in zip(variables, uvals):
        title += f'\n${nval}\,=\,{uval:L}$'

    # Display the title on the plot
    if ax is None:
        plt.text(rel_x, rel_y, title, size=12, ha='left', va='top', transform=plt.gca().transAxes)
    else:
        plt.text(rel_x, rel_y, title, size=12, ha='left', va='top', transform=ax.transAxes)

    return title

def predictionband(xfit, x0, y0, popt, func):
    '''
    Calculate the 1-sigma prediction band for the fitted function.

    Parameters:
    xfit (array-like): The x-values at which to calculate the prediction band.
    x0 (array-like): The original x-data used for the fit.
    y0 (array-like): The original y-data used for the fit.
    popt (array-like): The optimized fit parameters.
    func (callable): The function being fitted.

    Returns:
    dy (array-like): The 1-sigma prediction band.
    '''
    N = len(x0)
    dof = len(popt)  # Degree of freedom
    yhat = func(x0, *popt)

    # Calculate the single measurement standard error
    se = np.sqrt(np.sum((y0 - yhat)**2) / (N - dof))

    # Calculate the prediction band
    dy = se * np.sqrt(1 + 1/N + ((xfit - x0.mean())**2 / np.sum((x0 - x0.mean())**2)))

    return dy

def confidenceband(xfit, x0, y0, popt, func):
    '''
    Calculate the 1-sigma confidence band for the fitted function.

    Parameters:
    xfit (array-like): The x-values at which to calculate the confidence band.
    x0 (array-like): The original x-data used for the fit.
    y0 (array-like): The original y-data used for the fit.
    popt (array-like): The optimized fit parameters.
    func (callable): The function being fitted.

    Returns:
    dy (array-like): The 1-sigma confidence band.
    '''
    N = len(x0)
    dof = len(popt)  # Degree of freedom
    yhat = func(x0, *popt)

    # Calculate the single measurement standard error
    se = np.sqrt(np.sum((y0 - yhat)**2) / (N - dof))

    # Calculate the confidence band
    dy = se * np.sqrt(1/N + ((xfit - x0.mean())**2 / np.sum((x0 - x0.mean())**2)))

    return dy

def labels(xlabel='x', ylabel='y', grid=True, ax=None):
    '''
    Set the x-axis label, y-axis label, and grid on a plot.

    Parameters:
    xlabel (str, optional): The x-axis label.
    ylabel (str, optional): The y-axis label.
    grid (bool, optional): Whether to display the grid.
    ax (matplotlib.axes.Axes, optional): The axes object to set the labels and grid on.
    '''
    if ax is None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(grid)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(grid)

def loadstyle(kit_colors=False):
    '''
    Load a custom style for matplotlib plots.

    Parameters:
    kit_colors (bool, optional): If True, use the KIT color palette, otherwise use the standard matplotlib color palette.
    '''
    if kit_colors:
        cvals = ['#008e7b', '#eb4a00', '#128edb', '#88c906', '#d10434', '#4664aa', '#7f7f7f', '#fce500', '#a7822e', '#a3107c']
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
        'figure.figsize': (12, 6),
        'axes.prop_cycle': cycler('color', cvals),
    }
    plt.rcParams.update(custom_rcParams)
    print('Custom style set.')


def errorPrintFormatting(x, xstat, xsys):
    ''' 
    Print the statistical and systematic error, formatted correctly to the significant decimal places of the errors.

    Parameters:
    x, xstat, xsys (floats): Value and errors.

    Returns: 
    Formatted String.
    '''

    def round_to_two_significant(num):
        if num == 0:
            return 0
        # most significant digit in 10^(-magnitude) position 
        magnitude = int(np.floor(np.log10(abs(num))))
        # scale to 2 decimal place
        scale = 10**(magnitude - 1)
        return round(num / scale) * scale
    
    def format_uncertainty(value):
        if value == int(value):
            return str(int(value))
        
        # number of decimals needed
        num_decimals = max(0, -int(np.floor(np.log10(abs(value)))) + 1)
        formatted_value = f"{value:.{num_decimals}f}"
        # remove trailing zeros and decimal/s
        return formatted_value.rstrip('0').rstrip('.') if '.' in formatted_value else formatted_value


    xsys_rounded = round_to_two_significant(xsys)
    xstat_rounded = round_to_two_significant(xstat)

    # relevant digits
    sys_decimals = (-int(np.floor(np.log10(abs(xsys_rounded)))) + 1) if xsys_rounded != 0 else 0
    stat_decimals = (-int(np.floor(np.log10(abs(xstat_rounded)))) + 1) if xstat_rounded != 0 else 0
    decimals = max(sys_decimals, stat_decimals, 0)

    x_str = f"{x:.{decimals}f}"
    xsys_str = format_uncertainty(xsys_rounded)
    xstat_str = format_uncertainty(xstat_rounded)

    return f"{x_str} \\pm {xstat_str}(stat) \\pm {xsys_str}(sys)"


def calculateErrors(uarr, text=''):
    '''
    Calculate the statistical and combined systematic error *separately* of a measurement series or the results of one.
    
    Parameters:
    uarr (uncertainty array): the individual values with their (propagated) systematic errors.
    text (string, optional): additional text for printing the result.
    
    Returns:
    nominal value (float)
    statistical error (float)
    systematic error (float) 
    '''

    x_nom = np.mean(uarr).n
    x_stat = np.std(unp.nominal_values(uarr))
    x_sys = np.mean(uarr).s 

    output_string = errorPrintFormatting(x_nom, x_stat, x_sys)
    print(f'{text} = {output_string}')
    return (x_nom, x_stat, x_sys)
