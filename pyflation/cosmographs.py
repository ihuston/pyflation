"""cosmographs.py - Graphing functions for pyflation package

This module provides helper functions for graphing the results of 
pyflation simulations using the Matplotlib package (http://matplotlib.sf.net). 

Especially useful is the multi_format_save function which saves the specified
figure to different formats as requested.

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


import os
import itertools

try:
    import pylab as P
    import matplotlib
except ImportError:
    raise ImportError("Matplotlib is needed to use the plotting helper functions in cosmographs.py.")

# Local import from package
import helpers

#texts
calN = r"$\mathcal{N}_\mathrm{end} - \mathcal{N}$"

#Legend properties
legend_props = {"large": matplotlib.font_manager.FontProperties(size=12),
                "half": matplotlib.font_manager.FontProperties(size=10),
                "small": matplotlib.font_manager.FontProperties(size=9)}

line_props = {"colours": ["red", "green", "blue"],
              "dots": ["-", "--", ":"],
              "coloursanddots": ["r-", "g--", "b:"]}

line_prop_default = line_props["coloursanddots"]

class CosmoGraphError(StandardError):
    """Generic error for graphing facilities."""
    pass
    
def makeklegend(fig, k):
    """Attach a legend to the figure specified outlining the k modes used."""
    if not fig:
        fig = P.gcf()
    P.figure(fig.number)
    l = P.legend([r"$k=" + helpers.eto10(ks) + "$" for ks in k])
    P.draw()
    return l

def reversexaxis(a=None):
    """Reverse the direction of the x axis for the axes object a."""
    if not a:
        a = P.gca()
    a.set_xlim(a.get_xlim()[::-1])
    return
    
def multi_format_save(filenamestub, fig=None, formats=None, overwrite=False, **kwargs):
    """Save figure in multiple formats at once.
    
    Parameters
    ----------

    filenamestub: String
                  Filename with path to which will be added the appropriate 
                  extension for each different file saved.
                  
    fig: Matplotlib figure object, optional
         Figure to save to disk. Defaults to current figure.
         
    formats: list-like, optional
             List of format specifiers to save to,
             default is ["pdf", "eps", "png", "svg"]
             Must be of types supported by current installation.
             
    Other kwargs: These are provided to matplotlib.pylab.savefig.
                  See there for documentation.
    
    Returns
    -------
    filenames: list
               list of saved file names         
    """
    if not formats:
        formats = ["png", "pdf", "eps", "svg"]
    #Check directory exists
    if not os.path.isdir(os.path.dirname(filenamestub)):
        raise IOError("Directory specified does not exist!")
    if not fig:
        fig = P.gcf()
    #Check files don't exist
    savedfiles = []
    for f in formats:
        filename = filenamestub + "." + f
        if os.path.isfile(filename):
            if not overwrite:
                raise IOError("File " + filename + " already exists! File not overwritten.")
        try:
            fig.savefig(filename, format = f, **kwargs)
        except IOError:
            raise
        savedfiles.append(filename)
    return savedfiles
    
def save(fname, dir=None, fig=None):
    if fig is None:
        fig = P.gcf()
    if dir is None:
        dir = os.getcwd()
    multi_format_save(os.path.join(dir, fname), fig, formats=["pdf", "png", "eps"])

def save_with_prompt(fname):
    save = raw_input("Do you want to save the figure, filename:" + fname + "? (y/n) ")
    if save.lower() == "y":
        save(fname)

def set_size_small(fig):
    fig.set_size_inches((4,3))
    fig.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.93)
    
def set_size_large(fig):
    fig.set_size_inches((6,4.5))
    fig.subplots_adjust(left=0.12, bottom=0.10, right=0.90, top=0.90)
    
def set_size_half(fig):
    fig.set_size_inches((6,3))
    fig.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)

def set_size(fig, size="large"):
    if size=="small":
        set_size_small(fig)
    elif size=="large":
        set_size_large(fig)
    elif size=="half":
        set_size_half(fig)
    else:
        raise ValueError("Variable size should be either \"large\", \"half\" or \"small\"!")
    
class LogFormatterTeXExponent(P.LogFormatter, object):
    """Extends pyplot.LogFormatter to use tex notation for tick labels."""
    
    def __init__(self, *args, **kwargs):
        super(LogFormatterTeXExponent, self).__init__(*args, **kwargs)
        
    def __call__(self, *args, **kwargs):
        """Wrap call to parent class with change to tex notation."""
        label = super(LogFormatterTeXExponent, self).__call__(*args, **kwargs)
        label = "$" + label + "$"
        label = helpers.eto10(label)
        return label
    

def calN_figure(ts, ys, fig=None, plot_fn=None, models_legends=None, ylabel=None, 
                       size="large", ls=line_prop_default, halfticks=False):
    """Create a figure using \mathcal{N} on the x-axis.
    
    Parameters
    ----------
    ts: list of arrays
        x-axis values 
    
    ys: list of arrays
        y-axis values, each element should be the same length as corresponding ts
        element. 
    
    fig: Pylab figure instance, optional
         Figure to draw on, default is to create a new figure.
         
    plot_fn: function, optional
             The Pylab plotting function to use, defaults to P.plot.
             
    models_legends: list, optional
                    List of raw strings (including LaTeX) to use in legend
                    of plot. Defaults to not plotting legend.

    ylabel: string, optional
            Raw string (including LaTeX) for y-axis, defaults to empty.
            
    size: string, optional
          One of "small", "large", "half", which specifies the size of the
          figure by using pyflation.cosmographs.set_size. Defaults to "large".
    
    ls: list, optional
        List of linestyle strings to use with each successive line.
        Defaults to ["r-", "g--", "b:"]
    
    halfticks: boolean, optional
               If True only show half the ticks on the y-axis
    Returns
    -------
    fig: the figure instance
    """
    if plot_fn is None:
        plot_fn = P.plot
        
    #Setup figure
    if fig is None:
        fig = P.figure()
    set_size(fig, size)
    lprops = itertools.cycle(ls)
    
    #Plot using specified function
    lines = [plot_fn(t[-1] - t, s, lprops.next()) for t, s in zip(ts, ys)]
    
    #Reverse x axis to count in correct direction
    reversexaxis()
    
    #Get current axis
    ax = fig.gca()
    
    #Add small offset so end of inflation is shown
    ax.set_xlim(right=(t[0]-t[-1])/30.0)
    
    if halfticks:
        #Only show half the ticks on the y axis.
        oldticks = ax.get_yticks()
        ax.set_yticks(oldticks[::2])
        ax.set_yticks(oldticks[1::2], minor=True)
    #Add labels
    P.xlabel(calN)
    if ylabel:
        P.ylabel(ylabel)
        
    #Check if legends are given
    if models_legends:
        ax.legend(models_legends, loc=0, prop=legend_props[size])
    return fig

def generic_figure(xs, ys, fig=None, plot_fn=None, models_legends=None, xlabel=None, ylabel=None, 
                       size="large", ls=line_prop_default, halfticks=False):
    """Create a generic figure with standard x-axis.
    
    Parameters
    ----------
    xs: list of arrays
        x-axis values 
    
    ys: list of arrays
        y-axis values, each element should be the same length as corresponding xs
        element. 
    
    fig: Pylab figure instance, optional
         Figure to draw on, default creates a new figure.
         
    plot_fn: function, optional
             The Pylab plotting function to use, defaults to P.plot.
             
    models_legends: list, optional
                    List of raw strings (including LaTeX) to use in legend
                    of plot. Defaults to not plotting legend.

    xlabel: string, optional
            Raw string (including LaTeX) for x-axis, defaults to empty.

    ylabel: string, optional
            Raw string (including LaTeX) for y-axis, defaults to empty.
            
    size: string, optional
          One of "small", "large", "half", which specifies the size of the
          figure by using pyflation.cosmographs.set_size. Defaults to "large".
    
    ls: list, optional
        List of linestyle strings to use with each successive line.
        Defaults to ["r-", "g--", "b:"]
    
    halfticks: boolean, optional
               If True only show half the ticks on the y-axis
        
    Returns
    -------
    fig: the figure instance
    """
    if plot_fn is None:
        plot_fn = P.plot
        
    #Setup figure
    fig = P.figure()
    set_size(fig, size)
    lprops = itertools.cycle(ls)
    
    #Plot using specified function
    lines = [plot_fn(x, s, lprops.next()) for x, s in zip(xs, ys)]
    
    ax = fig.gca()
    
    if halfticks:
        #Only show half the ticks on the y axis.
        oldticks = ax.get_yticks()
        ax.set_yticks(oldticks[::2])
        ax.set_yticks(oldticks[1::2], minor=True)
    #Add labels
    if xlabel:
        P.xlabel(xlabel)
    if ylabel:
        P.ylabel(ylabel)
        
    #Check if legends are given
    if models_legends:
        ax.legend(models_legends, loc=0, prop=legend_props[size])
    return fig