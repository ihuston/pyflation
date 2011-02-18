"""cosmographs.py - Graphing functions for pyflation package

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

This module provides helper functions for graphing the results of 
pyflation simulations using the Matplotlib package (http://matplotlib.sf.net). 

Especially useful is the multi_format_save function which saves the specified
figure to different formats as requested.
"""
import os

try:
    import pylab as P
except ImportError:
    raise ImportError("Matplotlib is needed to use the plotting helper functions in cosmographs.py.")

# Local import from package
import helpers

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
    fig.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.93)
    
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
    