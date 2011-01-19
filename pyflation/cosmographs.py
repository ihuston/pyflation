"""cosmographs.py

Author: Ian Huston

Graphing functions for cosmomodels
"""
import pylab as P
import os

import helpers
import configuration

#Name of the directory with results directories inside
resdir = configuration.RESULTSDIR
#Name of directory to store graphs 
graphdir = os.path.abspath(os.path.join(resdir, "graphs"))
helpers.ensurepath(graphdir)

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
    
def multi_format_save(filenamestub, fig=None, formats=None, **kwargs):
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
            raise IOError("File " + filename + " already exists! File not overwritten.")
        try:
            fig.savefig(filename, format = f, **kwargs)
        except IOError:
            raise
        savedfiles.append(filename)
    return savedfiles
    
def save(fname, fig=None):
    if fig is None:
        fig = P.gcf()
    multi_format_save(os.path.join(graphdir, fname), fig, formats=["pdf", "png", "eps"])

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
    
def plotresults(m, fig=None, show=True, varindex=None, klist=None, saveplot=False, numks=5):
    """Plot results of simulation run on a graph.
        Return figure instance used."""
    if varindex is None:
        varindex = 0 #Set default list of variables to plot
    
    if fig is None:
        fig = P.figure() #Create figure
    else:
        P.figure(fig.number)
    #One plot command for with ks, one for without
    
    if m.k is None:
        P.plot(m.tresult, m.yresult[:,varindex])
    else:
        if klist is None:
            klist = slice(0, len(m.k), len(m.k)/numks)
        P.plot(m.tresult, m.yresult[:,varindex,klist])
    #Create legends and axis names
    P.xlabel(m.tname)
    P.ylabel(m.ynames[varindex])
    if klist is not None:
        leg = makeklegend(fig, m.k[klist])
    #P.title(m.plottitle, figure=fig)
    
    #Should we show it now or just return it without showing?
    if show:
        P.show()
    #Should we save the plot somewhere?
    if saveplot:
        m.saveplot(fig)   
    #Return the figure instance
    return fig

def plot_errors(results, labels=None, fig=None):
    if labels is None:
        labels = ["" for r in results]
    if fig is None:
        fig = P.figure()
    for e, l  in zip(results, labels):
        P.plot(e.k, e.postconv["rel_err"], label=l)
    