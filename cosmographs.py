"""cosmographs.py

Author: Ian Huston

Graphing functions for cosmomodels
"""
import pylab as P
import os
import helpers

class CosmoGraphError(StandardError):
    """Generic error for graphing facilities."""
    pass
    

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
    
def plotresults(m, fig=None, show=True, varindex=None, klist=None, saveplot=False, numks=5):
    """Plot results of simulation run on a graph.
        Return figure instance used."""
    if m.runcount == 0:
        raise CosmoGraphError("Model has not been run yet, cannot plot results!")
    
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
        P.legend([r"$k=" + helpers.eto10(ks) + "$" for ks in m.k[klist]])
    #P.title(m.plottitle, figure=fig)
    
    #Should we show it now or just return it without showing?
    if show:
        P.show()
    #Should we save the plot somewhere?
    if saveplot:
        m.saveplot(fig)   
    #Return the figure instance
    return fig
    