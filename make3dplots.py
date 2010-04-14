'''
make3dplots.py
Created on 14 Apr 2010

Plot 3d graphs of cosmomodels

@author: Ian Huston
'''

import cosmomodels as c
import numpy as np
import makeplots
import helpers
from mpl_toolkits.mplot3d import Axes3D
import pylab as P
import cosmographs as cg

def complex_dp1(fname="complex_dp1", size="large", kix=None, models=None):
    """
    Plot complex values of dp1 in a 3d plot for all models.
    """
    if models is None:
        models, legends = makeplots.get_cmb_models_K2()
        kix = 17
    fig = P.figure()
    makeplots.set_size(fig, size)
    lines = []
    
    ax = Axes3D(fig)
    for m, l in zip(models, legends):
        ts = helpers.find_nearest_ix(m.tresult, m.fotstart[kix])
        te = helpers.find_nearest_ix(m.tresult, m.fotstart[kix]+10)
        lines.append(ax.plot(m.tresult[ts:te]-m.tresult[ts], m.yresult[ts:te,3,kix], m.yresult[ts:te,5,kix], label=l))
    ax.set_xlabel(r"$\mathcal{N}$")
    ax.set_ylabel(r"$\Re \delta\varphi_1$")
    ax.set_zlabel(r"$\Im \delta\varphi_1$")
    P.draw()
    return fig, fname
                     
def complex_src(fname="complex_src", size="large", kix=None, models=None):
    """
    Plot complex values of source term in a 3d plot for all models.
    """
    if models is None:
        models, legends = makeplots.get_fo_models()
        kix = 17
    fig = P.figure()
    makeplots.set_size(fig, size)
    lines = []
    
    ax = Axes3D(fig)
    for m, l in zip(models, legends):
        ts = helpers.find_nearest_ix(m.tresult, m.fotstart[kix])
        te = helpers.find_nearest_ix(m.tresult, m.fotstart[kix]+10)
        lines.append(ax.plot(m.tresult[ts:te]-m.tresult[ts], m.source[ts:te,kix].real, m.source[ts:te,kix].imag, label=l))
    ax.set_xlabel(r"$\mathcal{N}$")
    ax.set_ylabel(r"$\Re S$")
    ax.set_zlabel(r"$\Im S$")
    P.draw()
    return fig, fname

 
