# -*- coding: utf-8 -*-
#Ipython log file to create graphics
from __future__ import division
import cosmographs as cg
import cosmomodels as c
import pylab as P
import numpy as np

#
graphdir = "./calN-graphs/"

#files
cmb=c.make_wrapper_model("./cmb-msqphisq-5e-62-1.0245e-58-1e-61.hf5")
fo=c.make_wrapper_model("./foandsrc-FOCanonicalTwoStage-msqphisq-5e-62-2.051e-58-1e-61-195554.hf5")
#k coefficient
kc = cmb.k**(1.5)/(np.sqrt(2)*np.pi)
#texts
calN=r"$\mathcal{N}_\mathrm{end} - \mathcal{N}$"

def save(fname, fig=None):
    if fig is None:
        fig = P.gcf()
    cg.multi_format_save(graphdir + fname, fig, formats=["pdf", "png", "eps"])

def save_with_prompt(fname):
    save = raw_input("Do you want to save the figure, filename:" + fname + "? (y/n) ")
    if save.lower() == "y":
        save(fname)

def set_size_small(fig):
    fig.set_size_inches((4,3))
    fig.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.95)
    
def set_size_large(fig):
    fig.set_size_inches((6,4.5))
    fig.subplots_adjust(left=0.12, bottom=0.10, right=0.90, top=0.90)

def set_size(fig, size="large"):
    if size=="small":
        set_size_small(fig)
    elif size=="large":
        set_size_large(fig)
    else:
        raise ValueError("Variable size should be either \"large\" or \"small\"!")
    
def dp1_kwmap(fname="dp1_kwmap", size="large"):
    fig = P.figure()
    set_size(fig, size)
    P.plot(cmb.tresult[-1]- cmb.tresult, kc[52]*cmb.dp1[:,52].real)
    P.plot(cmb.tresult[-1]- cmb.tresult, kc[52]*cmb.dp1[:,52].imag, ls="--", color="g")
    ax = P.gca()
    ax.axvline(cmb.tresult[-1]- 21.24, ls="-.", color="red")
    cg.reversexaxis()
    #P.draw()
    P.xlabel(calN)
    P.ylabel(r"$\frac{1}{\sqrt{2}\pi} k^{\frac{3}{2}} \delta\varphi_1$")
    #P.draw()
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    #set limits
    ax.set_xlim((64.7, 60.1))
    ax.set_ylim((-0.00033,0.00033))
    P.draw()
    
    return fig, fname

def dp2_kwmap(fname="dp2_kwmap", size="small"):
    fig = P.figure()
    set_size(fig, size)
    P.plot(cmb.tresult[-1]- cmb.tresult, kc[52]*cmb.dp2[:,52].real)
    P.plot(cmb.tresult[-1]- cmb.tresult, kc[52]*cmb.dp2[:,52].imag, ls="--", color="g")
    ax = P.gca()
    ax.axvline(cmb.tresult[-1]- 21.24, ls="-.", color="red")
    cg.reversexaxis()
    #P.draw()
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    ax.set_xlim((64.7, 60.1))
    ax.set_ylim((-4.1e-95,4.1e-95))
    P.xlabel(calN)
    P.ylabel(r"$\frac{1}{\sqrt{2}\pi} k^{\frac{3}{2}} \delta\varphi_2$")
    P.draw()
    return fig, fname
    
def src_kwmap(fname="src-kwmap", size="small"):
    #Get timeresult for correct range
    tr52 = cmb.tresult[-1] - cmb.tresult[865:]
    #Get source values at correct timesteps (every second one) for kwmap
    s52 = np.abs(fo.source[::2,52])
    #Create figure and set size
    fig = P.figure()
    set_size(fig, size)
    #Plot graph
    P.semilogy(tr52, s52[865:])
    cg.reversexaxis()
    P.xlabel(calN)
    P.ylabel(r"$|S|$")
    ax = P.gca()
    ax.set_yticks(np.array([1e-15, 1e-10, 1e-5, 1e-0]))
    ax.set_xlim((70.0, -3.0))
    P.draw()
    return fig, fname
    
def bgepsilon(fname="bgepsilon", size="large"):
    bgeps = cmb.bgepsilon
    fig = P.figure()
    set_size(fig, size)
    
    P.plot(cmb.bgmodel.tresult, bgeps)
    ax = P.gca()
    ax.axvline(cmb.tresult[-1], ls="--", color="red")
    P.xlabel(r"$\mathcal{N}$")
    P.ylabel(r"$\varepsilon_H$")
    ax.set_xlim((78.8, 83.2))
    ax.set_ylim((-0.1, 3.1))
    P.draw()
    return fig, fname