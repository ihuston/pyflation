# -*- coding: utf-8 -*-
#Ipython log file to create graphics
from __future__ import division
import cosmographs as cg
import cosmomodels as c
import pylab as P
import numpy as np
import matplotlib
import helpers
import sohelpers
import tables
import os.path

#


resdir = "/home/network/ith/results/analysis/newmass/"
graphdir = os.path.join(resdir, "calN-graphs/")
#files
cmbmsq=c.make_wrapper_model(os.path.join(resdir,"cmb-msqphisq-5e-62-1.0245e-58-1e-61.hf5"))
fomsq=c.make_wrapper_model(os.path.join(resdir,"foandsrc-FOCanonicalTwoStage-msqphisq-5e-62-2.051e-58-1e-61-195554.hf5"))

#texts
calN=r"$\mathcal{N}_\mathrm{end} - \mathcal{N}$"

#Legend properties
prop = matplotlib.font_manager.FontProperties(size=12)

def save(fname, fig=None):
    if fig is None:
        fig = P.gcf()
    cg.multi_format_save(os.path.join(graphdir, fname), fig, formats=["pdf", "png", "eps"])

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

def set_size(fig, size="large"):
    if size=="small":
        set_size_small(fig)
    elif size=="large":
        set_size_large(fig)
    else:
        raise ValueError("Variable size should be either \"large\" or \"small\"!")
    
def dp1_kwmap(fname="dp1_kwmap", size="large", m=cmbmsq):
    #k coefficient
    kc = m.k**(1.5)/(np.sqrt(2)*np.pi)
    fig = P.figure()
    set_size(fig, size)
    P.plot(m.tresult[-1]- m.tresult, kc[52]*m.dp1[:,52].real)
    P.plot(m.tresult[-1]- m.tresult, kc[52]*m.dp1[:,52].imag, ls="--", color="g")
    ax = P.gca()
    ax.axvline(m.tresult[-1]- 21.24, ls="-.", color="red")
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

def dp2_kwmap(fname="dp2_kwmap", size="small", m=cmbmsq):
    #k coefficient
    kc = m.k**(1.5)/(np.sqrt(2)*np.pi)
    fig = P.figure()
    set_size(fig, size)
    P.plot(m.tresult[-1]- m.tresult, kc[52]*m.dp2[:,52].real)
    P.plot(m.tresult[-1]- m.tresult, kc[52]*m.dp2[:,52].imag, ls="--", color="g")
    ax = P.gca()
    ax.axvline(m.tresult[-1]- 21.24, ls="-.", color="red")
    cg.reversexaxis()
    #P.draw()
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    ax.set_xlim((64.7, 60.1))
    ax.set_ylim((-4.1e-95,4.1e-95))
    P.xlabel(calN)
    P.ylabel(r"$\frac{1}{\sqrt{2}\pi} k^{\frac{3}{2}} \delta\varphi_2$")
    P.draw()
    return fig, fname
    
def src_kwmap(fname="src-kwmap", size="small", m=cmbmsq, fo=fomsq):
    #Get timeresult for correct range
    tr52 = m.tresult[-1] - m.tresult[865:]
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
    
def bgepsilon(fname="bgepsilon", size="large", m=cmbmsq):
    bgeps = m.bgepsilon
    fig = P.figure()
    set_size(fig, size)
    
    P.plot(m.bgmodel.tresult, bgeps)
    ax = P.gca()
    ax.axvline(m.tresult[-1], ls="--", color="red")
    P.xlabel(r"$\mathcal{N}$")
    P.ylabel(r"$\varepsilon_H$")
    ax.set_xlim((78.8, 83.2))
    ax.set_ylim((-0.1, 3.1))
    P.draw()
    return fig, fname
    
def pr_kwmap(fname="Pr-kwmap", size="small", horiz=True, m=cmbmsq):
    Pr = m.Pr
    scPr = m.k[52]**3/(2*np.pi**2) * Pr[:,52]
    fig = P.figure()
    set_size(fig, size)
    P.semilogy(m.tresult[-1]-m.tresult[865:], np.abs(scPr[865:]))
    ax = P.gca()
    if horiz:
        ax.axvline(m.tresult[-1] - 21.24, ls="--", color="red")
    cg.reversexaxis()
    
    P.xlabel(calN)
    P.ylabel(r"$\mathcal{P}_{\mathcal{R}} (k_\mathrm{WMAP})$")
    P.draw()
    return fig, fname
    
def pphi_kwmap(fname="Pphi-kwmap", size="small", horiz=True, m=cmbmsq):
    Pphi = m.Pphi
    scPphi = m.k[52]**3/(2*np.pi**2) * Pphi[:,52]
    fig = P.figure()
    set_size(fig, size)
    P.semilogy(m.tresult[-1]-m.tresult[865:], np.abs(scPphi[865:]))
    ax = P.gca()
    if horiz:
        ax.axvline(m.tresult[-1] - 21.24, ls="--", color="red")
    cg.reversexaxis()
    #Labels
    P.xlabel(calN)
    P.ylabel(r"$\mathcal{P}_{\delta\varphi} (k_\mathrm{WMAP})$")
    ax.set_xlim((70, -3.0))
    ax.set_ylim((2e-11, 2e-7))
    P.draw()
    return fig, fname
    
def src_kwmap_3ranges(fname="src-kwmap-3ranges", size="small", m=cmbmsq, fo=fomsq):
    fig = P.figure()
    set_size(fig, size)
    
    #Setup vars
    s1=abs(fo.source[::2,52])
    fo2=c.make_wrapper_model("./foandsrc-FOCanonicalTwoStage-msqphisq-1.5e-61-6.153e-58-3e-61-195554.hf5")
    fo3=c.make_wrapper_model("./foandsrc-FOCanonicalTwoStage-msqphisq-2.5e-61-2.0505e-57-1e-60-195554.hf5")
    s2=abs(fo2.source[::2,17])
    s3=abs(fo3.source[::2,5])

    tr = m.tresult[-1] - m.tresult[865:]
    P.semilogy(tr, s3[865:])
    P.semilogy(tr, s2[865:])
    P.semilogy(tr, s1[865:])
    P.legend([r"$k\in K_" + str(ix) + "$" for ix in [3,2,1]], prop=prop, loc=0)
    cg.reversexaxis()
    ax = P.gca()
    ax.set_yticks(np.array([1e-10, 1e-5, 1e-0,1e5]))
    
    ax.set_yticks(np.array([1e-15,1e-10, 1e-5, 1e-0,1e5]))
    
    P.xlabel(calN)
    P.ylabel(r"$|S|$")
    ax.set_xlim((70, -3.0))
    P.draw()
    return fig, fname

def src_3ks(fname="src-3ks", size="small", m=cmbmsq, fo=fomsq):
    #Initialise variables
    tr0 = m.tresult[-1] - m.tresult[631:]
    tr52 = m.tresult[-1] - m.tresult[865:]
    tr1024 = m.tresult[-1] - m.tresult[1015:]
    s0 = np.abs(fo.source[::2,0])
    s52 = np.abs(fo.source[::2,52])
    s1024 = np.abs(fo.source[::2,1024])
    #Setup legend
    l = helpers.klegend(m.k[:][[0,52,1024]])
    #Setup figure and specify size
    fig = P.figure()
    set_size(fig, size)
    #Plot lines
    P.semilogy(tr0, s0[631:])
    P.semilogy(tr52, s52[865:])
    P.semilogy(tr1024, s1024[1015:])
    #Change attributes
    ax = P.gca()
    ax.lines[0].set_color("r")
    ax.lines[1].set_color("g")
    ax.lines[2].set_color("blue")
    
    cg.reversexaxis()
    P.legend(l,prop=prop,loc=0)
    
    P.xlabel(calN)
    P.ylabel(r"$|S|$")
    ax.set_xlim((73, -3.0))
    ax.set_yticks(np.array([1e-15, 1e-10, 1e-5, 1e-0, 1e5,1e10]))
    #Draw figure again
    P.draw()
    return fig, fname

def src_vs_t_kwmap(fname="src-vs-t-kwmap", size="small", m=cmbmsq, fo=fomsq):
    #Get tresult
    tr52 = m.tresult[-1] - m.tresult[865:]
    #Get t term
    r = sohelpers.find_soderiv_terms(m, kix=np.array([52]))
    #find t term for kwmap
    t52 = np.abs(r[:,0,0]+r[:,1,0]+(r[:,2,0]+r[:,3,0])*1j)
    #Get source for kwmap
    s52 = np.abs(fo.source[::2,52])
    #Setup figure
    fig = P.figure()
    set_size(fig, size)
    P.semilogy(tr52, s52[865:])
    P.semilogy(tr52, t52[865:])
    cg.reversexaxis()
    #Change attributes 
    P.xlabel(calN)
    ax = P.gca()
    ax.set_xlim((70, -3.0))
    ax.set_yticks(np.array([1e-15, 1e-10, 1e-5, 1e-0]))
    P.legend([r"$|S|$", r"$|T|$"], loc=0, prop=prop)
    P.ylabel("")
    P.draw()
    return fig, fname

def s_over_t_3ks(fname="s-over-t-3ks", size="small", m=cmbmsq, fo=fomsq):
    #Initialise variables
    tr0 = m.tresult[-1] - m.tresult[631:]
    tr52 = m.tresult[-1] - m.tresult[865:]
    tr1024 = m.tresult[-1] - m.tresult[1015:]
    r = sohelpers.find_soderiv_terms(m, kix=np.array([0,52,1024]))
    t0 = np.abs(r[:,0,0]+r[:,1,0]+(r[:,2,0]+r[:,3,0])*1j)
    t1024 = np.abs(r[:,0,2]+r[:,1,2]+(r[:,2,2]+r[:,3,2])*1j)
    t52 = np.abs(r[:,0,1]+r[:,1,1]+(r[:,2,1]+r[:,3,1])*1j)
    s0 = np.abs(fo.source[::2,0])
    s52 = np.abs(fo.source[::2,52])
    s1024 = np.abs(fo.source[::2,1024])

    
    #Get differences
    d0 = s0/t0
    d52 =s52/t52
    d1024= s1024/t1024
    d0=d0[631:]
    d52=d52[865:]
    d1024=d1024[1015:]


    fig = P.figure()
    set_size(fig, size)
    #Draw lines and legend
    P.semilogy(tr1024, d1024)
    P.semilogy(tr52, d52)
    P.semilogy(tr0, d0)
    l = helpers.klegend(m.k[:][[0,52,1024]])
    cg.reversexaxis()
    
    #Set attributes
    ax = P.gca()
    ax.axhline(y=1, ls="--", color="grey")
    ax.set_xlim((73,-3))
    ax.set_ylim((1e-13, 40))
    ax.set_yticks(np.array([1e-10, 1e-5, 1e-0]))
    P.xlabel(calN)
    P.ylabel(r"$|S|/|T|$")
    P.legend(l[::-1], prop=prop, loc=0)
    
    #Draw again
    P.draw()
    return fig, fname

def src_mvl(fname="src-mvl", size="small", zoom=False):
    #Setup variables
    msq2 = c.make_wrapper_model("./foandsrc-FOCanonicalTwoStage-msqphisq-1.5e-61-6.153e-58-3e-61-195554.hf5")
    mt2 = msq2.tresult[1730:]
    mt2 = mt2[-1] - mt2
    ms2 = np.abs(msq2.source[1730:, 17])
    lph2 = c.make_wrapper_model("./foandsrc-FOCanonicalTwoStage-lambdaphi4-1.5e-61-6.153e-58-3e-61-113146.hf5")
    ls2 = np.abs(lph2.source[1291:,17])
    ls2 = ls2[:6435]
    lt2 = lph2.tresult[1291:]
    lt2 = lt2[:6435]
    lt2 = lt2[-1] -lt2 

    fig = P.figure()
    set_size(fig, size)
    P.semilogy(mt2,ms2)
    P.semilogy(lt2,ls2)
    cg.reversexaxis()
    P.xlabel(calN)
    P.ylabel(r"$|S|$")
    ax = P.gca()
    ax.lines[1].set_linestyle("--")
    P.legend([r"$U=\frac{1}{2}m^2 \varphi^2$", r"$U=\frac{1}{4} \lambda \varphi^4$"], loc=0, prop=prop)
    if zoom:
        ax.set_ylim((2.6e-11, 845))
        ax.set_xlim((65, 57))
    else:
        ax.set_xlim((70,-3))
    ax.set_yticks(ax.get_yticks()[::2])
    P.draw()
    return fig, fname
    
def compare_src(fname="compare_src", size="large", models=None, kix=52):
    if models is None:
        msq = c.make_wrapper_model("./foandsrc-FOCanonicalTwoStage-msqphisq-1.5e-61-6.153e-58-3e-61-195554.hf5")
        lph = c.make_wrapper_model("./foandsrc-FOCanonicalTwoStage-lambdaphi4-1.5e-61-6.153e-58-3e-61-113146.hf5")
        models = [msq, lph]
    ts = [m.tresult for m in models]
    srcs = [np.abs(m.source[:,kix]) for m in models]
    #Setup figure
    fig = P.figure()
    set_size(fig, size)
    lines = [P.semilogy(t[-1] - t, s) for t,s in zip(ts, srcs)]
    cg.reversexaxis()
    P.xlabel(calN)
    P.ylabel(r"$|S|$")
    ax = fig.gca()
    ax.set_yticks(ax.get_yticks()[::2])
    P.draw()
    return fig, fname
    
def comp_src_msq_phi23(fname="comp_src_msq_phi23", size="large"):
    msq =c.make_wrapper_model(os.path.join(resdir, "foandsrc-FOCanonicalTwoStage-msqphisq-5e-62-2.051e-58-1e-61-195554.hf5"))
    phi23 = c.make_wrapper_model(os.path.join(resdir, "foandsrc-FOCanonicalTwoStage-phi2over3-5e-62-2.051e-58-1e-61-120949.hf5"))
    models = [msq, phi23]
    fig, fname = compare_src(fname, size, models)
    ax = fig.gca()
    ax.lines[1].set_linestyle("--")
    ax.legend([r"$V(\varphi) = \frac{1}{2}m^2\varphi^2$", r"$V(\varphi) = \lambda\varphi^{\frac{2}{3}}$"], prop=prop, loc=0)
    ax.set_xlim((71,-1))
    P.draw()
    return fig, fname
    
def phi2over3_params(fname="phi2over3_params", size="large"):
    filename = "/home/network/ith/results/param-search/phi2over3-3.5e-10-4e-10.hf5"
    try:
        rf = tables.openFile(filename, "r")
        pr = rf.root.params_results[:]
    finally:
        rf.close()
    fig = P.figure()
    set_size(fig, size)
    P.plot(pr[:,0],pr[:,1], color="black")
    ax = P.gca()
    ax.set_xlim((3.48e-10, 4.02e-10))
    ax.set_ylim((2.23e-9, 2.6e-9))
    ax.axhline(2.457e-9, ls="--", color="black")
    P.xlabel(r"$\lambda$")
    P.ylabel(r"$\mathcal{P}_\mathcal{R} (k_\mathrm{WMAP})$")
    P.draw()
    return fig, fname
   
def msqphisq_params(fname="msqphisq_params", size="large"):
    filename = "/home/network/ith/results/param-search/msqphisq-6e-6-6.5e-6.hf5"
    try:
        rf = tables.openFile(filename, "r")
        pr = rf.root.params_results[:]
    finally:
        rf.close()
    fig = P.figure()
    set_size(fig, size)
    P.plot(pr[:,0],pr[:,1], color="black")
    ax = P.gca()
    ax.set_xlim((5.98e-6, 6.52e-6))
    ax.set_ylim((2.2e-9, 2.62e-9))
    ax.axhline(2.457e-9, ls="--", color="black")
    ax.ticklabel_format(style="sci", scilimits=(0,0))
    P.xlabel(r"$m$")
    P.ylabel(r"$\mathcal{P}_\mathcal{R} (k_\mathrm{WMAP})$")
    P.draw()
    return fig, fname
    
def lambdaphi4_params(fname="lambdaphi4_params", size="large"):
    filename = "/home/network/ith/results/param-search/lambdaphi4-1.5e-13-1.6e-13"
    try:
        rf = tables.openFile(filename, "r")
        pr = rf.root.params_results[:]
    finally:
        rf.close()
    fig = P.figure()
    set_size(fig, size)
    P.plot(pr[:,0],pr[:,1], color="black")
    ax = P.gca()
    ax.set_xlim((1.49e-13,1.61e-13))
    ax.set_ylim((2.35e-9,2.55e-9))
    ax.axhline(2.457e-9, ls="--", color="black")
    ax.ticklabel_format(style="sci", scilimits=(0,0))
    P.xlabel(r"$\lambda$")
    P.ylabel(r"$\mathcal{P}_\mathcal{R} (k_\mathrm{WMAP})$")
    P.draw()
    return fig, fname
    
def hybrid2and4_params(fname="hybrid2and4_params", size="large"):
    filename = "/home/network/ith/results/param-search/hybrid2and4-params.hf5"
    try:
        rf = tables.openFile(filename, "r")
        pr = rf.root.params_results[:]
    finally:
        rf.close()
    fig = P.figure()
    set_size(fig, size)
    P.plot(pr[:,1],pr[:,2], color="black")
    ax = P.gca()
    ax.set_xlim((1.49e-13,1.61e-13))
    ax.set_ylim((2.35e-9,2.55e-9))
    ax.axhline(2.457e-9, ls="--", color="black")
    ax.ticklabel_format(style="sci", scilimits=(0,0))
    P.xlabel(r"$\lambda$")
    P.ylabel(r"$\mathcal{P}_\mathcal{R} (k_\mathrm{WMAP})$")
    P.draw()
    return fig, fname
    
def linde_params(fname="linde_params", size="large"):
    filename = "/home/network/ith/results/param-search/linde-params.hf5"
    try:
        rf = tables.openFile(filename, "r")
        pr = rf.root.params_results[:]
    finally:
        rf.close()
    fig = P.figure()
    set_size(fig, size)
    P.plot(pr[:,1],pr[:,2], color="black")
    ax = P.gca()
    ax.set_xlim((1.49e-13,1.61e-13))
    ax.set_ylim((2.35e-9,2.55e-9))
    ax.axhline(2.457e-9, ls="--", color="black")
    ax.ticklabel_format(style="sci", scilimits=(0,0))
    P.xlabel(r"$\lambda$")
    P.ylabel(r"$\mathcal{P}_\mathcal{R} (k_\mathrm{WMAP})$")
    P.draw()
    return fig, fname
     
def plot_potential_phi(fname="plot_potential", size="large", m=cmbmsq):
    #Get background results and potential
    ym = m.bgmodel.yresult
    vm = np.array([m.potentials(y) for y in ym])
    fig = P.figure()
    set_size(fig, size)
    #Plot potential versus phi
    P.plot(ym[:,0], vm[:,0])
    ax=P.gca()
    P.ylabel(r"$V(\varphi)$")
    P.xlabel(r"$\varphi$")
    P.draw()
    return fig, fname
    
def msqphisq_potential(fname="msqphisq_potential", size="large"):
    m = cmbmsq
    fig, fname = plot_potential_phi(fname, size, m)
    ax = fig.gca()
    ax.set_ylim((-0.1e-9,7e-9))
    ax.set_xlim((19,-1))
    P.draw()
    return fig, fname
    
def lambdaphi4_potential(fname="lambdaphi4_potential", size="large"):
    lph = c.make_wrapper_model(os.path.join(resdir, "cmb-lambdaphi4-5e-62-1.0245e-58-1e-61.hf5"))
    fig, fname = plot_potential_phi(fname, size, lph)
    ax = fig.gca()
    ax.set_ylim((-0.5e-9, 16e-9))
    ax.set_xlim((26,-1))
    P.draw()
    return fig, fname

def hybrid2and4_potential(fname="hybrid2and4_potential", size="large"):
    m = c.make_wrapper_model(os.path.join(resdir, "cmb-hybrid2and4-5e-62-1.0245e-58-1e-61.hf5"))
    fig, fname = plot_potential_phi(fname, size, m)
    ax = fig.gca()
    ax.set_ylim((-0.5e-9, 16e-9))
    ax.set_xlim((26,-1))
    P.draw()
    return fig, fname
    
def linde_potential(fname="linde_potential", size="large"):
    m = c.make_wrapper_model(os.path.join(resdir, "cmb-linde-5e-62-1.0245e-58-1e-61.hf5"))
    fig, fname = plot_potential_phi(fname, size, m)
    ax = fig.gca()
    ax.set_ylim((-0.5e-9, 16e-9))
    ax.set_xlim((26,-1))
    P.draw()
    return fig, fname
    
def phi2over3_potential(fname="hybrid2and4_potential", size="large"):
    m = c.make_wrapper_model(os.path.join(resdir, "cmb-phi2over3-5e-62-1.0245e-58-1e-61.hf5"))
    fig, fname = plot_potential_phi(fname, size, m)
    ax = fig.gca()
    ax.set_ylim((-0.1e-9, 2e-9))
    ax.set_xlim((10.5,-0.5))
    P.draw()
    return fig, fname