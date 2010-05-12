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
import cPickle
from solutions import fixtures, analyticsolution, calcedsolution
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

def get_fo_models():
    fomsq = c.make_wrapper_model(os.path.join(resdir,"foandsrc-FOCanonicalTwoStage-msqphisq-1.5e-61-6.153e-58-3e-61-195554.hf5"))
    folph = c.make_wrapper_model(os.path.join(resdir,"foandsrc-FOCanonicalTwoStage-lambdaphi4-1.5e-61-6.153e-58-3e-61-171945.hf5"))
    fop23 = c.make_wrapper_model(os.path.join(resdir,"foandsrc-FOCanonicalTwoStage-phi2over3-1.5e-61-6.153e-58-3e-61-104701.hf5"))
    fov0 = c.make_wrapper_model(os.path.join(resdir,"foandsrc-FOCanonicalTwoStage-msqphisq_withV0-1.5e-62-6.153e-58-3e-61-115900.hf5"))
    models = [fomsq, folph, fop23, fov0]
    models_legends = [r"$V(\varphi)=\frac{1}{2}m^2\varphi^2$",
                        r"$V(\varphi)=\frac{1}{4}\lambda\varphi^4$",
                        r"$V(\varphi)=\sigma\varphi^{\frac{2}{3}}$",
                        r"$V(\varphi)=U_0 + \frac{1}{2}m_0^2\varphi^2$"]
    return models, models_legends
    
def get_fo_models_K1():
    fomsq = c.make_wrapper_model(os.path.join(resdir,"foandsrc-FOCanonicalTwoStage-msqphisq-5e-62-2.051e-58-1e-61-195554.hf5"))
    folph = c.make_wrapper_model(os.path.join(resdir,"foandsrc-FOCanonicalTwoStage-lambdaphi4-5e-62-2.051e-58-1e-61-122042.hf5"))
    fop23 = c.make_wrapper_model(os.path.join(resdir,"foandsrc-FOCanonicalTwoStage-phi2over3-5e-62-2.051e-58-1e-61-120949.hf5"))
    fov0 = c.make_wrapper_model(os.path.join(resdir,"foandsrc-FOCanonicalTwoStage-msqphisq_withV0-5e-62-2.051e-58-1e-61-095700.hf5"))
    models = [fomsq, folph, fop23, fov0]
    models_legends = [r"$V(\varphi)=\frac{1}{2}m^2\varphi^2$",
                        r"$V(\varphi)=\frac{1}{4}\lambda\varphi^4$",
                        r"$V(\varphi)=\sigma\varphi^{\frac{2}{3}}$",
                        r"$V(\varphi)=U_0 + \frac{1}{2}m_0^2\varphi^2$"]
    return models, models_legends
    
def get_cmb_models_K1():
    msq = c.make_wrapper_model(os.path.join(resdir,"cmb-msqphisq-5e-62-1.0245e-58-1e-61.hf5"))
    lph = c.make_wrapper_model(os.path.join(resdir, "cmb-lambdaphi4-5e-62-1.0245e-58-1e-61.hf5"))
    phi23 = c.make_wrapper_model(os.path.join(resdir, "cmb-phi2over3-5e-62-1.0245e-58-1e-61.hf5"))
    mv0 = c.make_wrapper_model(os.path.join(resdir, "cmb-msqphisq_withV0-5e-62-1.0245e-58-1e-61.hf5"))
    models = [msq,lph,phi23,mv0]
    models_legends = [r"$V(\varphi)=\frac{1}{2}m^2\varphi^2$",
                        r"$V(\varphi)=\frac{1}{4}\lambda\varphi^4$",
                        r"$V(\varphi)=\sigma\varphi^{\frac{2}{3}}$",
                        r"$V(\varphi)=U_0 + \frac{1}{2}m_0^2\varphi^2$"]
    return models, models_legends

def get_cmb_models_K2():
    msq = c.make_wrapper_model(os.path.join(resdir,"cmb-msqphisq-1.5e-61-3.0735e-58-3e-61.hf5"))
    lph = c.make_wrapper_model(os.path.join(resdir, "cmb-lambdaphi4-1.5e-61-3.0735e-58-3e-61.hf5"))
    phi23 = c.make_wrapper_model(os.path.join(resdir, "cmb-phi2over3-1.5e-61-3.0735e-58-3e-61.hf5"))
    mv0 = c.make_wrapper_model(os.path.join(resdir, "cmb-msqphisq_withV0-1.5e-61-3.0735e-58-3e-61.hf5"))
    models = [msq,lph,phi23,mv0]
    models_legends = [r"$V(\varphi)=\frac{1}{2}m^2\varphi^2$",
                        r"$V(\varphi)=\frac{1}{4}\lambda\varphi^4$",
                        r"$V(\varphi)=\sigma\varphi^{\frac{2}{3}}$",
                        r"$V(\varphi)=U_0 + \frac{1}{2}m_0^2\varphi^2$"]
    return models, models_legends

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

def src_onek(fname="src_onek", size="large", fo=None, kix=17):
    if fo is None:
        fo = fomsq
        kix = 52
    #Create figure and set size
    fig = P.figure()
    set_size(fig, size)
    #Plot graph
    kstart = int(fo.fotstart[kix]/fo.tstep_wanted)
    P.semilogy(fo.tresult[-1] - fo.tresult[kstart:], np.abs(fo.source[kstart:,kix]))
    cg.reversexaxis()
    P.xlabel(calN)
    P.ylabel(r"$|S|$")
    ax = P.gca()
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_xlim((70, -2.0))
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
    P.ylabel(r"$\mathcal{P}^2_{\mathcal{R}} (k_\mathrm{WMAP})$")
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
    P.ylabel(r"$\mathcal{P}^2_{\delta\varphi_1} (k_\mathrm{WMAP})$")
    ax.set_xlim((70, -3.0))
    ax.set_ylim((2e-11, 2e-7))
    if size == "small":
        fig.subplots_adjust(left=0.18)
    P.draw()
    return fig, fname
    
def src_kwmap_3ranges(fname="src-kwmap-3ranges", size="small", m=cmbmsq, fo=fomsq):
    fig = P.figure()
    set_size(fig, size)
    
    #Setup vars
    s1=np.abs(fo.source[::2,52])
    fo2=c.make_wrapper_model("./foandsrc-FOCanonicalTwoStage-msqphisq-1.5e-61-6.153e-58-3e-61-195554.hf5")
    fo3=c.make_wrapper_model("./foandsrc-FOCanonicalTwoStage-msqphisq-2.5e-61-2.0505e-57-1e-60-195554.hf5")
    s2=np.abs(fo2.source[::2,17])
    s3=np.abs(fo3.source[::2,5])

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
    ts = [m.tresult[int(m.fotstart[kix]/m.tstep_wanted):] for m in models]
    srcs = [np.abs(m.source[int(m.fotstart[kix]/m.tstep_wanted):,kix]) for m in models]
    #Setup figure
    fig = P.figure()
    set_size(fig, size)
    lines = [P.semilogy(t - t[0], s) for t,s in zip(ts, srcs)]
    #cg.reversexaxis()
    P.xlabel(r"$\mathcal{N} - \mathcal{N}_\mathrm{~init}$")
    P.ylabel(r"$|S|$")
    ax = fig.gca()
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_xlim((-3, max([t[-1]-t[0] for t in ts]) + 3))
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
    
def cmp_src_kwmap(fname="cmp_src_kwmap", size="large", models=None, models_legends=None, zoom=False):
    if models is None:
        models, models_legends = get_fo_models()
    fig, fname = compare_src(fname, size, models)
    ax = fig.gca()
    if zoom:
        ax.set_xlim((-0.2, 10))
        if size == "large":
            ax.legend(models_legends, loc=0, prop=prop)
    else:
        ax.legend(models_legends, loc=9, prop=prop)
    
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
    P.plot(pr[:,0],pr[:,1])
    ax = P.gca()
    ax.set_xlim((3.48e-10, 4.02e-10))
    ax.set_ylim((2.23e-9, 2.6e-9))
    ax.axhline(2.457e-9, ls="--", color="black")
    P.xlabel(r"$\sigma / M_{\mathrm{PL}}^{10/3}$")
    P.ylabel(r"$\mathcal{P}^2_{\mathcal{R}_1} (k_\mathrm{WMAP})$")
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
    P.plot(pr[:,0],pr[:,1])
    ax = P.gca()
    ax.set_xlim((5.98e-6, 6.52e-6))
    ax.set_ylim((2.2e-9, 2.62e-9))
    ax.axhline(2.457e-9, ls="--", color="black")
    ax.ticklabel_format(style="sci", scilimits=(0,0))
    P.xlabel(r"$m / M_{\mathrm{PL}}$")
    P.ylabel(r"$\mathcal{P}^2_{\mathcal{R}_1} (k_\mathrm{WMAP})$")
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
    P.plot(pr[:,0],pr[:,1])
    ax = P.gca()
    ax.set_xlim((1.49e-13,1.61e-13))
    ax.set_ylim((2.35e-9,2.55e-9))
    ax.axhline(2.457e-9, ls="--", color="black")
    ax.ticklabel_format(style="sci", scilimits=(0,0))
    P.xlabel(r"$\lambda$")
    P.ylabel(r"$\mathcal{P}^2_{\mathcal{R}_1} (k_\mathrm{WMAP})$")
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
    P.plot(pr[:,1],pr[:,2])
    ax = P.gca()
    ax.set_xlim((1.49e-13,1.61e-13))
    ax.set_ylim((2.35e-9,2.55e-9))
    ax.axhline(2.457e-9, ls="--", color="black")
    ax.ticklabel_format(style="sci", scilimits=(0,0))
    P.xlabel(r"$\lambda$")
    P.ylabel(r"$\mathcal{P}^2_{\mathcal{R}_1} (k_\mathrm{WMAP})$")
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
    P.plot(pr[:,1],pr[:,2])
    ax = P.gca()
    ax.set_xlim((1.49e-13,1.61e-13))
    ax.set_ylim((2.35e-9,2.55e-9))
    ax.axhline(2.457e-9, ls="--", color="black")
    ax.ticklabel_format(style="sci", scilimits=(0,0))
    P.xlabel(r"$\lambda$")
    P.ylabel(r"$\mathcal{P}^2_{\mathcal{R}_1} (k_\mathrm{WMAP})$")
    P.draw()
    return fig, fname
    
def msqphisq_withV0_params(fname="msqphisq_withV0_params", size="large"):
    filename = "/home/network/ith/results/param-search/msqphisq_withV0_params.hf5"
    try:
        rf = tables.openFile(filename, "r")
        pr = rf.root.params_results[:]
    finally:
        rf.close()
    fig = P.figure()
    set_size(fig, size)
    P.plot(pr[:,0],pr[:,2])
    ax = P.gca()
    ax.set_xlim((0.9e-6, 6.1e-6))
    ax.set_ylim((0,1e-8))
    ax.axhline(2.457e-9, ls="--", color="black")
    ax.ticklabel_format(style="sci", scilimits=(0,0))
    P.xlabel(r"$m_0 / M_{\mathrm{PL}}$")
    P.ylabel(r"$\mathcal{P}^2_{\mathcal{R}_1} (k_\mathrm{WMAP})$")
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
    P.ylabel(r"$V(\varphi) / M_{\mathrm{PL}}^4$")
    P.xlabel(r"$\varphi / M_{\mathrm{PL}}$")
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
    
def phi2over3_potential(fname="phi2over3_potential", size="large"):
    m = c.make_wrapper_model(os.path.join(resdir, "cmb-phi2over3-5e-62-1.0245e-58-1e-61.hf5"))
    fig, fname = plot_potential_phi(fname, size, m)
    ax = fig.gca()
    ax.set_ylim((-0.1e-9, 2e-9))
    ax.set_xlim((10.5,-0.5))
    P.draw()
    return fig, fname
    
def msqphisq_withV0_potential(fname="msqphisq_withV0_potential", size="large"):
    m = c.make_wrapper_model(os.path.join(resdir, "cmb-msqphisq_withV0-5e-62-1.0245e-58-1e-61.hf5"))
    fig, fname = plot_potential_phi(fname, size, m)
    ax = fig.gca()
    ax.set_ylim((5.7e-10, 1.03e-09))
    ax.set_xlim((18.3, 7.7))
    P.draw()
    return fig, fname
    
def compare_potential_phi(fname="compare_potential_phi", size="large", models=None, models_legends=None, vix=0):
    if models is None:
        msq = cmbmsq
        lph = c.make_wrapper_model(os.path.join(resdir, "cmb-lambdaphi4-5e-62-1.0245e-58-1e-61.hf5"))
        #hyb = c.make_wrapper_model(os.path.join(resdir, "cmb-hybrid2and4-5e-62-1.0245e-58-1e-61.hf5"))
        #lind = c.make_wrapper_model(os.path.join(resdir, "cmb-linde-5e-62-1.0245e-58-1e-61.hf5"))
        phi23 = c.make_wrapper_model(os.path.join(resdir, "cmb-phi2over3-5e-62-1.0245e-58-1e-61.hf5"))
        mv0 = c.make_wrapper_model(os.path.join(resdir, "cmb-msqphisq_withV0-5e-62-1.0245e-58-1e-61.hf5"))
        models = [lph, msq, phi23, mv0]
        if models_legends is None:
            models_legends = [r"$V(\varphi)=\frac{1}{4}\lambda\varphi^4$",
                              r"$V(\varphi)=\frac{1}{2}m^2\varphi^2$",
                              r"$V(\varphi)=\sigma\varphi^{\frac{2}{3}}$",
                              r"$V(\varphi)=U_0 + \frac{1}{2}m_0^2\varphi^2$"]
    #Get background results and potential
    yms = [m.bgmodel.yresult[:] for m in models]
    vms = [np.array([m.potentials(y) for y in ym]) for m, ym in zip(models, yms)]
    fig = P.figure()
    set_size(fig, size)
    #Plot potential versus phi
    lines = [P.plot(ym[:,0], vm[:,vix]) for ym, vm in zip(yms, vms)]
    ax = P.gca()
    cg.reversexaxis()
    ax.set_xlim((26, -1))
    if vix == 0:
        ax.set_ylim((-0.1e-8, 1.6e-8))
    P.legend(models_legends, prop=prop, loc=0)
    vlabels = [r"$V(\varphi) / M_{\mathrm{PL}}^4$", r"$V_{,\varphi} / M_{\mathrm{PL}}^3$", 
               r"$V_{,\varphi \varphi} / M_{\mathrm{PL}}^2$", r"$V_{,\varphi\varphi\varphi} /M_{\mathrm{PL}}$"]
    P.ylabel(vlabels[vix])
    P.xlabel(r"$\varphi / M_{\mathrm{PL}}$")
    P.draw()
    return fig, fname

def compare_potential_n(fname="compare_potential_n", size="large", models=None, models_legends=None, vix=0):
    if models is None:
        msq = cmbmsq
        lph = c.make_wrapper_model(os.path.join(resdir, "cmb-lambdaphi4-5e-62-1.0245e-58-1e-61.hf5"))
        #hyb = c.make_wrapper_model(os.path.join(resdir, "cmb-hybrid2and4-5e-62-1.0245e-58-1e-61.hf5"))
        #lind = c.make_wrapper_model(os.path.join(resdir, "cmb-linde-5e-62-1.0245e-58-1e-61.hf5"))
        phi23 = c.make_wrapper_model(os.path.join(resdir, "cmb-phi2over3-5e-62-1.0245e-58-1e-61.hf5"))
        mv0 = c.make_wrapper_model(os.path.join(resdir, "cmb-msqphisq_withV0-5e-62-1.0245e-58-1e-61.hf5"))
        models = [lph, msq, phi23, mv0]
        if models_legends is None:
            models_legends = [r"$V(\varphi)=\frac{1}{4}\lambda\varphi^4$",
                              r"$V(\varphi)=\frac{1}{2}m^2\varphi^2$",
                              r"$V(\varphi)=\sigma\varphi^{\frac{2}{3}}$",
                              r"$V(\varphi)=U_0 + \frac{1}{2}m_0^2\varphi^2$"]
    #Get background results and potential
    
    yms = [m.bgmodel.yresult[int(m.tresult[0]/(m.tstep_wanted/2.0)):int((m.tend-m.tresult[0])/(m.tstep_wanted/2.0))] for m in models]
    vms = [np.array([m.potentials(y) for y in ym]) for m, ym in zip(models, yms)]
    fig = P.figure()
    set_size(fig, size)
    #Plot potential versus phi
    lines = [P.plot(m.tend-m.bgmodel.tresult[int(m.tresult[0]/(m.tstep_wanted/2.0)):int((m.tend-m.tresult[0])/(m.tstep_wanted/2.0))], 
                vm[:,vix]) for m, vm in zip(models, vms)]
    ax = P.gca()
    cg.reversexaxis()
    ax.set_xlim((72, -3))
    if vix == 0:
        ax.set_ylim((-0.1e-8, 1.3e-8))
    P.legend(models_legends, prop=prop, loc=0)
    vlabels = [r"$V(\varphi) / M_{\mathrm{PL}}^4$", r"$V_{,\varphi} / M_{\mathrm{PL}}^3$", 
               r"$V_{,\varphi \varphi} / M_{\mathrm{PL}}^2$", r"$V_{,\varphi\varphi\varphi} /M_{\mathrm{PL}}$"]
    P.ylabel(vlabels[vix])
    P.xlabel(calN)
    P.draw()
    return fig, fname
    
def src_3ns(fname="src_3ns", size="large"):
    
    fig = P.figure()
    set_size(fig, size)
    ax = fig.gca()
    P.loglog(fomsq.k, np.abs(fomsq.source[2354,:]), color="green")
    P.loglog(fomsq.k, np.abs(fomsq.source[3000,:]), ls="--", color="r")
    
    P.xlabel(r"$k / M_{\mathrm{PL}}$")
    
    P.draw()
    ax.set_ylim((3e-15, 3e-15*26000))
    ax.legend([r"$\mathcal{N}_\mathrm{end} - \mathcal{N} = 58.1$", 
                r"$\mathcal{N}_\mathrm{end} - \mathcal{N} = 51.64$"], prop=prop)
    ax.set_xlim((3e-62, 2e-58))
    P.draw()
    return fig, fname

def cmp_src_allks(fname="cmp_src_allks", size="large", nefolds=5, models=None, models_legends=None):
    if models is None:
        models, models_legends = get_fo_models()
    fig = P.figure()
    set_size(fig, size)
    for m, mleg in zip(models, models_legends):
        kcrossend = m.findkcrossing(m.k[-1], m.tresult, m.yresult[:,2,-1], factor=1)[0]
        tix = int(kcrossend + nefolds/m.tstep_wanted)
        P.loglog(m.k, np.abs(m.source[tix,:]), label=mleg)
    P.xlabel(r"$k / M_{\mathrm{PL}}$")
    P.ylabel(r"$|S|$")
    ax = fig.gca()
    ax.legend(prop=prop, loc=0)
    ax.set_xlim((1e-61, 4e-58))
    P.draw()
    return fig, fname
    
def cmp_Pr_allks(fname="cmp_Pr_allks", size="large", nefolds=5, models=None, models_legends=None):
    if models is None:
        models, models_legends = get_fo_models()
    fig = P.figure()
    set_size(fig, size)
    for m, mleg in zip(models, models_legends):
        kcrossend = m.findkcrossing(m.k[-1], m.tresult, m.yresult[:,2,-1], factor=1)[0]
        tix = int(kcrossend + nefolds/m.tstep_wanted)
        m.runcount = 1
        dp = m.yresult[tix,3,:] + m.yresult[tix,5,:]*1j
        scPr = m.k**3/(2*np.pi**2) * (dp*dp.conj()) / (m.yresult[tix,1,:]**2)
        P.semilogx(m.k, scPr, label=mleg)
    P.xlabel(r"$k / M_{\mathrm{PL}}$")
    P.ylabel(r"$\mathcal{P}^2_{\mathcal{R}_1}$")
    ax = fig.gca()
    if size == "large":
        ax.legend(prop=prop, loc=0)
    ax.set_xlim((1e-61, 4e-58))
    P.draw()
    return fig, fname, models
    
def cmp_dp2_allks(fname="cmp_dp2_allks", size="large", nefolds=5, models=None, models_legends=None):
    if models is None:
        models, models_legends = get_cmb_models_K2()
    fig = P.figure()
    set_size(fig, size)
    for m, mleg in zip(models, models_legends):
        kcrossend = m.findkcrossing(m.k[-1], m.tresult, m.yresult[:,2,-1], factor=1)[0]
        tix = int(kcrossend + nefolds/m.tstep_wanted)
        m.runcount = 1
        dp2 = m.yresult[tix,7,:] + m.yresult[tix,9,:]*1j
        scdp = m.k**1.5/(np.sqrt(2)*np.pi) * np.abs(dp2) / (m.yresult[tix,1,:]**2)
        P.loglog(m.k, scdp, label=mleg)
    P.xlabel(r"$k$")
    P.ylabel(r"$\mathcal{P}^2_\mathcal{R}$")
    ax = fig.gca()
    if size == "large":
        ax.legend(prop=prop, loc=0)
    ax.set_xlim((1e-61, 4e-58))
    P.draw()
    return fig, fname
 
def errors_3ranges(fname="errors_3ranges", size="half", termix=0):
    try:
        err_file=open(os.path.join(resdir, "errors-3Ks-starpc34.dat"), "r")
        results = cPickle.load(err_file)
    except IOError:
        raise
    finally:
        err_file.close()
    #
    fig = P.figure()
    set_size(fig, size)
    K_legends = [r"$k\in K_1$", r"$k\in K_2$", r"$k\in K_3$"]
    ylims = [(3e-11,3e-6), (7e-10,1e-3), (1e-8,1e-7), (3e-8,1e-5)]
    for err, K_legend in zip(results, K_legends):
        P.loglog(err.k, err.postconv["rel_err"][termix], label=K_legend)
    ax = fig.gca()
    ax.set_xlim((3e-62, 3e-57))
    ax.set_ylim(ylims[termix])
    ax.legend(prop=prop, loc=3)
    P.xlabel(r"$k / M_{\mathrm{PL}}$")
    P.ylabel(r"$\epsilon_\mathrm{rel}$")
    P.draw()
    return fig, fname
        
def errors_analytic(fname="errors_aterm", size="large", termix=0, Kix=0):
    try:
        err_file=open(os.path.join(resdir, "errors-3Ks-starpc34.dat"), "r")
        results = cPickle.load(err_file)
    except IOError:
        raise
    finally:
        err_file.close()
    #
    fig = P.figure()
    set_size(fig, size)
    K_legends = [r"$k\in K_1$", r"$k\in K_2$", r"$k\in K_3$"]
    term_legends = [r"$I_{\mathcal{A}}(k) / M_{\mathrm{PL}}^2$", r"$|I_{\mathcal{B}}(k)|$", 
                    r"$|I_{\widetilde{\mathcal{C}}}(k)|$", r"$|I_{\widetilde{\mathcal{D}}}(k)|$"]
    err = results[Kix]
    P.semilogx(err.k, np.abs(err.postconv["analytic"][termix]), label=K_legends[Kix])
    ax = fig.gca()
    P.xlabel(r"$k / M_{\mathrm{PL}}$")
    P.ylabel(term_legends[termix])
    P.draw()
    return fig, fname

def errors_general(fname="errors_general", size="large", fixture_ixs=None, fx_labels=None):
    if not fixture_ixs:
        fixture_ins = [1]
        fx_labels=[r""]
    try:
        err_file=open(os.path.join(resdir, "errors-results-virgo.dat"), "r")
        results = np.array(cPickle.load(err_file))
    except IOError:
        raise
    finally:
        err_file.close()
    #
    fig = P.figure()
    set_size(fig, size)
    errors = results[fixture_ixs]
    for eix, err in enumerate(errors):
        P.loglog(err.k, np.abs(err.postconv["rel_err"]), label=fx_labels[eix])
    ax = fig.gca()
    P.xlabel(r"$k / M_{\mathrm{PL}}$")
    P.ylabel(r"$\epsilon_\mathrm{rel}$")
    P.draw()
    return fig, fname
    
def err_nthetas(fname="err_nthetas", size="large"):
    fxixs = [62, 71, 80]
    fxls = ['$\\Delta k= 1\\times 10^{-61}M_\\mathrm{PL}$',
            '$\\Delta k= 3\\times 10^{-61}M_\\mathrm{PL}$',
            '$\\Delta k= 1\\times 10^{-60}M_\\mathrm{PL}$']
            
    fig, fname = errors_general(fname, size, fxixs, fxls)
    ax = fig.gca()
    ax.set_xlim((7e-61, 2e-57))
    ax.legend(prop=prop, loc=0)
    P.draw()
    return fig, fname

def err_deltak_kmin(fname="err_nthetas", size="large"):
    fxixs = [8, 17, 26]
    fxls = ['$\\Delta k= 1\\times 10^{-61}M_\\mathrm{PL}$',
            '$\\Delta k= 3\\times 10^{-61}M_\\mathrm{PL}$',
            '$\\Delta k= 1\\times 10^{-60}M_\\mathrm{PL}$']
            
    fig, fname = errors_general(fname, size, fxixs, fxls)
    ax = fig.gca()
    ax.set_xlim((7e-62, 2e-57))
    ax.legend(prop=prop, loc=0)
    P.draw()
    return fig, fname

def errors_nthetas(fname="errors_general", size="large"):
    try:
        err_file=open(os.path.join(resdir, "errors-nthetas-starpc34.dat"), "r")
        results = np.array(cPickle.load(err_file))
    except IOError:
        raise
    finally:
        err_file.close()
    #
    fig = P.figure()
    set_size(fig, size)
    labels = [r"$N_\theta = " + str(t) + "$" for t in [129,257,513]]
    for eix, err in enumerate(results):
        P.loglog(err.k, np.abs(err.postconv["rel_err"][0]), label=labels[eix])
    ax = fig.gca()
    ax.set_xlim((1e-61, 2e-57))
    P.xlabel(r"$k / M_{\mathrm{PL}}$")
    P.ylabel(r"$\epsilon_\mathrm{rel}$")
    ax.legend(prop=prop, loc=0)
    P.draw()
    return fig, fname

def epsilon_slowroll(fname="epsilon_slowroll", size="large", models=None, kix=None, zoom=False):
    if models is None:
        models, legends = get_cmb_models_K2()
        kix = 17
    fig = P.figure()
    set_size(fig, size)
    lines = []
    #Find starting time indexes of kix mode
    for m,l in zip(models, legends):
        ts = int(helpers.find_nearest_ix(m.bgmodel.tresult, m.fotstart[kix]))
        te = int(helpers.find_nearest_ix(m.bgmodel.tresult, m.tend))
        #Get epsilon
        epsilon = m.bgepsilon[ts:te]
        #Print epsilon line
        lines.append(P.plot(m.bgmodel.tresult[ts:te] - m.bgmodel.tresult[ts], epsilon, label=l))
    cg.reversexaxis()
    P.xlabel(r"$\mathcal{N}-\mathcal{N}_\mathrm{~init}$")
    P.ylabel(r"$\epsilon_H$")
    ax = fig.gca()
    
    ax.legend(prop=prop, loc=0)
    if zoom:
        ax.set_xlim((-0.2, 10))
        ax.set_ylim((-0.01, 0.045))
    else:
        ax.set_xlim((-3, 67))
        ax.set_ylim((-0.05, 1.05))
    
    P.draw()
    return fig, fname
    
def eta_slowroll(fname="eta_slowroll", size="large", models=None, legends=None, kix=None, zoom=False):
    if models is None:
        models, legends = get_cmb_models_K2()
        kix = 17
    fig = P.figure()
    set_size(fig, size)
    lines = []
    #get slow roll parameters
    for m, l in zip(models, legends):
        ts = int(helpers.find_nearest_ix(m.bgmodel.tresult, m.fotstart[kix]))
        te = int(helpers.find_nearest_ix(m.bgmodel.tresult, m.tend))
                #get slow roll parameters
        ym = m.bgmodel.yresult[ts:te]
        vm = np.array(map(m.potentials, ym))
        eta = vm[:,2]/(3*m.bgmodel.yresult[ts:te,2]**2)
        lines.append(P.plot(m.bgmodel.tresult[ts:te] - m.bgmodel.tresult[ts], eta, label=l))
    cg.reversexaxis()
    P.xlabel(r"$\mathcal{N}-\mathcal{N}_\mathrm{~init}$")
    P.ylabel(r"$\eta_H$")
    ax = fig.gca()
    ax.legend(prop=prop, loc=0)
    if zoom:
        ax.set_xlim((-0.2, 10))
        ax.set_ylim((-0.01, 0.055))
    else:
        ax.set_xlim((-3, 67))
#        ax.set_ylim((-0.05, 1.05))
    
    P.draw()    
    return fig, fname    

def analytic_v_calced_prehorizon(fname="analytic_v_calced_prehorizon", size="large"):
    fig = P.figure()
    set_size(fig, size)
    try:
        cfile = open("/home/network/ith/results/analysis/newmass/csrc-beforehorizon.dat", "r")
        afile = open("/home/network/ith/results/analysis/newmass/asrc-beforehorizon.dat", "r")
        tfile = open("/home/network/ith/results/analysis/newmass/tres-beforehorizon.dat", "r")
        csrc = np.load(cfile)
        asrc = np.load(afile)
        tres = np.load(tfile)
    finally:
        cfile.close()
        afile.close()
        tfile.close()
    
    P.semilogy(tres[:250], np.abs(csrc[:250]), label="Calculated Solution")
    P.semilogy(tres[:250], np.abs(asrc[:250]), label="Analytic Solution")
    cg.reversexaxis()
    P.xlabel(r"$\mathcal{N}_\mathrm{end} - \mathcal{N}$")
    P.ylabel(r"$|S(k_\mathrm{WMAP})|$")
    ax = P.gca()
    ax.legend(prop=prop, loc=0)
    
    P.draw()
    return fig, fname

def analytic_v_calced_prehorizon_errors(fname="analytic_v_calced_prehorizon", size="large"):
    fig = P.figure()
    set_size(fig, size)
    try:
        cfile = open("/home/network/ith/results/analysis/newmass/csrc-beforehorizon.dat", "r")
        afile = open("/home/network/ith/results/analysis/newmass/asrc-beforehorizon.dat", "r")
        tfile = open("/home/network/ith/results/analysis/newmass/tres-beforehorizon.dat", "r")
        csrc = np.load(cfile)
        asrc = np.load(afile)
        tres = np.load(tfile)
    finally:
        cfile.close()
        afile.close()
        tfile.close()
    
    err = np.abs(csrc[:250]-asrc[:250])/np.abs(asrc[:250])
    
    P.semilogy(tres[:250], err)
    
    cg.reversexaxis()
    P.xlabel(r"$\mathcal{N}_\mathrm{end} - \mathcal{N}$")
    P.ylabel(r"$\epsilon_\mathrm{rel}$")
    ax = P.gca()
    ax.legend(prop=prop, loc=0)
    
    P.draw()
    return fig, fname
    
def analytic_v_calced_onetstep(fname="analytic_v_calced_onetstep", size="large", nix=1000):
    fig = P.figure()
    set_size(fig, size)
    m = c.make_wrapper_model("/home/network/ith/results/analysis/newmass/cmb-msqphisq-1.5e-61-3.0735e-58-3e-61.hf5")
    fx = fixtures.fixture_from_model(m)
    asol = analyticsolution.NoPhaseWithEtaSolution(fx)
    csol = calcedsolution.NoPhaseWithEtaCalced(fx)
    
    asol.k = np.float128(asol.k)
    
    asrc = asol.full_source_from_model(m, nix)
    csrc = csol.full_source_from_model(m, nix)
    
    P.semilogx(csol.k, np.abs(csrc), label="Calculated Solution")
    P.semilogx(asol.k, np.abs(asrc), label="Analytic Solution")
    
    P.xlabel(r"$k$")
    P.ylabel(r"$|S(k_\mathrm{WMAP})|$")
    ax = P.gca()
    ax.legend(prop=prop, loc=0)
    
    P.draw()
    
    return fig, fname

def analytic_v_calced_onetstep_error(fname="analytic_v_calced_onetstep", size="large", nix=1000):
    fig = P.figure()
    set_size(fig, size)
    m = c.make_wrapper_model("/home/network/ith/results/analysis/newmass/cmb-msqphisq-1.5e-61-3.0735e-58-3e-61.hf5")
    fx = fixtures.fixture_from_model(m)
    asol = analyticsolution.NoPhaseWithEtaSolution(fx)
    csol = calcedsolution.NoPhaseWithEtaCalced(fx)
    
    asol.k = np.float128(asol.k)
    
    asrc = asol.full_source_from_model(m, nix)
    csrc = csol.full_source_from_model(m, nix)
    
    err = np.abs(asrc-csrc)/np.abs(asrc)
    
    P.loglog(csol.k, err)
    
    P.xlabel(r"$k$")
    P.ylabel(r"$\epsilon_\mathrm{rel}$")
    ax = P.gca()
    P.draw()
    return fig, fname