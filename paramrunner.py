# -*- coding: utf-8 -*-
# paramrunner.py
# Run findparams on linde model and save results
import helpers
import findparams
import tables
import numpy as np 

savedir = "/home/ith/results/"
savefile = "linde-params.hf5"
WMAP5PIVOT = np.array([5.25e-60])

lindefx = {"vars":["mass","lambda"], "values": [np.linspace(4.9e-8,6e-8), np.linspace(5e-14, 1.6e-13)], 
               "pivotk":WMAP5PIVOT, "pot": "linde",
               "ystart": np.array([25.0,
                                   -1.0,
                                   0.0,
                                   1.0,
                                   0,
                                   1.0,
                                   0])}

def run_linde_model(sf=None):
    """Run linde model and save results."""
    if sf is None:
        sf = savefile
    helpers.ensurepath(savedir)
    fx = lindefx
    
    results = findparams.param_vs_spectrum(fx)
    
    try:
        rfile = tables.openFile(savedir + savefile, "w")
        rfile.createArray(rfile.root, "linde-params", results, "Linde model parameter results")
        print "Results saved in %s" % str(savedir + savefile)
    finally:
        rfile.close()
    
if __name__ == "__main__":
    run_linde_model()