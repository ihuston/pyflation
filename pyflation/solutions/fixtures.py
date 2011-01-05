'''fixtures.py

Module with fixture information and generating functions
Created on 22 Apr 2010

@author: Ian Huston
'''

from pyflation.run_config import getkend
from pyflation import helpers

kmins_default = [1e-61, 3e-61, 1e-60]
deltaks_default = [1e-61, 3e-61, 1e-60]
nthetas_default = [129, 257, 513]
numsoks_default = [257, 513, 1025]
As_default = [2.7e57]
Bs_default = [1e-62]
etas_default = [-2.7559960682873626e+68]

def generate_fixtures(kmins=kmins_default, deltaks=deltaks_default, numsoks=numsoks_default,
                      nthetas=nthetas_default):
    """Generator for fixtures created from cartesian products of input lists."""
    c = helpers.cartesian_product([kmins, deltaks, numsoks, nthetas])
    for now in c:
        fullkmax = getkend(now[0], now[1], now[2])
        fx = {"kmin":now[0], "deltak":now[1], "numsoks":now[2], "fullkmax":fullkmax, 
              "nthetas":now[3]}
        yield fx
        
def fixture_from_model(m, numsoks=None, nthetas=nthetas_default[-1]):
    """Generate a single fixture from a cosmomodels model.
    
    If numsoks is not specified, then use the last value in the defaults.
    """
    if not numsoks:
        numsoks = numsoks_default[-1]
        
    fullkmax = getkend(m.k[0], m.k[1]-m.k[0], numsoks)
    
    fx = {"kmin":m.k[0], "deltak":m.k[1]-m.k[0], "numsoks":numsoks, 
               "fullkmax":fullkmax, "nthetas":nthetas}
    
    return fx