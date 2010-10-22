'''
Created on 22 Oct 2010

@author: ith
'''
import tables
import numpy as np
import sys

def main():
    ystart = np.zeros((4,1025))
    callingparams = {
              'datetime': '20101022173354',
              'dxsav': 0.0,
              'eps': 1e-10,
              'potential_func': 'msqphisq',
              'solver': 'rkdriver_new',
              'tend': 81.640000000009962,
              'tstart': 0.0,
              'tstep_min': 0.00020000000000000001,
              'tstep_wanted': 0.02,
              'ystart': ystart}
    
    hf5dict = {
        "solver" : tables.StringCol(50),
        "ystart" : tables.Float64Col(ystart.shape),
        "tstart" : tables.Float64Col(),
        "simtstart" : tables.Float64Col(),
        "potential_func" : tables.StringCol(255),
        "tend" : tables.Float64Col(),
        "tstep_wanted" : tables.Float64Col(),
        "tstep_min" : tables.Float64Col(),
        "eps" : tables.Float64Col(),
        "dxsav" : tables.Float64Col(),
        "datetime" : tables.Float64Col()
        }
    
    rf = tables.openFile("./test.hf5", "w")
    filters = tables.Filters(complevel=2, complib="blosc")
    resgroup = rf.createGroup(rf.root, "results", "Results of simulation")
    paramstab = rf.createTable(resgroup, "parameters", hf5dict, filters=filters)
    #Save parameters
    paramstabrow = paramstab.row
    params = callingparams
    for key in params:
        paramstabrow[key] = params[key]
    paramstabrow.append() #Add to table
    paramstab.flush()
    rf.flush()
    rf.close()
    return 0



if __name__ == "__main__":
    sys.exit(main())