'''
Created on 22 Oct 2010

@author: ith
'''
import tables
import numpy as np
import sys

def main():
    ystart = np.zeros((4,1000))
    callingparams = {
              'ystart': ystart}
    
    hf5dict = {
        "ystart" : tables.Float64Col(ystart.shape)
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