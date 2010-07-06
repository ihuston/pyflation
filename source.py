'''
Created on 6 Jul 2010

@author: ith
'''



def runfullsourceintegration(modelfile, ninit=0, nfinal=-1, sourcefile=None, numsoks=1025, ntheta=513):
    """Run source integrand calculation."""
    try:
        m = c.make_wrapper_model(modelfile)
    except:
        harness_logger.exception("Error wrapping model file.")
        raise
    if sourcefile is None:
        sourcefile = run_config.RESULTSDIR + "src-" + m.potential_func + "-" + str(min(m.k)) + "-" + str(max(m.k))
        sourcefile += "-" + str(m.k[1]-m.k[0]) + "-" + time.strftime("%H%M%S") + ".hf5"
    #get source integrand and save to file
    try:
        ensurepath(sourcefile)
        filesaved = sosource.getsourceandintegrate(m, sourcefile, ninit=ninit, nfinal=nfinal, ntheta=ntheta, numks=numsoks)
        harness_logger.info("Source term saved as " + filesaved)
    except Exception:
        harness_logger.exception("Error getting source term.")
        raise
    #Destroy model instance to save memory
    harness_logger.debug("Destroying model instance to reclaim memory...")
    try:
        del m
    except IOError:
        harness_logger.exception("Error closing model file!")
        raise
    return filesaved


def runparallelintegration(modelfile, ninit=0, nfinal=None, sourcefile=None, ntheta=513, numsoks=1025, soargs=None):
    """Run parallel source integrand and second order calculation."""
    try:
        from mpi4py import MPI
    except ImportError:
        raise

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nprocs = comm.Get_size() - 1 #Don't include host
    status = 0
    try:
        m = c.make_wrapper_model(modelfile)
    except:
        harness_logger.exception("Error wrapping model file.")
        if myrank != 0:
            comm.send([{'rank':myrank, 'status':10}], dest=0, tag=10) #Send error msg
        raise
    if sourcefile is None:
        if "fo-" in modelfile:
            srcstub = os.path.splitext(os.path.basename(modelfile))[0].replace("fo", "src")
        else:
            srcstub = "-".join(["src", m.potential_func, str(min(m.k)), str(max(m.k)), str(m.k[1]-m.k[0]), time.strftime("%Y%m%d")])
        sourcefile = run_config.RESULTSDIR + srcstub + "/src-part-" + str(myrank) + ".hf5"
    if myrank == 0:
        #Check sourcefile directory exists:
        ensurepath(os.path.dirname(sourcefile))
        harness_logger.info("Source file path is %s." %sourcefile)
    if myrank != 0:
        #Do not include host node in execution
        if nfinal == -1:
            nfinal = m.tresult.shape[0]
        nfostart = min(m.fotstartindex).astype(int)
        nstar = max(nfostart, ninit)
        totalnrange = len(m.tresult[nstar:nfinal])
        nrange = N.ceil(totalnrange/nprocs)
        if myrank == 1:
            myninit = ninit
        else:
            myninit = nstar + (myrank-1)*nrange
        mynend = nstar + myrank*nrange
        harness_logger.info("Process rank: %d, ninit: %d, nend: %d", myrank, myninit, mynend)
        
        #get source integrand and save to file
        try:
            filesaved = sosource.getsourceandintegrate(m, sourcefile, ninit=myninit, nfinal=mynend,
                                                       ntheta=ntheta, numks=numsoks)
            harness_logger.info("Source term saved as " + filesaved)
        except Exception:
            harness_logger.exception("Error getting source term.")
            comm.send([{'rank':myrank, 'status':10}], dest=0, tag=10) #Tag=10 signals an error
            raise
        #Destroy model instance to save memory
        harness_logger.debug("Destroying model instance to reclaim memory...")
        try:
            del m
        except IOError:
            harness_logger.exception("Error closing model file!")
            comm.send([{'rank':myrank, 'status':10}], dest=0, tag=10) #Tag=10 signals an error
            raise
        comm.send([{'rank':myrank, 'status':0}], dest=0, tag=0) #Tag=0 signals success
        return filesaved
    else:
        #Get rid of model object
        del m
        process_list = range(1, nprocs+1)
        status_list = []
        while len(status_list) < len(process_list):
            status = MPI.Status()
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            status_list.append(data[0])
            if status.tag > 0:
                harness_logger.error("Error in subprocess %d!", status.source)
                raise IOError("Error in subprocess %d!" % status.source)
        harness_logger.info("All processes finished! Starting merger...")
        srcdir = os.path.dirname(sourcefile)
        newsrcfile = os.path.dirname(srcdir) + os.sep + srcstub + ".hf5" 
        srcmergefile = srcmerge.mergefiles(newfile=newsrcfile, dirname=srcdir)
        harness_logger.info("Merger complete. File saved as %s.", srcmergefile)
        #Start combination of first order and source files
        harness_logger.info("Starting to combine first order and source files.")
        foandsrcfile = sohelpers.combine_source_and_fofile(srcmergefile, modelfile)
        harness_logger.info("Combination complete, saved in %s.", foandsrcfile)
        harness_logger.info("Starting second order run...")
        sofile = runsomodel(foandsrcfile, soargs=soargs)
        harness_logger.info("Second order run complete. Starting to combine first and second order results.")
        cfilename = sofile.replace("so", "cmb")
        cfile = sohelpers.combine_results(foandsrcfile, sofile, cfilename)
        harness_logger.info("Combined results saved in %s.", cfile)
        return cfile

if __name__ == '__main__':
    pass