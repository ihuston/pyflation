'''
Created on 6 Jul 2010

@author: ith
'''

def runsomodel(fofile, filename=None, soargs=None):
    """Execute a SOCanonicalThreeStage model and save results.
    
    A new instance of SOCanonicalThreeStage is created, from the specified first order file.
    The model is run and the results are then saved into a file with the specified filename.
    
    Parameters
    ----------
    fofile : String
             Filename of first order file to use in simulation. First order file must contain
             source term and have correct data structure.
    
    filename : String, optional
               Name of file to save results to. File will be created in the directory
               specified by `RESULTSDIR` module variable.
               
    soargs : dict, optional
             Dictonary of arguments to be sent to second order class method. 
    
    Returns
    -------
    filename: String
              Name of the file where results have been saved.
              
    Raises
    ------
    Exception
       Any exception raised during saving of code.
    """
    try:
        fomodel = c.make_wrapper_model(fofile)
    except:
        harness_logger.exception("Error wrapping model file.")
        raise
    if soargs is None:
        soargs = {}
    somodel = c.SOCanonicalThreeStage(fomodel, **soargs)
    try:
        harness_logger.debug("Starting model run...")
        somodel.run(saveresults=False)
        harness_logger.debug("Model run finished.")
    except c.ModelError:
        harness_logger.exception("Something went wrong with model, quitting!")
        sys.exit(1)
    if filename is None:
        kinit, kend, deltak = somodel.k[0], somodel.k[-1], somodel.k[1]-somodel.k[0]
        filename = run_config.RESULTSDIR + "so-" + somodel.potential_func + "-" + str(kinit) + "-" + str(kend) + "-" + str(deltak) + ".hf5"
    try:
        harness_logger.debug("Trying to save model data to %s...", filename)
        ensurepath(filename)
        somodel.saveallresults(filename=filename)
    except Exception:
        harness_logger.exception("IO error, nothing saved!")
    #Destroy model instance to save memory
#     harness_logger.debug("Destroying model instance...")
#     del somodel
    #Success!
    harness_logger.info("Successfully ran and saved simulation in file %s.", filename)
    return filename


if __name__ == '__main__':
    pass