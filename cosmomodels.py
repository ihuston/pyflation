"""Cosmological Model simulations by Ian Huston
    $Id: cosmomodels.py,v 1.79 2008/08/06 17:18:03 ith Exp $
    
    Provides generic class CosmologicalModel that can be used as a base for explicit models."""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
import pylab as P
from matplotlib import axes3d
import rk4
import sys
import os.path
import datetime
import pickle
from scipy.integrate import odeint as scipy_odeint
from scipy import interpolate
import helpers 

#debugging
from IPython.Debugger import Pdb


class ModelError(StandardError):
    """Generic error for model simulating. Attributes include current results stack."""
    pass

class CosmologicalModel:
    """Generic class for cosmological model simulations.
    Has no derivs function to pass to ode solver, but does have
    plotting function and initial conditions check.
    
    Results can be saved in a pickled file as a list of tuples of the following
    structure:
       resultset = (lastparams, tresult, yresult)
       
       lastparams is formatted as in the function callingparams(self) below
    """
    solverlist = ["odeint", "rkdriver_dumb", "scipy_odeint"]
    
    def __init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, eps=1.0e-6, dxsav=0.0, solver="scipy_odeint"):
        """Initialize model variables, some with default values. Default solver is odeint."""
        
        self.ystart = ystart
        self.k = None #so we can test whether k is set
        
        if N.all(tstart < tend): 
            self.tstart, self.tend = tstart, tend
        elif N.all(tstart==tend):
            raise ValueError, "End time is the same as start time!"
        else:
            raise ValueError, "Ending time is before starting time!"
        
        if tstep_wanted >= tstep_min:
            self.tstep_wanted, self.tstep_min = tstep_wanted, tstep_min
        else:
            raise ValueError, "Desired time step is smaller than specified minimum!"
        
        if eps < 1:
            self.eps = eps
        else:
            raise ValueError, "Not enough accuracy! Change eps < 1."
        
        if dxsav >=0.0:
            self.dxsav = dxsav
        else:
            raise ValueError, "Data saving step must be 0 or positive!"
        
        if solver in self.solverlist:
            self.solver = solver
        else:
            raise ValueError, "Solver not recognized!"
        
        self.tresult = None #Will hold last time result
        self.yresult = None #Will hold array of last y results
        self.runcount = 0 #How many times has the model been run?
        self.resultlist = [] #List of all completed results.
        
        
        self.plottitle = "A generic Cosmological Model"
        self.tname = "Time"
        self.ynames = ["First dependent variable"]
        
    def derivs(self, yarray, t):
        """Return an array of derivatives of the dependent variables yarray at timestep t"""
        pass
    
    def potentials(self, y):
        """Return a 3-tuple of potential, 1st and 2nd derivs given y."""
        pass
    
    def findH(self,potential,y):
        """Return value of comoving Hubble variable given potential and y."""
        pass
    
    def run(self, saveresults=True):
        """Execute a simulation run using the parameters already provided."""
        if self.solver not in self.solverlist:
            raise ModelError("Unknown solver!")
        #Test whether k exists and if so change init conditions
        
#         if self.k is not None and (self.solver == "odeint" or self.solver == "rkdriver_dumb"):
#             #Make the initial conditions the right shape
#             if type(self.k) is N.ndarray or type(self.k) is list: 
#                 self.ystart = self.ystart.reshape((len(self.ystart),1))*N.ones((len(self.ystart),len(self.k)))
                        
        if self.solver == "odeint":
            try:
                self.tresult, self.yresult, self.nok, self.nbad = rk4.odeint(self.ystart, self.tstart,
                    self.tend, self.tstep_wanted, self.tstep_min, self.derivs, self.eps, self.dxsav)
                #Commented out next line to work with array of k values
                #self.yresult = N.hsplit(self.yresult, self.yresult.shape[1])
            except rk4.SimRunError, er:
                self.yresult = N.array(er.yresult)
                self.tresult = N.array(er.tresult)
                print "Error during run, but some results obtained: ", er.message
            except StandardError, er:
                #raise ModelError("Error running odeint", self.tresult, self.yresult)
                raise
        
        if self.solver == "rkdriver_dumb":
            #Loosely estimate number of steps based on requested step size
            nstep = N.ceil((self.tend - self.tstart)/self.tstep_wanted)
            try:
                self.tresult, self.yresult = rk4.rkdriver_dumb(self.ystart, self.tstart, self.tend, nstep, self.derivs)
            except StandardError, er:
                merror = ModelError("Error running rkdriver_dumb:\n" + er.message)
                raise merror
        
        if self.solver == "scipy_odeint":
            
            #Use scipy solver. Need to massage derivs into right form.
            #swap_derivs = lambda y, t : self.derivs(t,y)
                        
            #Now split depending on whether k exists
            if type(self.k) is N.ndarray or type(self.k) is list:
                #Get set of times for each k
                if type(self.tstart) is N.ndarray or type(self.tstart) is list:
                    times = N.arange(self.tstart.min(), self.tend + self.tstep_wanted, self.tstep_wanted)
                    startindices = [N.where(ts <= times)[0][0] for ts in self.tstart]
                else:
                    times = N.arange(self.tstart, self.tend + self.tstep_wanted, self.tstep_wanted)
                    startindices = [0]
                #Make a copy of k and ystart while we work
                klist = N.copy(self.k)
                yslist = N.rollaxis(N.copy(self.ystart),1,0)
                
                #Do calculation
                #Compute list of ks in a row
                ylist = [scipy_odeint(self.derivs, ys, times[ts:]) for self.k, ys, ts in zip(klist,yslist,startindices)]
                ylistlengths = [len(ys) for ys in ylist]
                ylist = [helpers.nanfillstart(y, max(ylistlengths)) for y in ylist]
                #Now stack results to look like as normal (time,variable,k)
                self.yresult = N.dstack(ylist)
                self.tresult = times
                #Return klist to normal
                self.k = klist
            else:
                times = N.arange(self.tstart, self.tend + self.tstep_wanted, self.tstep_wanted)
                self.yresult = scipy_odeint(self.derivs, self.ystart, times)
                self.tresult = times
            
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()
        
        self.resultlist.append([self.lastparams, self.tresult, self.yresult])        
        self.runcount += 1
        
        if saveresults:
            try:
                print "Results saved in " + self.saveallresults()
            except IOError, er:
                print "Error trying to save results! Results NOT saved."
                print er
            
        return
    
    def callingparams(self):
        """Returns list of parameters to save with results."""
        #Test whether k has been set
        try:
            self.k
        except (NameError, AttributeError):
            self.k=None
        try:
            self.mass
        except (NameError, AttributeError):    
            self.mass=None
            
        #Form dictionary of inputs
        params = {"ystart":self.ystart, 
                  "tstart":self.tstart,
                  "tend":self.tend,
                  "tstep_wanted":self.tstep_wanted,
                  "tstep_min":self.tstep_min,
                  "k":self.k, #model dependent params
                  "mass":self.mass,
                  "eps":self.eps,
                  "dxsav":self.dxsav,
                  "solver":self.solver,
                  "classname":self.__class__.__name__,
                  "CVSRevision":"$Revision: 1.79 $",
                  "datetime":datetime.datetime.now()
                  }
        return params
               
    def argstring(self):
        a = r"; Arguments: ystart="+ str(self.ystart) + r", tstart=" + str(self.tstart) 
        a += r", tend=" + str(self.tend) + r", mass=" + str(self.mass)
        return a
    
    def saveplot(self, fig):
        """Save figure fig in directory graphs"""
        time = self.lastparams["datetime"].strftime("%Y%m%d%H%M%S")
        filename = "./graphs/run" + time + ".png"
            
        if os.path.isdir(os.path.dirname(filename)):
            if os.path.isfile(filename):
                raise IOError("File already exists!")
        else:
            raise IOError("Directory 'graphs' does not exist")
        try:
            fig.savefig(filename)
            print "Plot saved as " + filename
        except IOError:
            raise
        return
        
    def plotresults(self, fig=None, show=True, varindex=None, klist=None, saveplot=False):
        """Plot results of simulation run on a graph.
            Return figure instance used."""
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!")
        
        if varindex is None:
            varindex = 0 #Set default list of variables to plot
        
        if fig is None:
            fig = P.figure() #Create figure
        else:
            P.figure(fig.number)
        #One plot command for with ks, one for without
        
        if klist is None:
            P.plot(self.tresult, self.yresult[:,varindex])
        else:
            P.plot(self.tresult, self.yresult[:,varindex,klist])
        #Create legends and axis names
        P.xlabel(self.tname)
        P.ylabel(self.ynames[varindex])
        P.legend([r"$k=" + helpers.eto10(ks) + "$" for ks in self.k[klist]])
        #P.title(self.plottitle, figure=fig)
        
        #Should we show it now or just return it without showing?
        if show:
            P.show()
        #Should we save the plot somewhere?
        if saveplot:
            self.saveplot(fig)   
        #Return the figure instance
        return fig
            
    def plot3dresults(self, fig=None, show=True, varindex=None, klist=None, kfunction=None, saveplot=False):
        """Plot results for different ks in 3d plot. Can only plot a single variable at a time."""
        #Test whether model has run yet
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!")
        
        #Test whether model has k variable dependence
        try:
            self.yresult[0,0,0] #Does this exist?
        except IndexError, er:
            raise ModelError("This model does not have any k variable to plot in third dimension! Got " + er.message)
        
        if varindex is None:
            varindex = 0 #Set variable to plot
        if klist is None:
            klist = N.arange(len(self.k)) #Plot all ks
        
        if fig is None:
            fig = P.figure() #Create figure
        else:
            P.figure(fig.number)
        
        #Plot 3d figure
        
        x = self.tresult
        
        ax = axes3d.Axes3D(fig)
        #plot lines in reverse order
        for kindex in klist[::-1]:
            z = self.yresult[:,varindex,kindex]
            #Do we need to change k by some function (e.g. log)?
            if kfunction is None:
                y = self.k[kindex]*N.ones(len(x))
            else:
                y = kfunction(self.k[kindex])*N.ones(len(x))
            #Plot the line
            ax.plot3D(x,y,z,color="b")
        ax.set_xlabel(self.tname)
        if kfunction is None:
            ax.set_ylabel(r"$k$")
        else:
            ax.set_ylabel(r"Function of k: " + kfunction.__name__)
        ax.set_zlabel(self.ynames[varindex])
        P.title(self.plottitle + self.argstring())
        
        #Should we show it now or just return it without showing?
        if show:
            P.show()
        #Should we save the plot somewhere?
        if saveplot:
            self.saveplot(fig)
        return fig
            
    def plotkcrosssection(self, tindex=None, fig=None, show=True, varindex=None, klist=None, kfunction=None, saveplot=False):
        """Plot results for different ks in 3d plot. Can only plot a single variable at a time."""
        #Test whether model has run yet
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!")
        
        #Test whether model has k variable dependence
        try:
            self.yresult[0,0,0] #Does this exist?
        except IndexError, er:
            raise ModelError("This model does not have any k variable to plot! Got " + er.message)
        
        if varindex is None:
            varindex = 0 #Set variable to plot
        if klist is None:
            klist = N.arange(len(self.k)) #Plot all ks
        if tindex is None:
            tindex = N.arange(0,len(self.tresult), 1000) #Selection of time slices
        #Set names for t slices
        #tnames = str(self.tresult[tindex])
        
        if fig is None:
            fig = P.figure() #Create figure
        else:
            P.figure(fig.number)
        
        #Plot figure, default is semilogx for k
        P.semilogx(self.k[klist], self.yresult[tindex,varindex,:][:,klist].transpose(), 'o-')
                
        #Create legends and axis names
        P.xlabel(r"$k$")
        #P.legend(tnames)
        P.legend([r"$t=" + str(ts) + "$" for ts in self.tresult[tindex]])
                
        #Should we show it now or just return it without showing?
        if show:
            P.show()
        #Should we save the plot somewhere?
        if saveplot:
            self.saveplot(fig)
        return fig
    def saveallresults(self, filename=None):
        """Tries to save file as a pickled object in directory 'results'."""
        
        now = self.lastparams["datetime"].strftime("%Y%m%d%H%M%S")
        if not filename:
            filename = "./results/run" + now
            
        if os.path.isdir(os.path.dirname(filename)):
            if os.path.isfile(filename):
               raise IOError("File already exists!")
        else:
            raise IOError("Directory 'results' does not exist")
        
        try:
            resultsfile = open(filename, "w")
            try:
                pickle.dump(self.resultlist, resultsfile)
            finally:
                resultsfile.close()
        except IOError:
            raise
        
        return filename
    
    def loadresults(self, filename):
        """Loads results from a file and appends them to current results list."""
        
        if not os.path.isfile(filename):
            raise IOError("File does not exist!")
        
        try:
            resultsfile = open(filename, "r")
            try:
                newresults = pickle.load(resultsfile)
                #The following doesn't check to see if type of object is right!
                self.resultlist.extend(newresults) 
                self.runcount = len(self.resultlist)
            finally:
                resultsfile.close()
        except IOError:
                raise
            
class TestModel(CosmologicalModel):
    """Test class defining a very simple function"""
    def __init__(self, ystart=N.array([1.0,1.0]), tstart=0.0, tend=1.0, tstep_wanted=0.01, tstep_min=0.001):
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min)
        
        self.plottitle = r"TestModel: $\frac{d^2y}{dt^2} = y$"
        self.tname = "Time"
        self.ynames = [r"Simple $y$", r"$\dot{y}$"]
    
    def derivs(self, y, t):
        """Very simple set of ODEs"""
        dydx = N.zeros(2)
        
        dydx[0] = y[1]
        dydx[1] = y[0]
        return dydx

class BasicBgModel(CosmologicalModel):
    """Basic model with background equations
        Array of dependent variables y is given by:
        
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - a : Scale Factor
    """
    
    def __init__(self, ystart=N.array([0.1,0.1,0.1]), tstart=0.0, tend=120.0, 
                    tstep_wanted=0.02, tstep_min=0.0001, solver="scipy_odeint"):
        
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, solver=solver)
        
        #Mass of inflaton in Planck masses
        self.mass = 1.0
        
        self.plottitle = "Basic Cosmological Model"
        self.tname = "Conformal time"
        self.ynames = [r"Inflaton $\phi$", "", r"Scale factor $a$"]
        
    def potentials(self, y):
        """Return value of potential at y, along with first and second derivs."""
        
        #Use inflaton mass
        mass2 = self.mass**2
        
        #potential U = 1/2 m^2 \phi^2
        U = 0.5*(mass2)*(y[0]**2)
        #deriv of potential wrt \phi
        dUdphi =  (mass2)*y[0]
        #2nd deriv
        d2Udphi2 = mass2
        
        return U,dUdphi,d2Udphi2
    
    def derivs(self, y, t):
        """Basic background equations of motion.
            dydx[0] = dy[0]/d\eta etc"""
        
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y)
        
        #factor in eom [1/3 a^2 U_0]^{1/2}
        Ufactor = N.sqrt((1.0/3.0)*(y[2]**2)*U)
        
        #Set derivatives
        dydx = N.zeros(3)
        
        #d\phi_0/d\eta = y_1
        dydx[0] = y[1] 
        
        #dy_1/d\eta = -2
        dydx[1] = -2*Ufactor*y[1] - (y[2]**2)*dUdphi
        
        #da/d\eta = [1/3 a^2 U_0]^{1/2}*a
        dydx[2] = Ufactor*y[2]
        
        return dydx
    
    def plotresults(self, saveplot = False):
        """Plot results of simulation run on a graph."""
        
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!", self.tresult, self.yresult)
        
        CosmologicalModel.plotresults(self, varindex=[0,2], saveplot=saveplot)
        
        return

class EfoldModel(CosmologicalModel):
    """Base class for models which use efold time variable n.
        Provides some of the functions needed to deal with this situation.
        Need at least the following three variables:
        y[0] - \phi_0 : Background inflaton
        y[1] - d\phi_0/d\n : First deriv of \phi
        y[2] - H: Hubble parameter
        """
    
    def __init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, solver):
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, solver=solver)
        #Mass of inflaton in Planck masses
        self.mass = 1.0e-6 # COBE normalization from Liddle and Lyth
    
    def potentials(self, y):
        """Return value of potential at y, along with first and second derivs."""
        
        #Use inflaton mass
        mass2 = self.mass**2
        
        #potential U = 1/2 m^2 \phi^2
        U = 0.5*(mass2)*(y[0]**2)
        #deriv of potential wrt \phi
        dUdphi =  (mass2)*y[0]
        #2nd deriv
        d2Udphi2 = mass2
        
        return U,dUdphi,d2Udphi2         
    
    def findH(self, U, y):
        """Return value of Hubble variable, H at y for given potential."""
        phidot = y[1]
        
        #Expression for H
        H = N.sqrt(U/(3.0-0.5*(phidot**2)))
        return H
    
    def findinflend(self):
        """Find the efold time where inflation ends,
            i.e. the hubble flow parameter epsilon >1.
            Returns tuple of endefold and endindex (in tresult)."""
        
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!", self.tresult, self.yresult)
        
        self.epsilon = self.getepsilon()
        if not any(self.epsilon>1):
            raise ModelError("Inflation did not end during specified number of efoldings. Increase tend and try again!")
        endindex = N.where(self.epsilon>1)[0][0]
        
        #Interpolate results to find more accurate endpoint
        tck = interpolate.splrep(self.tresult[:endindex], self.epsilon[:endindex])
        t2 = N.linspace(self.tresult[endindex-1], self.tresult[endindex], 100)
        y2 = interpolate.splev(t2, tck)
        endindex2 = N.where(y2>1)[0][0]
        #Return efold of more accurate endpoint
        endefold = t2[endindex2]
        
        return endefold, endindex
    
    def getepsilon(self):
        """Return an array of epsilon = -\dot{H}/H values for each timestep."""
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!")

        #Find Hdot
        if self.yresult.ndim == 3:
            Hdot = N.array(map(self.derivs, self.yresult, self.tresult))[:,2,0]
            epsilon = - Hdot/self.yresult[:,2,0]
        else:
            Hdot = N.array(map(self.derivs, self.yresult, self.tresult))[:,2]
            epsilon = - Hdot/self.yresult[:,2]
        
        return epsilon
        
class BgModelInN(EfoldModel):
    """Basic model with background equations in terms of n
        Array of dependent variables y is given by:
        
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\n : First deriv of \phi
       y[2] - H: Hubble parameter
    """
    
    def __init__(self, ystart=N.array([15.0,-1.0,0.0]), tstart=0.0, tend=80.0, tstep_wanted=0.01, tstep_min=0.0001, solver="scipy_odeint"):
        EfoldModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, solver=solver)
        
        #Mass of inflaton in Planck masses
        #self.mass = 1.0e-6 # COBE normalization from Liddle and Lyth
        
        #Set initial H value if None
        if self.ystart[2] == 0.0:
            U = self.potentials(self.ystart)[0]
            #self.ystart[2] = N.sqrt(U/(3.0-0.5*(self.ystart[1]**2)))
            self.ystart[2] = self.findH(U, self.ystart)
        
        #Titles
        self.plottitle = r"Basic (improved) Cosmological Model in $n$"
        self.tname = r"E-folds $n$"
        self.ynames = [r"$\phi$", r"$\dot{\phi}_0$", r"$H$"]
    
      
    
    def derivs(self, y, t):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y)        
        
        #Set derivatives
        dydx = N.zeros(3)
        
        #d\phi_0/dn = y_1
        dydx[0] = y[1] 
        
        #dphi^prime/dn
        dydx[1] = -(U*y[1] + dUdphi)/(y[2]**2)
        
        #dH/dn
        dydx[2] = -0.5*(y[1]**2)*y[2]

        return dydx
    
         
    def plotresults(self, saveplot = False):
        """Plot results of simulation run on a graph."""
        
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!", self.tresult, self.yresult)
        
        f = P.figure()
        
        #First plot of phi and phi^dot
        P.subplot(121)
        EfoldModel.plotresults(self, fig=f, show=False, varindex=[0,1], saveplot=False)
        
        #Second plot of H
        P.subplot(122)
        EfoldModel.plotresults(self, fig=f, show=False, varindex=[2], saveplot=False)
        
        P.show()
        return

class FirstOrderInN(EfoldModel):
    """First order model using efold as time variable.
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - H : Hubble parameter
       y[3] - \delta\varphi_1 : First order perturbation
       y[4] - \delta\varphi_1^\prime : Derivative of first order perturbation
       """
    def __init__(self, ystart=None, tstart=0.0, tend=80.0, tstep_wanted=0.01, tstep_min=0.0001, k=None, ainit=None, solver="scipy_odeint"):
        """Initialize all variables and call ancestor's __init__ method."""
        EfoldModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, solver=solver)
        
        if ainit is None:
            #Don't know value of ainit yet so scale it to 1
            self.ainit = 1
        else:
            self.ainit = ainit
        
        #Let k roam for a start
        if k is None:
            self.k = 10**(N.arange(10.0)-8)
        else:
            self.k = k
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart = N.array([15.0,-0.1,0.0,0.1,0.1])   
        
        #Set initial H value if None
        if self.ystart[2] == 0.0:
            U = self.potentials(self.ystart)[0]
            self.ystart[2] = self.findH(U, self.ystart)
        #Make ystart right shape
        self.ystart = N.outer(N.ones(len(self.k)), self.ystart)
                
        #Text for graphs
        self.plottitle = "First Order Model in Efold time"
        self.tname = r"$n$"
        self.ynames = [r"$\varphi_0$", 
                        r"$\dot{\varphi_0}$",
                        r"$H$",
                        r"$\delta\varphi_1$",
                        r"$\dot{\delta\varphi_1}$"]
                    
    def derivs(self, y, t):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y)        
        
        #Set derivatives taking care of k type
        if type(self.k) is N.ndarray or type(self.k) is list: 
            dydx = N.zeros((5,len(self.k)))
        else:
            dydx = N.zeros(5)
            
        
        #d\phi_0/dn = y_1
        dydx[0] = y[1] 
        
        #dphi^prime/dn
        dydx[1] = -(U*y[1] + dUdphi)/(y[2]**2)
        
        #dH/dn
        dydx[2] = -0.5*(y[1]**2)*y[2]
        
        #d\deltaphi_1/dn = y[4]
        dydx[3] = y[4]
        
        #Get a
        a = self.ainit*N.exp(t)
        
        #d\deltaphi_1^prime/dn
        dydx[4] = (-(3 + dydx[2]/y[2])*y[4] - ((self.k/(a*y[2]))**2)*y[3] - (d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[3]/(y[2]**2)))
        
        return dydx       

class ComplexFirstOrderInN(EfoldModel):
    """First order model using efold as time variable.
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - H : Hubble parameter
       y[3] - \delta\varphi_1 : First order perturbation [Real Part]
       y[4] - \delta\varphi_1^\prime : Derivative of first order perturbation [Real Part]
       y[5] - \delta\varphi_1 : First order perturbation [Imag Part]
       y[6] - \delta\varphi_1^\prime : Derivative of first order perturbation [Imag Part]
       """
    def __init__(self, ystart=None, tstart=0.0, tend=80.0, tstep_wanted=0.01, tstep_min=0.0001, k=None, ainit=None, solver="scipy_odeint"):
        """Initialize all variables and call ancestor's __init__ method."""
        EfoldModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, solver=solver)
        
        if ainit is None:
            #Don't know value of ainit yet so scale it to 1
            self.ainit = 1
        else:
            self.ainit = ainit
        
        #Let k roam for a start
        if k is None:
            self.k = 10**(N.arange(10.0)-8)
        else:
            self.k = k
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart = N.array([15.0,-0.1,0.0,1.0,0.0,1.0,0.0])   
        
        #Set initial H value if None
        if all(self.ystart[2] == 0.0):
            U = self.potentials(self.ystart)[0]
            self.ystart[2] = self.findH(U, self.ystart)
            
        #Text for graphs
        self.plottitle = "Complex First Order Model in Efold time"
        self.tname = r"$n$"
        self.ynames = [r"$\varphi_0$",
                        r"$\dot{\varphi_0}$",
                        r"$H$",
                        r"Real $\delta\varphi_1$",
                        r"Real $\dot{\delta\varphi_1}$",
                        r"Imag $\delta\varphi_1$",
                        r"Imag $\dot{\delta\varphi_1}$"]
                    
    def derivs(self, y, t):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y)        
        
        #Set derivatives taking care of k type
        if type(self.k) is N.ndarray or type(self.k) is list: 
            dydx = N.zeros((7,len(self.k)))
        else:
            dydx = N.zeros(7)
            
        
        #d\phi_0/dn = y_1
        dydx[0] = y[1] 
        
        #dphi^prime/dn
        dydx[1] = -(U*y[1] + dUdphi)/(y[2]**2)
        
        #dH/dn
        dydx[2] = -0.5*(y[1]**2)*y[2]
        
        #d\deltaphi_1/dn = y[4]
        dydx[3] = y[4]
        
        #Get a
        a = self.ainit*N.exp(t)
        
        #d\deltaphi_1^prime/dn
        dydx[4] = (-(3 + dydx[2]/y[2])*y[4] - ((self.k/(a*y[2]))**2)*y[3] 
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[3]/(y[2]**2)))
        #print dydx[4]
        
        #Complex parts
        dydx[5] = y[6]
        
        dydx[6] = (-(3 + dydx[2]/y[2])*y[6] - ((self.k/(a*y[2]))**2)*y[5] 
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[5]/(y[2]**2)))
        
        return dydx
   
class TwoStageModel(EfoldModel):
    """Uses both BgModelInN and ComplexModelInN to run a full simulation.
        Main additional functionality is in determining initial conditions.
        Variables finally stored are as in ComplexModelInN.
    """                
    def __init__(self, ystart=None, tstart=0.0, tend=120.0, tstep_wanted=0.01, tstep_min=0.0001, k=None, ainit=None, solver="scipy_odeint"):
        """Initialize model and ensure initial conditions are sane."""
        EfoldModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min, solver=solver)
        
        if ainit is None:
            #Don't know value of ainit yet so scale it to 1
            self.ainit = 1
        else:
            self.ainit = ainit
        
        #Set constant factor for 1st order initial conditions
        self.cq = 50
        
        #Let k roam if we don't know correct ks
        if k is None:
            self.k = 10**(N.arange(7.0)-62)
        else:
            self.k = k
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            #Initial conditions for all variables
            self.ystart = N.array([18.0, # \phi_0
                                   -0.1, # \dot{\phi_0}
                                    0.0, # H - leave as 0.0 to let program determine
                                    1.0, # Re\delta\phi_1
                                    0.0, # Re\dot{\delta\phi_1}
                                    1.0, # Im\delta\phi_1
                                    0.0  # Im\dot{\delta\phi_1}
                                    ])   
        
        #Set initial H value if None
        if self.ystart[2] == 0.0:
            U = self.potentials(self.ystart)[0]
            self.ystart[2] = self.findH(U, self.ystart)
        
        #Set up variables for the two models
        self.bgmodel = self.firstordermodel = None
    
    def finda_end(self, Hend, Hreh=None):
        """Given the Hubble parameter at the end of inflation and at the end of reheating
            calculate the scale factor at the end of inflation."""
        if Hreh is None:
            Hreh = Hend #Instantaneous reheating
        a_0 = 1 # Normalize today
        a_end = a_0*N.exp(-72.3)*((Hreh/(Hend**4.0))**(1.0/6.0))
        return a_end
        
    def findkcrossing(self, k, t, H):
        """Given k, time variable and Hubble parameter, find when mode k crosses the horizon."""
        #threshold
        err = 1.0e-26
        #get aHs
        aH = self.ainit*N.exp(t)*H
        try:
            kcrindex = N.where(N.sign(k - (self.cq*aH))<=0)[0][0]
        except IndexError, ex:
            raise ModelError("k mode " + str(k) + " crosses horizon after end of inflation!")
        kcrefold = t[kcrindex]
        return kcrindex, kcrefold
    
    def findallkcrossings(self, t, H):
        """Iterate over findkcrossing to get full list"""
        return N.array([self.findkcrossing(onek, t, H) for onek in self.k])
        
    def setfirstorderics(self):
        """After a bg run has completed, set the initial conditions for the 
            first order run."""
        #Check if bg run is completed
        if self.bgmodel.runcount == 0:
            raise ModelError("Background system must be run first before setting 1st order ICs!")
        
        #Find initial conditions for 1st order model
        #Find a_end using instantaneous reheating
        Hend = self.bgmodel.yresult[self.fotendindex,2]
        self.a_end = self.finda_end(Hend)
        self.ainit = self.a_end*N.exp(-self.fotend)
        
        #find k crossing indices
        kcrossings = self.findallkcrossings(self.bgmodel.tresult[:self.fotendindex], 
                            self.bgmodel.yresult[:self.fotendindex,2])
        kcrossefolds = kcrossings[:,1]
                
        #If mode crosses horizon before t=0 then we will not be able to propagate it
        if any(kcrossefolds==0):
            raise ModelError("Some k modes crossed horizon before simulation began and cannot be initialized!")
        
        #Find new start time from earliest kcrossing
        self.fotstart, self.fotstartindex = kcrossefolds, kcrossings[:,0].astype(N.int)
        
        #Reset starting conditions at new time
        self.foystart = N.zeros((len(self.ystart), len(self.k)))
        
        
        #Mould init conditions into right shape for number of ks
        #if self.foystart.ndim == 1:
         #   self.foystart = self.foystart[:,N.newaxis]*N.ones(len(self.k))
            
        self.foystart[0:3] = self.bgmodel.yresult[self.fotstartindex,:].transpose()
                   
        #Set Re\delta\phi_1 initial condition
        self.foystart[3,:] = 1/(N.sqrt(2*self.cq))
        #set Re\dot\delta\phi_1 ic
        self.foystart[4,:] = -N.sqrt(self.cq/2)
        #Set Im\delta\phi_1
        self.foystart[5,:] = 0
        #Set Im\dot\delta\phi_1
        self.foystart[6,:] = 0
        
        return
        
    def setfoicsplain(self):
        #Testing
        
        #Check if bg run is completed
        if self.bgmodel.runcount == 0:
            raise ModelError("Background system must be run first before setting 1st order ICs!")
        
        #Find initial conditions for 1st order model
        #Find a_end using instantaneous reheating
        Hend = self.bgmodel.yresult[self.fotendindex,2]
        self.a_end = self.finda_end(Hend)
        self.ainit = self.a_end*N.exp(-self.fotend)
        
        #Reset starting conditions at new time
        self.foystart = N.zeros((len(self.ystart), len(self.k)))
        self.fotstart = N.zeros((len(self.k)))
        
        self.foystart[0:3] = self.bgmodel.yresult[N.zeros(len(self.k)).astype(int),:].transpose()
        self.foystart[3,:] = self.foystart[5,:] = 0.01
        self.foystart[4,:] = self.foystart[6,:] = 0.0
        #self.fotstart[:] = 0
        
        return
        
    def runbg(self):
        """Run bg model after setting initial conditions."""
        #Check ystart is in right form (1-d array of three values)
        if self.ystart.ndim == 1:
            ys = self.ystart[0:3]
        elif self.ystart.ndim == 2:
            ys = self.ystart[0:3,0]
        self.bgmodel = BgModelInN(ystart=ys, tstart=self.tstart, tend=self.tend, 
                            tstep_wanted=self.tstep_wanted, tstep_min=self.tstep_min, solver=self.solver)
        
        #Start background run
        print("Running background model...\n")
        try:
            self.bgmodel.run(saveresults=False)
        except ModelError, er:
            print "Error in background run, aborting! Message: " + er.message
        #Find end of inflation
        self.fotend, self.fotendindex = self.bgmodel.findinflend()
        print("Background run complete, inflation ended " + str(self.fotend) + " efoldings after start.")
        return
        
    def runfo(self):
        """Run first order model after setting initial conditions."""
                
        #Initialize first order model
        self.firstordermodel = ComplexFirstOrderInN(ystart=self.foystart, tstart=self.fotstart, tend=self.fotend,
                                tstep_wanted=self.tstep_wanted, tstep_min=self.tstep_min, solver=self.solver,
                                k=self.k, ainit=self.ainit)
        #Set names as in ComplexModel
        self.tname, self.ynames = self.firstordermodel.tname, self.firstordermodel.ynames
        #Start first order run
        print("Beginning first order run...\n")
        try:
            self.firstordermodel.run(saveresults=False)
        except ModelError, er:
            print "Error in first order run, aborting! Message: " + er.message
        
        #Set results to current object
        self.tresult, self.yresult = self.firstordermodel.tresult, self.firstordermodel.yresult
        return
    
    def run(self, saveresults=True, plain=False):
        """Run BgModelInN with initial conditions and then use the results
            to run ComplexModelInN."""
        #Run bg model
        self.runbg()
        
        if plain:
            self.setfoicsplain()
        else:
            #Set initial conditions for first order model
            self.setfirstorderics()
        
        #Run first order model
        self.runfo()
        
        #Save results in resultlist and file
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()
        
        self.resultlist.append([self.lastparams, self.tresult, self.yresult])        
        self.runcount += 1
        
        if saveresults:
            try:
                print "Results saved in " + self.saveallresults()
            except IOError, er:
                print "Error trying to save results! Results NOT saved."
                print er                
        
        return
        
            
class FirstOrderModel(CosmologicalModel):
    """First order model with background equations
        Array of dependent variables y is given by:
        
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - \delta\varphi_1 : First order perturbation
       y[3] - \delta\varphi_1^\prime : Derivative of first order perturbation
       y[4] - a : Scale Factor
       
       Results can be saved in a pickled file as a list of tuples of the following
       structure:
       resultset = (callingparams, tresult, yresult)
       
       callingparams is formatted as in the function callingparams(self) below
       
    """
    
    def __init__(self, ystart=None, tstart=0.0, tend=120.0, tstep_wanted=0.02, tstep_min=0.0001, k=None):
        """Initialize all variables and call ancestor's __init__ method."""
        
        #Let k roam for a start
        if k is None:
            self.k = 10**(N.arange(10.0)-8)
        else:
            self.k = k
        
        #Initial conditions for each of the variables.
        if ystart is None:
            ystart = (N.array([[0.1],[0.1],[0.1],[0.1],[0.1]])*N.ones((5,len(self.k))))
        
        CosmologicalModel.__init__(self, ystart, tstart, tend, tstep_wanted, tstep_min)
        
        
        #Mass of inflaton in Planck masses
        self.mass = 1.0
                        
        #Text for graphs
        self.plottitle = "First Order Model"
        self.tname = r"$\eta$"
        self.ynames = [r"$\varphi_0$", 
                        r"$\varphi_0^\prime$",
                        r"$\delta\varphi_1$",
                        r"$\delta\varphi_1^\prime$",
                        r"$a$"]
        
    def callingparams(self):
        """Returns dictionary of parameters to save with results."""
        
        params = {"ystart":self.ystart, 
                  "tstart":self.tstart,
                  "tend":self.tend,
                  "tstep_wanted":self.tstep_wanted,
                  "tstep_min":self.tstep_min,
                  "k":self.k,
                  "mass":self.mass,
                  "eps":self.eps,
                  "dxsav":self.dxsav,
                  "solver":self.solver,
                  "classname":self.__class__.__name__,
                  "CVSRevision":"$Revision: 1.79 $",
                  "datetime":datetime.datetime.now()
                  }
        return params
    
    def derivs(self, y, t):
        """First order equations of motion.
            dydx[0] = dy[0]/d\eta etc"""
        
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y)
        
        #Things we only want to calculate once
        #a^2
        asq = y[4]**2
        
        #factor in eom \mathcal{H} = [1/3 a^2 U_0]^{1/2}
        H = self.findH(U, y)
        
        #Set derivatives
        dydx = N.zeros((5,len(self.k)))
        
        #d\phi_0/d\eta = y_1
        dydx[0] = y[1] 
        
        #d^2phi/d\eta^2 = -2Hphi^prime -a^2U_,phi
        dydx[1] = -2*H*y[1] - asq*dUdphi
        
        #dy_2/d\eta = \delta\phi_1^prime
        dydx[2] = y[3]
        
        #delta\phi^prime^prime
        dydx[3] = -2*H*y[3] - (self.k**2)*y[2] - asq*d2Udphi2*y[2]
        
        #da/d\eta = [1/3 a^2 U_0]^{1/2}*a
        dydx[4] = H*y[4]
        
        return dydx
        
    def findH(self, potential, y):
        """Return value of comoving Hubble variable, \mathcal{H} at y for given potential."""
        phiprime = y[1]
        a = y[4]
        
        #Expression for H
        H = N.sqrt((1.0/3.0)*((a**2)*potential + 0.5*(phiprime**2)))
        
        return H
    
    def potentials(self, y):
        """Return value of potential at y, along with first and second derivs."""
        
        #Use inflaton mass
        mass2 = self.mass**2
        
        #potential U = 1/2 m^2 \phi^2
        U = 0.5*(mass2)*(y[0]**2)
        #deriv of potential wrt \phi
        dUdphi =  (mass2)*y[0]
        #2nd deriv
        d2Udphi2 = mass2
        
        return U,dUdphi,d2Udphi2
        
    def plotresults(self, saveplot = False):
        """Plot results of simulation run on a graph."""
        
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!", self.tresult, self.yresult)
        for kindex in N.arange(len(self.k)):
            P.plot(self.tresult, self.yresult[:,0,kindex], self.tresult, self.yresult[:,2,kindex])
        P.xlabel(self.tname)
        P.ylabel("")
        P.legend((self.ynames[0], self.ynames[2]))
        P.title(self.plottitle + self.argstring())
        P.show()
        return
    
    def plotallks(self, varindex=2, kfunction=None):
        """Plot results from all ks run in a plot where k is plotted by kfunction (e.g. log)."""
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!", self.tresult, self.yresult)
                
        x = self.tresult
        
        fig = P.figure()
        ax = axes3d.Axes3D(fig)
        #plot lines in reverse order
        for index, kitem in zip(range(len(self.k))[::-1], self.k[::-1]):
            z = self.yresult[:,varindex,index]
            if kfunction is None:
                y = kitem*N.ones(len(x))
            else:
                y = kfunction(kitem)*N.ones(len(x))
            ax.plot3D(x,y,z,color="b")
        ax.set_xlabel(self.tname)
        if kfunction is None:
            ax.set_ylabel(r"$k$")
        else:
            ax.set_ylabel(r"Function of k: " + kfunction.__name__)
        ax.set_zlabel(self.ynames[varindex])
        P.title(self.plottitle + self.argstring())
        P.show()
        
        return
        
    def plotkdiffs(self, varindex=2, kfunction=None, basekindex=0):
        """Plot the difference of k results to a specific k"""
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!", self.tresult, self.yresult)
                
        x = self.tresult
        
        fig = P.figure()
        ax = axes3d.Axes3D(fig)
        #plot lines in reverse order
        for index, kitem in zip(range(len(self.k))[::-1], self.k[::-1]):
            z = self.yresult[:,varindex,index] - self.yresult[:,varindex,basekindex]
            if kfunction is None:
                y = kitem*N.ones(len(x))
            else:
                y = kfunction(kitem)*N.ones(len(x))
            ax.plot3D(x,y,z)
        ax.set_xlabel(self.tname)
        if kfunction is None:
            ax.set_ylabel(r"$k$")
        else:
            ax.set_ylabel(r"Function of k: " + kfunction.__name__)
        ax.set_zlabel(self.ynames[varindex])
        P.title(self.plottitle + self.argstring())
        P.show()
        
        return
    
    def getallHs(self):
        """Computes all H values for current tresult and yresult results. Stored in self.Hresult."""
        self.Hresult = N.array([self.findH(self.potentials(row)[0],row) for row in self.yresult])
        return
        
class FullFirstOrder(FirstOrderModel):
    """Full (not slow roll) first order model"""
    
    def findH(self, potential, y):
        """Return value of comoving Hubble variable, \mathcal{H} at y for given potential."""
        phiprime = y[1]
        a = y[4]
        
        #Expression for H
        H = N.sqrt((1.0/3.0)*((a**2)*potential + 0.5*(phiprime**2)))
        
        return H
    
    def potentials(self, y):
        """Return value of potential at y, along with first and second derivs."""
        
        #Use inflaton mass
        mass2 = self.mass**2
        
        #potential U = 1/2 m^2 \phi^2
        U = 0.5*(mass2)*(y[0]**2)
        #deriv of potential wrt \phi
        dUdphi =  (mass2)*y[0]
        #2nd deriv
        d2Udphi2 = mass2
        
        return U,dUdphi,d2Udphi2
    
    def derivs(self, y, t):
        """First Order eqs of motion"""
        
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y)
        
        #Things we only want to calculate once
        #a^2
        asq = y[4]**2
        
        #factor in eom \mathcal{H} = [1/3 a^2 U_0]^{1/2}
        H = self.findH(U, y)
        
        #Set derivatives
        dydx = N.zeros((5,len(self.k)))
        
        #d\phi_0/d\eta = y_1
        dydx[0] = y[1] 
        
        #d^2phi/d\eta^2 = -2Hphi^prime -a^2U_,phi
        dydx[1] = -2*H*y[1] - asq*dUdphi
        
        #dy_2/d\eta = \delta\phi_1^prime
        dydx[2] = y[3]
        
        #delta\phi^prime^prime
        dydx[3] = -2*H*y[3] - (self.k**2)*y[2] - y[2]*( asq*d2Udphi2  + (y[1]**2)*U*asq/(H**2)  + 2*y[1]*dUdphi*asq/H )
        
        #da/d\eta = [1/3 a^2 U_0]^{1/2}*a
        dydx[4] = H*y[4]
        
        return dydx
        
            
class HarmonicFirstOrder(FirstOrderModel):
    """Just change derivs to get harmonic motion"""
            
    def derivs(self, y, t):
        """First order equations of motion.
            dydx[0] = dy[0]/d\eta etc"""
        
        #Use inflaton mass
        mass2 = self.mass**2
        
        #potential U = 1/2 m^2 \phi^2
        U = 0.5*(mass2)*(y[0]**2)
        #deriv of potential wrt \phi
        dUdphi =  (mass2)*y[0]
        #2nd deriv
        d2Udphi2 = mass2
        
        #Things we only want to calculate once
        #a^2
        asq = y[4]**2
        
        #factor in eom \mathcal{H} = [1/3 a^2 U_0]^{1/2}
        H = N.sqrt((1.0/3.0)*(asq)*U + 0.5*(y[1]**2))
        
        #Set derivatives
        dydx = N.zeros((5,len(self.k)))
        
        #d\phi_0/d\eta = y_1
        dydx[0] = y[1] 
        
        #d^2phi/d\eta^2 = -2Hphi^prime -a^2U_,phi
        dydx[1] = -2*H*y[1] - asq*dUdphi
        
        #dy_2/d\eta = \delta\phi_1^prime
        dydx[2] = y[3]
        
        #delta\phi^prime^prime
        dydx[3] = -(self.k**2)*y[2] - asq*d2Udphi2*y[2]
        
        #da/d\eta = [1/3 a^2 U_0]^{1/2}*a
        dydx[4] = H*y[4]
        
        return dydx