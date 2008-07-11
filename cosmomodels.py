"""Cosmological Model simulations by Ian Huston
    $Id: cosmomodels.py,v 1.55 2008/07/11 14:07:32 ith Exp $
    
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
        
        if tstart < tend: 
            self.tstart, self.tend = tstart, tend
        elif tstart==tend:
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
        
    def derivs(self, t, yarray):
        """Return an array of derivatives of the dependent variables yarray at timestep t"""
        pass
    
    def potentials(self, y):
        """Return a 3-tuple of potential, 1st and 2nd derivs given y."""
        pass
    
    def findH(self,potential,y):
        """Return value of comoving Hubble variable given potential and y."""
        pass
    
    def run(self):
        """Execute a simulation run using the parameters already provided."""
        
        if self.solver not in self.solverlist:
            raise ModelError("Unknown solver!")
        #Test whether k exists and if so change init conditions
        
        if self.k is not None and (self.solver == "odeint" or self.solver == "rkdriver_dumb"):
            #Make the initial conditions the right shape
            
            if type(self.k) is N.ndarray or type(self.k) is list: 
                self.ystart = self.ystart.reshape((len(self.ystart),1))*N.ones((len(self.ystart),len(self.k)))
                        
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
            swap_derivs = lambda y, t : self.derivs(t,y)
            times = N.arange(self.tstart, self.tend, self.tstep_wanted)
            
            #Deal with complex init conditions.
            if self.ystart.dtype == complex and self.decoupled == False:
                raise ModelError("Cannot run coupled complex model in scipy_odeint!")
            #Now split depending on whether k exists
            if type(self.k) is N.ndarray or type(self.k) is list:
                #Make a copy of k while we work
                klist = N.copy(self.k)
                
                #Do calculation in both real and complex case
                #Compute list of ks in a row
                ylist = [scipy_odeint(swap_derivs, self.ystart.real, times) for self.k in klist] 
                if self.ystart.dtype == complex:
                    ycompl = [scipy_odeint(swap_derivs, self.ystart.imag, times) for self.k in klist]
                    ylist = N.array(ylist) + N.array(ycompl)*1j
                #Now stack results to look like as normal (time,variable,k)
                self.yresult = N.dstack(ylist)
                self.tresult = times
                #Return klist to normal
                self.k = klist
            else:
                self.yresult = scipy_odeint(swap_derivs, self.ystart, times)
                self.tresult = times
            
        #Aggregrate results and calling parameters into results list
        self.lastparams = self.callingparams()
        
        self.resultlist.append([self.lastparams, self.tresult, self.yresult])
        
        self.runcount += 1
        
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
                  "CVSRevision":"$Revision: 1.55 $",
                  "datetime":datetime.datetime.now()
                  }
        return params
               
    def argstring(self):
        a = r"; Arguments: ystart="+ str(self.ystart) + r", tstart=" + str(self.tstart) 
        a += r", tend=" + str(self.tend) + r", mass=" + str(self.mass)
        return a
    
    def plotresults(self, fig=None, show=True, varindex=None, klist=None, saveplot=False):
        """Plot results of simulation run on a graph.
            Return figure instance used."""
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!")
        
        if varindex is None:
            varindex = [0] #Set default list of variables to plot
        
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
        P.legend(N.array(self.ynames)[varindex])
        #P.title(self.plottitle, figure=fig)
        
        #Should we show it now or just return it without showing?
        if show:
            P.show()
        #Should we save the plot somewhere?
        if saveplot:
            time = self.lastparams["datetime"].strftime("%Y%m%d%H%M%S")
            filename = "./graphs/run" + time + ".png"
                
            if os.path.isdir(os.path.dirname(filename)):
                if os.path.isfile(filename):
                    raise IOError("File already exists!")
            else:
                raise IOError("Directory 'graphs' does not exist")
            try:
                f.savefig(filename)
                print "Plot saved as " + filename
            except IOError:
                raise
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
            time = self.lastparams["datetime"].strftime("%Y%m%d%H%M%S")
            filename = "./graphs/run" + time + ".png"
                
            if os.path.isdir(os.path.dirname(filename)):
                if os.path.isfile(filename):
                    raise IOError("File already exists!")
            else:
                raise IOError("Directory 'graphs' does not exist")
            try:
                f.savefig(filename)
                print "Plot saved as " + filename
            except IOError:
                raise
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
    
    def derivs(self, t, y):
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
    
    def derivs(self, t, y):
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
        endefold = self.tresult[endindex]
        
        return endefold, endindex
    
    def getepsilon(self):
        """Return an array of epsilon = -\dot{H}/H values for each timestep."""
        if self.runcount == 0:
            raise ModelError("Model has not been run yet, cannot plot results!")

        #Find Hdot
        if self.k is not None:
            Hdot = N.array(map(self.derivs, self.tresult, self.yresult))[:,2,0]
            epsilon = - Hdot/self.yresult[:,2,0]
        else:
            Hdot = N.array(map(self.derivs, self.tresult, self.yresult))[:,2]
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
    
      
    
    def derivs(self, t, y):
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
        CosmologicalModel.plotresults(self, fig=f, show=False, varindex=[0,1], saveplot=False)
        
        #Second plot of H
        P.subplot(122)
        CosmologicalModel.plotresults(self, fig=f, show=False, varindex=[2], saveplot=False)
        
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
        
        #Are the complex equations decoupled?
        self.decoupled = True
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart = N.array([15.0,-0.1,0.0,0.1,0.1])   
        
        #Set initial H value if None
        if self.ystart[2] == 0.0:
            U = self.potentials(self.ystart)[0]
            self.ystart[2] = self.findH(U, self.ystart)
            
        #Text for graphs
        self.plottitle = "First Order Model in Efold time"
        self.tname = r"$n$"
        self.ynames = [r"$\varphi_0$", 
                        r"$\dot{\varphi_0}$",
                        r"$H$",
                        r"$\delta\varphi_1$",
                        r"$\dot{\delta\varphi_1}$"]
                    
    def derivs(self, t, y):
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
        dydx[4] = (-(3 + dydx[2]/y[2])*y[4] - ((self.k/(a*y[2]))**2)*y[3] 
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[3]/(y[2]**2)))
        #print dydx[4]
        return dydx       
        
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
                  "CVSRevision":"$Revision: 1.55 $",
                  "datetime":datetime.datetime.now()
                  }
        return params
    
    def derivs(self, t, y):
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
    
    def derivs(self, t, y):
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
            
    def derivs(self, t, y):
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