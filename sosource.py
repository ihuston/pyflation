"""Second order helper functions to set up source term"""

def getsourceterm(self):
    """Return source term (slow-roll for now), once first order system has been executed."""
    #Make sure first order is complete
    self.checkfirstordercomplete()
    
    #Get first order results (from file or variables)
    phi, phidot, H, dphi1real, dphi1dotreal, dphi1imag, dphi1dotimag = self.foyresult[:,i,:] for i in range(7)
    dphi1 = dphi1real + dphi1imag*1j
    dphi1dot = dphi1dotreal + dphi1dotimag*1j
    
    #Initialize variables to store result
    s2shape = (len(self.k),len(self.k)
    source = []
    #Main loop over each time step
    for nix, n in enumerate(self.fotresult):
        #Single time step
        a = self.ainit*exp(n)
        
        #Initialize result variable for k modes
        s2 = N.empty(s2shape)
        for kix, k in enumerate(self.k):
            #Single k mode
            
            #Result variable for source
            s1 = N.empty_like(self.k)
            for qix, q in enumerate(self.k):
                #Single q mode
                #First major term:
                term1 = 1/(2*N.pi**2) * (potentialterms) * q**2 *dphi1dot[t,N.abs(qix-kix+1)]
                s1[qix] = term1 + term2 + term3
            #add sourceterm for each q
            s2[kix] = s1
        #save results for each q
        source.append(s2)