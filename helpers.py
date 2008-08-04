"""Helper functions by Ian Huston
    $Id: helpers.py,v 1.1 2008/08/04 16:58:41 ith Exp $
    
    Provides helper functions for use elsewhere"""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N

def nanfillstart(a, l):
    """Return an array of length l by appending array a to end of block of NaNs."""
    if len(a) >= l:
        return a #Array already as long or longer than required
    else:
        b = N.ones((l-len(a)))*N.NaN
        c = N.append(b,a)
        return c
