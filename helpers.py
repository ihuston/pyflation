"""Helper functions by Ian Huston
    $Id: helpers.py,v 1.3 2008/08/06 17:17:47 ith Exp $
    
    Provides helper functions for use elsewhere"""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N

def nanfillstart(a, l):
    """Return an array of length l by appending array a to end of block of NaNs along axis 0."""
    if len(a) >= l:
        return a #Array already as long or longer than required
    else:
        bshape = N.array(a.shape)
        bshape[0] = l - bshape[0]
        b = N.ones(bshape)*N.NaN
        c = N.concatenate((b,a))
        return c

def eto10(number):
    """Convert scientific notation e.g. 1e-5 to 1x10^{-5} for use in LaTeX, converting to string."""
    s = str(number)
    s = s.replace("e", r"\times10^{")
    s = s + "}"
    return s 