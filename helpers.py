"""Helper functions by Ian Huston
    $Id: helpers.py,v 1.10 2009/02/09 13:31:23 ith Exp $
    
    Provides helper functions for use elsewhere"""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
import re

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
    s = re.sub(r'e(\S\d+)', r'\\times 10^{\1}', str(number))
    return s

def klegend(ks):
    """Return list of string representations of k modes for legend."""
    klist = [r"$k=" + eto10(k) + r"M_{\mathrm{PL}} = " + eto10(mpl2invmpc(k)) + r" M\mathrm{pc}$" for k in ks]
    return klist

def invmpc2mpl(x=1):
    """Convert from Mpc^-1 to Mpl (reduced Planck Mass)"""
    return 2.625e-57*x


def mpl2invmpc(x=1):
    """Convert from Mpl (reduced Planck Mass) to Mpc^-1"""
    return 3.8095e+56*x

def ispower2(n):
    """Returns the log base 2 of n if n is a power of 2, zero otherwise.

    Note the potential ambiguity if n==1: 2**0==1, interpret accordingly."""

    bin_n = N.binary_repr(n)[1:]
    if '1' in bin_n:
        return 0
    else:
        return len(bin_n)

def removedups(l):
    """Return an array with duplicates removed but order retained. 
    
    An array is returned no matter what the input type. The first of each duplicate is retained.
    
    Parameters
    ----------
    l: array_like
       Array (or list etc.) of values with duplicates that are to be removed.
       
    Returns
    -------
    retlist: ndarray
             Array of values with duplicates removed but order intact.
    """
    retlist = N.array([])
    for x in l:
        if x not in retlist:
            retlist = N.append(retlist, x)
    return retlist
