# -*- coding: utf-8 -*-
"""helpers.py - Helper functions

Provides helper functions for use by the package elsewhere.

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.



from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as np
import re
from scipy import integrate
import os.path
import logging

def nanfillstart(a, l):
    """Return an array of length l by appending array a 
    to end of block of NaNs along axis 0.
    
    Parameters
    ----------
    a: numpy array
       array to append to end of block of NaNs.
       
    l: integer
       length of the new array to return.
       
    Returns
    -------
    c: numpy_array
       array of length l with initial values filled with NaNs
       and array a at the end.
    """
    if len(a) >= l:
        return a #Array already as long or longer than required
    else:
        bshape = np.array(a.shape)
        bshape[0] = l - bshape[0]
        b = np.ones(bshape)*np.NaN
        c = np.concatenate((b,a))
        return c

def getkend(kinit, deltak, numsoks):
    """Correct kend value given the values of kinit, deltak and numsoks.
    
    Parameters
    ----------
    kinit: float
           Initial k value in range.
    
    deltak: float
            Difference between two k values in range.
            
    numsoks: integer
             The number of second order k modes required.
             
    Returns
    -------
    kend: float
          The value of the end of the first order k range required so that
          the second order k range has numsoks modes in it.
          
    """
    #Change from numsoks-1 to numsoks to include extra point when deltak!=kinit
    return 2*((numsoks)*deltak + kinit)

def eto10(number):
    """Convert scientific notation e.g. 1e-5 to 1x10^{-5} for use in LaTeX, converting to string."""
    s = re.sub(r'e(\S)0?(\d+)', r'\\times 10^{\1\2}', str(number))
    return s

def klegend(ks,mpc=False):
    """Return list of string representations of k modes for legend."""
    klist = []
    for k in ks:
        if mpc:
            str = r"$k=" + eto10(k) + r"M_{\mathrm{PL}} = " + eto10(mpl2invmpc(k)) + r" M\mathrm{pc}$"
        else:
            str = r"$k=" + eto10(k) + r"M_{\mathrm{PL}}$"
        klist.append(str)
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

    bin_n = np.binary_repr(n)[1:]
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
    retlist = np.array([])
    for x in l:
        if x not in retlist:
            retlist = np.append(retlist, x)
    return retlist

def getintfunc(x):
    """Return the correct function to integrate with.
    
    Checks the given set of values and returns either scipy.integrate.romb 
    or scipy.integrate.simps. This depends on whether the number of values is a
    power of 2 + 1 as required by romb.
    
    Parameters
    ----------
    x: array_like
       Array of x values to check
    
    Returns
    -------
    intfunc: function object
             Correct integration function depending on length of x.
             
    fnargs: dictionary
            Dictionary of arguments to integration function.
    """
    if ispower2(len(x)-1):
        intfunc = integrate.romb
        fnargs = {"dx":x[1]-x[0]}
    elif len(x) > 0:
        intfunc = integrate.simps
        fnargs = {"x":x}
    else:
        raise ValueError("Cannot integrate length 0 array!")
    return intfunc, fnargs

def cartesian_product(lists, previous_elements = []):
    """Generator of cartesian products of lists."""
    if len(lists) == 1:
        for elem in lists[0]:
            yield previous_elements + [elem, ]
    else:
        for elem in lists[0]:
            for x in cartesian_product(lists[1:], previous_elements + [elem, ]):
                yield x

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def ensurepath(path):
    """Check that the path for given directory exists and create it if not."""
    
    #Get absolute path
    path = os.path.abspath(path)
    #Does path exist?
    if not os.path.isdir(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            raise OSError("Error creating results directory!")
        
def seq(min=0.0, max=None, inc=1.0, type=float,
        return_type='NumPyArray'):
    """
    Generate numbers from min to (and including!) max,
    with increment of inc. Safe alternative to arange.
    The return_type string governs the type of the returned
    sequence of numbers ('NumPyArray', 'list', or 'tuple').
    """
    if max is None: # allow sequence(3) to be 0., 1., 2., 3.
        # take 1st arg as max, min as 0, and inc=1
        max = min; min = 0.0; inc = 1.0
    r = np.arange(min, max + inc/2.0, inc, type)
    if return_type == 'NumPyArray' or return_type == np.ndarray:
        return r
    elif return_type == 'list':
        return r.tolist()
    elif return_type == 'tuple':
        return tuple(r.tolist())
    return

def find_nearest(array,value):
    """
    Find the index of the number in `array` which is nearest to `value`.
    """
    idx=(np.abs(array-value)).argmin()
    return array[idx]

def find_nearest_ix(array,value):
    """
    Find the index of the number in `array` which is nearest to `value`.
    """
    return (np.abs(array-value)).argmin()

def startlogging(log, logfile, loglevel=logging.INFO, consolelevel=None):
    """Start the logging system to store rotational file based log."""
    
    try:
        from cloghandler import ConcurrentRotatingFileHandler as RFHandler
    except ImportError:
    # Next 2 lines are optional:  issue a warning to the user
        from warnings import warn
        warn("ConcurrentLogHandler package not installed.  Using builtin log handler")
        from logging.handlers import RotatingFileHandler as RFHandler

    
    if not consolelevel:
        consolelevel = loglevel

    log.setLevel(loglevel)
    #create file handler and set level to debug
    fh = RFHandler(filename=logfile, maxBytes=2**20, backupCount=50)
    fh.setLevel(loglevel)
    #create console handler and set level to error
    ch = logging.StreamHandler()
    ch.setLevel(consolelevel)
    #create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    #add formatter to fh
    fh.setFormatter(formatter)
    #add formatter to ch
    ch.setFormatter(formatter)
    #add fh to logger
    log.addHandler(fh)
    #add ch to logger
    log.addHandler(ch)
    log.debug("Logging started at level %d", loglevel)
    return log


