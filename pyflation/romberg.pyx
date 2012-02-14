"""romberg.pyx - Romb integration for samples from two dimensional complex arrays.

Adapted by Ian Huston. 
Includes code modified from scipy.integrate.romb released under BSD license:

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2009 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Enthought nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

"""

from numpy import add, isscalar, asarray

import numpy as N
cimport numpy as N
cimport cython

DTYPED = N.float64
DTYPEI = N.int
ctypedef N.double_t DTYPED_t
ctypedef N.int_t DTYPEI_t
DTYPEC = N.complex128
ctypedef N.complex128_t DTYPEC_t




# Modified from scipy.integrate.romb released under BSD license:
#
#Copyright (c) 2001, 2002 Enthought, Inc.
#All rights reserved.
#
#Copyright (c) 2003-2009 SciPy Developers.
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#  a. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  b. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#  c. Neither the name of the Enthought nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#DAMAGE.


def romb(N.ndarray[DTYPEC_t, ndim=2] y, DTYPED_t dx=1.0):
    """Romberg integration of two dimensional complex array
    
    Modified from scipy.integrate.romb
    """
#    y = asarray(y)
    cdef DTYPEI_t Nsamps = y.shape[1]
    cdef DTYPEI_t Ninterv = Nsamps-1
    cdef DTYPEI_t n = 1
    cdef DTYPEI_t k = 0
    cdef int i, j
    cdef DTYPEI_t start, stop, step
    
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError, \
              "Number of samples must be one plus a non-negative power of 2."

    R = {}
    h = Ninterv*asarray(dx)*1.0
    R[(1,1)] = (y[:,0] + y[:,-1])/2.0*h
    start = stop = step = Ninterv
    for i in range(2,k+1):
        start >>= 1
        R[(i,1)] = 0.5*(R[(i-1,1)] + h*add.reduce(y[:,start:stop:step],1))
        step >>= 1
        for j in range(2,i+1):
            R[(i,j)] = R[(i,j-1)] + \
                       (R[(i,j-1)]-R[(i-1,j-1)]) / ((1 << (2*(j-1)))-1)
        h = h / 2.0

    return R[(k,k)]
