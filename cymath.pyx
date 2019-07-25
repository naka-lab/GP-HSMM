#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import random
import matplotlib.mlab as mlab
import sys

cdef extern from "math.h":
    double log(double)
    double exp(double)


cpdef logsumexp( double[:,:] a ):
    cdef double max_val = -sys.float_info.max
    cdef double sum_exp = 0
    cdef int I = a.shape[0]
    cdef int J = a.shape[1]
    
    for i in range(I):
        for j in range(J):
            if max_val<a[i,j]:
                max_val = a[i,j]
                
    for i in range(I):
        for j in range(J):
            sum_exp += exp( a[i,j] - max_val )
    return log(sum_exp) + max_val
    