#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import random
import matplotlib.mlab as mlab
import sys
import math

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

cdef _logsumexp( double a, double b ):
    cdef double max_val
    cdef double sum_exp

    if a>b:
        max_val = a
    else:
        max_val = b

    sum_exp = exp(a-max_val)+ exp(b-max_val)
    return log(sum_exp) + max_val
    
cpdef calc_forward_probability(double[:,:,:] emission_prob_all, double[:,:] trans_prob, double[:] trans_prob_bos, double[:] trans_prob_eos, int T, int MIN_LEN, int SKIP_LEN, int MAX_LEN, int num_class):
    cdef int t, k, c, tt, kk, cc
    cdef double foward_prob
    cdef double[:,:,:] log_a = np.zeros( (T, MAX_LEN, num_class) )  - 99999999999999
    cdef double[:,:,:] valid = np.zeros( (T, MAX_LEN, num_class) ) # 計算された有効な値可どうか．計算されていない場所の確率を0にするため．
    cdef double[:] z = np.ones( T ) # 正規化定数
    cdef double[:,:] m = np.zeros( (T, num_class) )  # t-kの計算結果を入れるバッファ

    for t in range(T):
        for k in range(MIN_LEN, MAX_LEN, SKIP_LEN):
            if t-k<0:
                break

            for c in range(num_class):
                out_prob = emission_prob_all[c,k,t-k] 
                foward_prob = 0.0

                # 遷移確率
                tt = t-k-1
                if tt>=0:
                    #if m[tt]==0:
                    #    m[tt] = logsumexp( log_a[tt,:,:] + z[tt] + np.log(trans_prob[:,c]) )
                    #foward_prob = m[tt] + out_prob

                    if m[tt,c]==0:
                        s = -999999999999
                        for kk in range(MAX_LEN):
                            for cc in range(num_class):
                                s = _logsumexp( s, log_a[tt,kk,cc] +  z[tt] + log(trans_prob[cc, c]))
                        m[tt,c] = s
                    foward_prob = m[tt,c] + out_prob
                else:
                    # 最初の単語
                    foward_prob = out_prob + log(trans_prob_bos[c])

                if t==T-1:
                    # 最後の単語
                    foward_prob += log(trans_prob_eos[c])

                # 正規化を元に戻す
                log_a[t,k,c] = foward_prob
                valid[t,k,c] = 1.0
                if math.isnan(foward_prob):
                    print( "a[t=%d,k=%d,c=%d] became NAN!!" % (t,k,c) )
                    sys.exit(-1)
        # 正規化
        if t-MIN_LEN>=0:
            z[t] = logsumexp( log_a[t,:,:] )
            #log_a[t,:,:] -= z[t]
            for k in range(MAX_LEN):
                for c in range(num_class):
                    log_a[t, k, c] -= z[t]

    return np.exp(log_a)*valid
