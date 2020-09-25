# encoding: utf8
from __future__ import unicode_literals
import numpy as np
import random
import matplotlib.mlab as mlab


cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)


cdef class GP:
    cdef double beta
    cdef int ns
    cdef xt, yt
    cdef double[:] param
    cdef dict param_cache
    cdef indpoints
    cdef int M
    cdef double sig2
    cdef double[:,:] S
    cdef double[:,:] Kmn

    cdef double covariance_func(self, double xi, double xj):
        cdef double theta0 = 1.0
        cdef double theta1 = 1.0
        cdef double theta2 = 0
        cdef double theta3 = 16.0
        return theta0 * exp(-0.5 * theta1 * (xi - xj) * (xi - xj)) + theta2 + theta3 * xi * xj

    cdef double normpdf(self, double x, double mu, double sigma):
        return 1./(sqrt(2*np.pi)*sigma)*exp(-0.5 * ((x - mu)/sigma)**2)


    def __init__( self, indpoints ):
        self.beta = 10.0
        self.param_cache = {}

        self.indpoints = indpoints
        self.M = len(self.indpoints)
        self.sig2 = 1.0
        self.ns = 0

    def learn(self, xt, yt ):
        cdef int i,j
        self.xt = xt
        self.yt = yt
        self.ns = len(xt)

        cdef double[:,:] Kmm = np.zeros((self.M, self.M))
        cdef double[:,:] Knm = np.zeros((self.ns, self.M))

        ## K
        for i in range(self.M):
            for j in range(self.M):
                Kmm[i,j] = self.covariance_func(self.indpoints[i], self.indpoints[j])
                if i==j:
                    Kmm[i,j] += 1/self.beta
            for ii in range(self.ns):
                Knm[ii, i] = self.covariance_func(self.xt[ii], self.indpoints[i])

        self.Kmn = Knm.T
        self.S = np.linalg.inv( Kmm + 1/self.sig2 * np.dot(self.Kmn, Knm) )

        self.param_cache.clear()


    cpdef double calc_lik( self, double[:] xs, double[:] ys ):
        cdef int k,i
        cdef int n = len(xs)
        cdef double lik = 0
        cdef double c,p,mu,sigma
        cdef double[:] kmx= np.zeros(self.M)

        if self.ns == 0:
          p_ = 0.000000000001
          for k in range(n):
            lik += log( p_ )
          return lik

        S_ = self.S[0:self.M][0:self.M]
        Kmn_ = self.Kmn[0:self.M][0:self.ns]
        mus = np.zeros((n))
        sigmas = np.zeros((n))

        for k in range(n):
            # 計算結果をキャッシュして使い回す
            if xs[k] in self.param_cache:
                mu, sigma = self.param_cache[ xs[k] ]
            else:
                kxm = np.zeros((self.M))
                for i in range(self.M):
                    kxm[i] = self.covariance_func(xs[k], self.indpoints[i])
                kmx = kxm.T
                sigma = np.dot(np.dot( kxm, S_ ), kmx )
                mu = 1/self.sig2 * np.dot( np.dot( np.dot(kxm, S_ ), Kmn_), self.yt )

                self.param_cache[ xs[k] ] = (mu, sigma)

            mus[k] = mu
            sigmas[k] = sigma
            p = self.normpdf( ys[k] , mu, sigma )
            if p<=0:
                p = 0.000000000001
            lik += log( p )

        return lik


if __name__=='__main__':
    pass
