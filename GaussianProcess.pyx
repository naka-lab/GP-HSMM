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
    cdef double[:,:] i_cov
    cdef double[:] param
    cdef dict param_cache

    cdef double covariance_func(self, double xi, double xj):
        cdef double theta0 = 1.0
        cdef double theta1 = 1.0
        cdef double theta2 = 0
        cdef double theta3 = 16.0
        return theta0 * exp(-0.5 * theta1 * (xi - xj) * (xi - xj)) + theta2 + theta3 * xi * xj

    cdef double normpdf(self, double x, double mu, double sigma):
        return 1./(sqrt(2*np.pi)*sigma)*exp(-0.5 * ((x - mu)/sigma)**2)


    def __init__( self ):
        self.beta = 10.0
        self.param_cache = {}

    def learn(self, xt, yt ):
        cdef int i,j
        self.xt = xt
        self.yt = yt
        self.ns = len(xt)
        # construct covariance
        cdef double[:,:] cov = np.zeros((self.ns, self.ns))

        for i in range(self.ns):
            for j in range(self.ns):
                cov[i,j] = self.covariance_func(xt[i], xt[j])
                if i==j:
                    cov[i,j] += 1/self.beta


        self.i_cov = np.linalg.inv(cov)
        self.param = np.dot(self.i_cov, self.yt)
        
        self.param_cache.clear()


    def predict( self, x ):
        mus = []
        sigmas = []
        n = len(x)
        tt = [y - np.random.normal() / self.beta for y in self.yt]
        for k in range(n):
            v = np.zeros((self.ns))
            for i in range(self.ns):
                v[i] = self.covariance_func(x[k], self.xt[i])
            c = self.covariance_func(x[k], x[k]) + 1.0 / self.beta
            
            mu = np.dot(v, np.dot(self.i_cov, tt))
            sigma = c - np.dot(v, np.dot(self.i_cov, v))
            
            mus.append(mu)
            sigmas.append(sigma)
        
        return np.array(mus), np.array(sigmas)


    cpdef double calc_lik( self, double[:] xs, double[:] ys ):
        cdef int k,i
        cdef int n = len(xs)
        cdef double lik = 0
        cdef int ns = self.ns
        cdef double c,p,mu,sigma
        cdef double[:] v= np.zeros((ns))

        for k in range(n):
            # 計算結果をキャッシュして使い回す
            if xs[k] in self.param_cache:
                mu, sigma = self.param_cache[ xs[k] ]
            else:
                v = np.zeros((ns))
                for i in range(ns):
                    v[i] = self.covariance_func(xs[k], self.xt[i])
                c = self.covariance_func(xs[k], xs[k]) + 1.0 / self.beta
                mu = np.dot(v, self.param)
                sigma = c - np.dot(v, np.dot(self.i_cov, v))
                
                self.param_cache[ xs[k] ] = (mu, sigma)

            p = self.normpdf( ys[k] , mu, sigma )
            if p<=0:
                p = 0.000000000001
            lik += log( p )

        return lik



if __name__=='__main__':
    pass