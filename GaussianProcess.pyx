# encoding: utf8
from __future__ import unicode_literals
import numpy as np
import random
import matplotlib.mlab as mlab
import cython
import math


cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)


cdef class GP:
    cdef int ns
    cdef double[:] xt, yt
    cdef double[:,:] i_cov
    cdef double[:] param
    cdef dict param_cache

    cdef double beta
    cdef double theta0
    cdef double theta1
    cdef double theta2
    cdef double theta3

    cdef double covariance_func(self, double xi, double xj):
        return self.theta0 * exp(-0.5 * self.theta1 * (xi - xj) * (xi - xj)) + self.theta2 + self.theta3 * xi * xj

    cdef double normpdf(self, double x, double mu, double sigma):
        return 1./(sqrt(2*np.pi)*sigma)*exp(-0.5 * ((x - mu)/sigma)**2)

    def __init__( self ):
        self.param_cache = {}

        self.beta = 10.0
        self.theta0 = 1.0
        self.theta1 = 1.0
        self.theta2 = 0
        self.theta3 = 16.0


    cpdef learn(self, double[:] xt, double[:] yt ):
        cdef int i,j
        cdef double c
        self.xt = np.array( xt )
        self.yt = np.array( yt )
        self.ns = len(xt)
        cdef double[:,:] cov = np.zeros((self.ns, self.ns))

        for i in range(self.ns):
            for j in range(i+1):
                c = self.covariance_func(xt[i], xt[j])
                cov[i,j] = c
                cov[j,i] = c
                if i==j:
                    cov[i,j] += 1/self.beta


        self.i_cov = np.linalg.inv(cov)
        self.param = np.dot(self.i_cov, self.yt)
        
        self.param_cache.clear()


    cpdef predict( self, double[:] x ):
        cdef int k, i
        cdef double mu, sigma, c
        cdef int N = x.shape[0]
        cdef double[:] v
        cdef double[:] tt = self.yt #[y - np.random.normal() / self.beta for y in self.yt]
        cdef double[:] mus = np.zeros( N )
        cdef double[:] sigmas = np.zeros(N)

        for k in range(N):
            v = np.zeros((self.ns))
            for i in range(self.ns):
                v[i] = self.covariance_func(x[k], self.xt[i])
            c = self.covariance_func(x[k], x[k]) + 1.0 / self.beta
            
            mus[k] = np.dot(v, np.dot(self.i_cov, tt))
            sigmas[k] = c - np.dot(v, np.dot(self.i_cov, v))
        
        return mus, sigmas

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
    
    cpdef estimate_hyperparams(self, int niter, double step=0.2):
        cdef int itr, d
        cdef double init_lik = self.calc_lik( self.xt, self.yt )
        cdef double max_lik = init_lik
        cdef double old_lik = init_lik

        max_params = [self.beta, self.theta0, self.theta1, self.theta2, self.theta3]
        new_params = [self.beta, self.theta0, self.theta1, self.theta2, self.theta3]
        
        for itr in range(niter):
            
            for d in range(5):
                old_params = [self.beta, self.theta0, self.theta1, self.theta2, self.theta3]
                
                # 対数空間でランダムに動かす
                # p' = exp( log(p) + rand )
                new_params[d] = old_params[d] * exp (step * random.gauss(0,1))
                
                # 新しいパラメータで学習・尤度計算
                self.beta, self.theta0, self.theta1, self.theta2, self.theta3 = new_params
                self.learn(self.xt, self.yt)
                new_lik = self.calc_lik(self.xt, self.yt)


                # accept or reject
                if math.exp(new_lik-old_lik)>random.random():
                    # acceptの場合は更新
                    old_lik = new_lik
                else:
                    # rejectの場合は元に戻す
                    self.beta, self.theta0, self.theta1, self.theta2, self.theta3 = old_params
                
                # 最大のものを保存する
                if max_lik<old_lik:
                    max_lik = old_lik
                    max_params = [self.beta, self.theta0, self.theta1, self.theta2, self.theta3]

        self.beta, self.theta0, self.theta1, self.theta2, self.theta3 = max_params
        # print(init_lik, "->", max_lik)
        return
        



if __name__=='__main__':
    pass