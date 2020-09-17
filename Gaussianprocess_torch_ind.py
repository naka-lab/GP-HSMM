# encoding: utf8
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

"""
sor GP
no cacheVer
"""


class GP:
  def __init__(self, MAX_LEN, dim, theta=16.0, device="cuda" ):
    self.beta = 10.0
    self.dim = dim
    self.device = device
    self.theta = theta
    self.param_cache = {}
    self.N = 0
    #self.c_time = 0

    #Inducing_points 現状等間隔(1つ飛ばし)
    inducing_points = np.arange(MAX_LEN)[::1]
    self.M = len(inducing_points)
    self.inducing_points = torch.tensor( inducing_points  )
    self.sig2 = 1.0


  def k(self, xi, xj):
    return 1.0 * torch.exp(-0.5 * 1.0 * torch.sum((xi - xj) * (xi - xj), 2)) + (self.theta*( xi.view(xi.shape[0], -1) * xj.view(xj.shape[0], -1) ))

  def cov(self, xi, xj ):
    a = xi.view(-1,1,self.dim).repeat(1, len(xj), 1)
    b = xj.view(1,-1,self.dim).repeat(len(xi), 1, 1)
    return self.k(a,b)

  def learn(self, xt, yt ):
    self.xt = torch.tensor(xt).reshape(-1,self.dim)
    self.yt = torch.tensor(yt)
    self.N = len(xt)

    # to gpu
    self.xt.to(self.device)
    self.yt.to(self.device)
    self.inducing_points.to(self.device)

    # カーネル行列を定義
    #self.K = self.cov( self.xt, self.xt ) + torch.eye(self.N, self.N)/self.beta
    #self.K_inv = torch.inverse( self.K )
    self.Kmm = self.cov( self.inducing_points, self.inducing_points )
    self.Kmm_inv = torch.inverse( self.Kmm+torch.eye(self.M, self.M) ).double()
    self.Knm = self.cov( self.xt, self.inducing_points ).double()
    self.Kmn = torch.t( self.Knm ).double()
    self.Knn = self.cov( self.xt, self.xt )
    self.Knn_ = torch.mm( torch.mm(self.Knm, self.Kmm_inv), self.Kmn )

    self.S = torch.inverse( self.Kmm + 1/self.sig2 * torch.mm(self.Kmn, self.Knm) )

    self.param_cache.clear()

  def predict( self, x ):
    x = torch.tensor(x).reshape(-1,self.dim)

    kxm = self.cov( x, self.inducing_points )
    kmx = torch.t( kxm )

    sig = torch.mm(torch.mm( kxm, self.S ), kmx )
    mu = 1/self.sig2 * torch.mm( torch.mm( torch.mm(kxm, self.S ), self.Kmn), self.yt.reshape(-1,self.dim) )

    #t_ = time.time()
    #mus = mu.detach().numpy().flatten()
    #sigs = sig.diag().detach().numpy().flatten()
    #self.c_time += time.time() -t_

    return mu.detach().numpy().flatten(), sig.diag().detach().numpy().flatten()

  def normpdf(self, ys, mu, sigma):
    return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5 * ((ys - mu)/sigma)**2)

  def calc_lik(self, xs, ys):
    n = len(xs)
    lik = 0
    if self.N == 0:
      for j in range(n):
        p = 0.000000000001
        lik += np.log( p )
      return lik

    mu = np.zeros(n)
    sigma = np.zeros(n)

    for j in range(n):
      #cache
      if xs[j] in self.param_cache:
        mu[j], sigma[j] = self.param_cache[ xs[j] ]
      else:
        mu[j], sigma[j] = self.predict(xs[j])
        self.param_cache[xs[j]] = (mu[j], sigma[j])

      p = self.normpdf(ys[j], mu[j], sigma[j])
      if p<=0:
        p = 0.000000000001
      lik += np.log( p )

    return mu.flatten(), np.diag(sigma).flatten(), lik

  #def time_r__(self):
  #    return self.c_time
