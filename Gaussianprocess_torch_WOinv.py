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
  def __init__(self, dim, theta=16.0, device="cuda" ):
    self.beta = 10.0
    self.dim = dim
    self.device = device
    self.theta = theta
    self.param_cache = {}
    self.N = 0
    #self.c_time = 0

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

    # カーネル行列を定義
    self.K = self.cov( self.xt, self.xt ) + torch.eye(self.N, self.N)/self.beta
    #self.K_inv = torch.inverse( self.K )
    self.param_cache.clear()

  def predict( self, x ):
    x = torch.tensor(x).reshape(-1,self.dim)

    kx = self.cov( x, self.xt )
    k = self.cov( x, x) + 1.0/self.beta

    #mu = torch.mm( torch.mm( kx, self.K_inv ), self.yt.reshape(-1,1) )
    #sig = k - torch.mm( kx, torch.mm(self.K_inv, torch.t(kx)) )

    #t_ = time.time()
    #mus = mu.detach().numpy().flatten()
    #sigs = sig.diag().detach().numpy().flatten()
    #self.c_time += time.time() -t_

    # K a = yを満たすa（ = K^-1 y）を逆行列を使わずに解く
    a, _ = torch.solve(self.yt.reshape(-1,1) , self.K)
    mu = torch.mm( kx, a )

    # K b = kx^Tを満たすb（= K^-1 kx^T）を逆行列を使わずに解く
    b, _ = torch.solve( torch.t(kx), self.K)
    sig = k - torch.mm( kx, b )

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
