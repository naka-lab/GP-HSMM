import numpy as np
import matplotlib.pyplot as plt

"""
sor GP
no cacheVer
"""


class SORGP:
  def __init__(self, denom, dim ):
    self.D = 1
    self.param_cache = {}
    self.denom = denom

  def k(self, xi, xj):
    return 1.0 * np.exp(-0.5 * 1.0 * np.sum((xi - xj) * (xi - xj), 2))

  def cov(self, xi, xj ):
    a = np.tile( xi.reshape(-1,1,self.D), (1, len(xj), 1) )
    b = np.tile( xj.reshape(1,-1,self.D), (len(xi), 1, 1) )
    #return self.k(a,b) + 16.0 * np.multiply(xi, xj.T)
    return self.k(a,b)

  def learn(self, xt, yt ):
    max_xt = np.max(xt)
    self.learn_inducing_points = np.arange(max_xt)[::self.denom]
    self.M = len( self.learn_inducing_points)

    self.xt = np.array(xt)
    self.yt = np.array(yt)
    self.N = len(xt)

    self.sig2 = 1.0

    # kernel
    self.Kmm = self.cov( self.learn_inducing_points, self.learn_inducing_points )
    self.Kmm_inv = np.linalg.inv( self.Kmm+np.eye(self.M, self.M) )
    self.Knm = self.cov( self.xt, self.learn_inducing_points )
    self.Kmn = self.Knm.T
    #self.Knn = self.cov( self.xt, self.xt )
    #p.150
    self.Knn = self.cov( self.xt, self.xt ) * np.eye(self.N, self.N)
    self.Knn_ = np.dot( np.dot(self.Knm, self.Kmm_inv), self.Kmn )

    # Sigma (original)
    self.S = np.linalg.inv( self.Kmm + 1/self.sig2 * np.dot(self.Kmn, self.Knm))
    # Sigma
    #self.S = np.linalg.inv( self.Kmm + 1/self.sig2 * np.dot(self.Kmn, self.Knm) + np.eye(self.M, self.M))


  def plot(self, x, y=False):
      max_x = np.max(x)
      inducing_points = np.arange(max_x)[::self.denom]

      mus, sigmas, lik = self.predict( x.reshape(-1,1), y)
      plt.plot( x, mus )

      y_max = mus + np.sqrt(sigmas.flatten())
      y_min = mus - np.sqrt(sigmas.flatten())

      plt.fill_between(x, y_min, y_max, facecolor="lavender" , alpha=0.9 , edgecolor="lavender"  )

      for p in inducing_points:
        plt.plot( p, [0.0], "kx" )
      plt.plot(self.xt, self.yt)
      plt.show()

      print ("lik",lik)

  def normpdf(self, y, mu, sigma):
        return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5 * ((y - mu)/sigma)**2)

  def predict( self, x, y ):
    x = np.array(x)
    mus = []
    sigmas = []
    K = len(x)

    Kxm = self.cov( x.reshape(-1,1), self.learn_inducing_points )
    Kmx = Kxm.T

    sig = np.dot(np.dot( Kxm, self.S ), Kmx )
    mu = 1/self.sig2 * np.dot( np.dot( np.dot(Kxm, self.S ), self.Kmn), self.yt.reshape(-1,1) )

    p = self.normpdf(y, mu.flatten(), np.diag(sig).flatten())

    p[p <= 0] = 0.000000000001

    p_ = p.sum()
    lik = np.log(p_)

    return mu.flatten(), np.diag(sig).flatten(), lik
