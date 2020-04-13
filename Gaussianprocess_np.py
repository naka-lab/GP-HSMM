import numpy as np
import matplotlib.pyplot as plt

class SORGP:
  def __init__(self, inducing_points, dim ):
    self.M = len(inducing_points)
    self.D = dim
    self.inducing_points = inducing_points

  def k(self, xi, xj):
    return 1.0 * np.exp(-0.5 * 1.0 * np.sum((xi - xj) * (xi - xj), 2))

  def cov(self, xi, xj ):
    a = np.tile( xi.reshape(-1,1,self.D), (1, len(xj), 1) )
    b = np.tile( xj.reshape(1,-1,self.D), (len(xi), 1, 1) )
    return self.k(a,b)

  def learn(self, xt, yt ):
    self.xt = np.array(xt)
    self.yt = np.array(yt)
    N = len(xt)

    self.sig2 = 1.0

    # カーネル行列を定義
    self.Kmm = self.cov( self.inducing_points, self.inducing_points )
    self.Kmm_inv = np.linalg.inv( self.Kmm+np.eye(self.M, self.M) )
    self.Knm = self.cov( self.xt, self.inducing_points )
    self.Kmn = self.Knm.T
    self.Knn = self.cov( self.xt, self.xt )
    self.Knn_ = np.dot( np.dot(self.Knm, self.Kmm_inv), self.Kmn )

    # Σ
    self.S = np.linalg.inv( self.Kmm + 1/self.sig2 * np.dot(self.Kmn, self.Knm) )

  def plot(self, x):
      mus, sigmas = self.predict( x.reshape(-1,1) )
      plt.plot( x, mus )

      y_max = mus + np.sqrt(sigmas.flatten())
      y_min = mus - np.sqrt(sigmas.flatten())

      plt.fill_between(x, y_min, y_max, facecolor="lavender" , alpha=0.9 , edgecolor="lavender"  )

      for p in self.inducing_points:
        plt.plot( p, [0.0], "kx" )
      plt.plot(self.xt, self.yt)
      plt.show()

  def predict( self, x ):
    x = np.array(x)
    mus = []
    sigmas = []
    K = len(x)

    Kxm = self.cov( x.reshape(-1,1), self.inducing_points )
    Kmx = Kxm.T

    sig = np.dot(np.dot( Kxm, self.S ), Kmx )
    mu = 1/self.sig2 * np.dot( np.dot( np.dot(Kxm, self.S ), self.Kmn), self.yt.reshape(-1,1) )

    return mu.flatten(), np.diag(sig).flatten()
