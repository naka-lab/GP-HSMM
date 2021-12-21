# encoding: utf8
from __future__ import unicode_literals
from __future__ import print_function
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import GaussianProcess
import matplotlib.pyplot as plt


class GPMD:
    def __init__(self, dim):
        self.__dim = dim
        self.__gp = [ GaussianProcess.GP() for d in range(self.__dim) ]

    def learn(self,x, y, same_cov=True ):
        y = np.array(y, dtype=np.float).reshape( (-1,self.__dim) )
        x = np.array(x,dtype=np.float)
        i_cov = None

        for d in range(self.__dim):
            if not same_cov:
                i_cov = None

            if len(y)!=0:
                i_cov = self.__gp[d].learn( x, y[:,d], i_cov )
            else:
                i_cov = self.__gp[d].learn( x, np.array([]), i_cov )

            


    def calc_lik(self, x, y ):
        lik = 0.0

        if self.__dim==1:
            y = np.asarray(y, dtype=np.float).reshape( (-1,self.__dim) )
        #x = np.asarray(x,dtype=np.float)
        for d in range(self.__dim):
            lik += self.__gp[d].calc_lik( x , y[:,d] )

        return lik

    def plot(self, x ):
        for d in range(self.__dim):
            plt.subplot( self.__dim, 1, d+1 )

            mus, sigmas = self.__gp[d].predict(x)
            y_min = mus - sigmas*2
            y_max = mus + sigmas*2

            plt.fill_between( x, y_min, y_max, facecolor="lavender" , alpha=0.9 , edgecolor="lavender"  )
            plt.plot(x, y_min, 'b--')
            plt.plot(x, mus, 'b-')
            plt.plot(x, y_max, 'b--')

    def predict(self, x ):
        params = []
        for d in range(self.__dim):
            mus, sigmas = self.__gp[d].predict(np.array(x, dtype=np.float))
            params.append( (mus, sigmas) )
        return params

    def estimate_hyperparams(self, niter):
        for d in range(self.__dim):
            self.__gp[d].estimate_hyperparams(niter)


def main():
    pass

if __name__ == '__main__':
    main()