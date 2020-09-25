# encoding: utf8
from __future__ import unicode_literals
from __future__ import print_function
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import GaussianProcess
import matplotlib.pyplot as plt
import time


class GPMD:
    def __init__(self, dim, indpoints):
        self.__dim = dim
        self.indpoints = indpoints
        self.__gp = [ GaussianProcess.GP(self.indpoints) for d in range(self.__dim) ]
        self.b_time = 0

    def learn(self,x, y ):
        y = np.array(y, dtype=np.float).reshape( (-1,self.__dim) )
        x = np.array(x,dtype=np.float)

        for d in range(self.__dim):
            if len(y)!=0:
                self.__gp[d].learn( x, y[:,d] )
            else:
                #self.__gp[d].learn( x, [] )
                self.__gp[d] = GaussianProcess.GP(self.indpoints)


    def calc_lik(self, x, y ):
        s_time = time.time()
        lik = 0.0

        if self.__dim==1:
            y = np.asarray(y, dtype=np.float).reshape( (-1,self.__dim) )
        #x = np.asarray(x,dtype=np.float)
        for d in range(self.__dim):
            lik += self.__gp[d].calc_lik( x , y[:, d] )
        self.b_time += time.time() - s_time
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


def main():
    pass

if __name__ == '__main__':
    main()
