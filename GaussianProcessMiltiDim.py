# encoding: utf8
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import GaussianProcess
import matplotlib.pyplot as plt


class GPMD:
    def __init__(self, dim):
        self.__dim = dim
        self.__gp = [ GaussianProcess.GP() for d in range(self.__dim) ]

    def learn(self,x, y ):
        y = np.array(y, dtype=np.float).reshape( (-1,self.__dim) )
        x = np.array(x,dtype=np.float)

        for d in range(self.__dim):
            if len(y)!=0:
                self.__gp[d].learn( x, y[:,d] )
            else:
                self.__gp[d].learn( x, [] )


    def calc_lik(self, x, y ):
        lik = 0.0
        y = np.array(y, dtype=np.float).reshape( (-1,self.__dim) )
        x = np.array(x,dtype=np.float)
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


def main():
    pass  

if __name__ == '__main__':
    main()