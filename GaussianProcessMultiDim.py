# encoding: utf8
from __future__ import unicode_literals
from __future__ import print_function
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import Gaussianprocess_np
import matplotlib.pyplot as plt


class GPMD:
    def __init__(self, denom, dim):
        self.__dim = dim
        self.__gp = [ Gaussianprocess_np.SORGP( denom, dim ) for d in range(self.__dim) ]

    def learn(self,x, y ):
        y = np.array(y, dtype=np.float).reshape( (-1,self.__dim) )
        x = np.array(x,dtype=np.float)

        for d in range(self.__dim):
            if len(y)!=0:
                #print ("check")
                #print (x)
                #print (y[:,d])
                self.__gp[d].learn( x, y[:,d] )
            else:
                self.__gp[d].learn( x, [] )


    def calc_lik(self, x, y ):
        liks = 0.0

        if self.__dim==1:
            y = np.asarray(y, dtype=np.float).reshape( (-1,self.__dim) )
        #x = np.asarray(x,dtype=np.float)
        for d in range(self.__dim):
            mu, sig, lik = self.__gp[d].predict( x , y[:,d] )
            liks += lik
            #print (lik)

        return liks
