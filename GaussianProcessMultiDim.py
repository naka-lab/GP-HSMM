# encoding: utf8
from __future__ import unicode_literals
from __future__ import print_function
import pyximport
import numpy as np
#pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import Gaussianprocess_torch_WOinv as GP_lu
import matplotlib.pyplot as plt
import time


class GPMD:
    def __init__(self, dim):
        self.__dim = dim
        self.__gp = [ GP_lu.GP(1, 16.0, "cpu") for d in range(self.__dim) ]
        #self.b_time = 0

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
                #self.__gp[d].learn( x, [] )
                self.__gp[d] = GP_lu.GP(1,16.0, "cpu")


    def calc_lik(self, x, y ):
        #s_time = time.time()
        liks = 0.0

        if self.__dim==1:
            y = np.asarray(y, dtype=np.float).reshape( (-1,self.__dim) )
        #x = np.asarray(x,dtype=np.float)
        for d in range(self.__dim):
            mu, sig, lik = self.__gp[d].predict( x , y[:,d] )
            liks += lik
            #print (lik)
        #self.b_time += time.time() - s_time
        return liks
