# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from GPSegmentation import GPSegmentation
import time
import matplotlib.pyplot as plt
import numpy as np

def learn( savedir ):
    gpsegm = GPSegmentation(2,5)

    files =  [ "testdata2d_%03d.txt" % j for j in range(4) ]
    gpsegm.load_data( files )

    liks = []
    num_learn = 8

    start = time.clock()
    for it in range(num_learn):
        print( "-----", it, "-----" )
        gpsegm.learn()
        gpsegm.save_model( savedir )
        lik = gpsegm.calc_lik()
        print( "lik =", lik )
        liks.append(lik)
    print( time.clock()-start )

    plt.figure()
    plt.plot(np.arange(num_learn), np.array(liks))
    plt.savefig(savedir+"liks.png")
    return gpsegm.calc_lik()

"""
def recog( modeldir, savedir ):
    gpsegm = GPSegmentation(2,5)

    gpsegm.load_data( [ "testdata2d_%03d.txt" % j for j in range(4) ] )
    gpsegm.load_model( modeldir )


    start = time.clock()
    for it in range(5):
        print( "-----", it, "-----" )
        gpsegm.recog()
        print( "lik =", gpsegm.calc_lik() )
    print( time.clock()-start )
    gpsegm.save_model( savedir )
"""


def main():
    learn( "learn/" )
    #recog( "learn/" , "recog/" )
    return

if __name__=="__main__":
    main()
