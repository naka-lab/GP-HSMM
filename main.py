# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from GPSegmentation import GPSegmentation
import time
import matplotlib.pyplot as plt
import numpy as np

def learn( savedir ):
    dim = 1
    classes = 5
    num_learn = 10

    gpsegm = GPSegmentation(dim, classes)

    files =  [ "40fps_data_norma%03d.txt" % j for j in range(8) ][::2]
    gpsegm.load_data( files )

    liks = []

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


def recog( modeldir, savedir ):
    gpsegm = GPSegmentation(1,5)

    gpsegm.load_data( [ "testdata2d_%03d.txt" % j for j in range(4) ] )
    gpsegm.load_model( modeldir )


    start = time.clock()
    for it in range(5):
        print( "-----", it, "-----" )
        gpsegm.recog()
        print( "lik =", gpsegm.calc_lik() )
    print( time.clock()-start )
    gpsegm.save_model( savedir )



def main():
    learn( "learn/" )
    recog( "learn/" , "recog/" )
    return

if __name__=="__main__":
    main()
