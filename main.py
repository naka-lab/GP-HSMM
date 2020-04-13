# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from GPSegmentation import GPSegmentation
import time

def learn( savedir ):
    gpsegm = GPSegmentation(2,5)

    files =  [ "testdata2d_%03d.txt" % j for j in range(4) ]
    gpsegm.load_data( files )

    start = time.clock()
    for it in range(3):
        print( "-----", it, "-----" )
        gpsegm.learn()
        gpsegm.save_model( savedir )
        print( "lik =", gpsegm.calc_lik() )
    print( time.clock()-start )
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
