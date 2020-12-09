# encoding: utf8
from __future__ import unicode_literals
from __future__ import print_function
import GaussianProcessMultiDim
import random
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import os
#from scipy.misc import logsumexp
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from cymath import logsumexp




class GPSegmentation():
    # parameters
    MAX_LEN = 100
    MIN_LEN = 50
    AVE_LEN = 75
    SKIP_LEN = 1

    def __init__(self, dim, nclass):
        self.dim = dim
        self.numclass = nclass
        self.segmlen = 3
        self.gps = [ GaussianProcessMultiDim.GPMD( dim, self.MAX_LEN ) for i in range(self.numclass) ]
        self.segm_in_class= [ [] for i in range(self.numclass) ]
        self.segmclass = {}
        self.segments = []
        self.trans_prob = np.ones( (nclass,nclass) )
        self.trans_prob_bos = np.ones( nclass )
        self.trans_prob_eos = np.ones( nclass )
        self.is_initialized = False

        self.a_time = 0
        #self.prior_table = [self.AVE_LEN**i * math.exp(-self.AVE_LEN) / math.factorial(i) for i in range(self.MAX_LEN)]

    def load_data(self, filenames, classfile=None ):
        self.data = []
        self.segments = []
        self.is_initialized = False

        for fname in filenames:
            y = np.loadtxt( fname )
            segm = []
            self.data.append( y )

            # ランダムに切る
            i = 0
            while i<len(y):
                length = random.randint(self.MIN_LEN, self.MAX_LEN)

                if i+length+1>=len(y):
                    length = len(y)-i

                segm.append( y[i:i+length+1] )

                i+=length

            self.segments.append( segm )

            # ランダムに割り振る
            for i,s in enumerate(segm):
                c = random.randint(0,self.numclass-1)
                self.segmclass[id(s) ] = c

        # 遷移確率更新
        self.calc_trans_prob()



    def load_model( self, basename ):
        # GP読み込み
        for c in range(self.numclass):
            filename = basename + "class%03d.npy" % c
            self.segm_in_class[c] = np.load( filename, allow_pickle=True)
            self.update_gp( c )

        # 遷移確率更新
        self.trans_prob = np.load( basename+"trans.npy", allow_pickle=True )
        self.trans_prob_bos = np.load( basename+"trans_bos.npy", allow_pickle=True )
        self.trans_prob_eos = np.load( basename+"trans_eos.npy", allow_pickle=True )



    def update_gp(self, c ):
        #print ("update_gp")
        datay = []
        datax = []
        for s in self.segm_in_class[c]:
            datay += [ y for y in s ]
            datax += range(len(s))

        t_time = time.time()
        self.gps[c].learn( datax, datay )
        self.a_time += time.time() - t_time


    def calc_emission_logprob( self, c, segm ):
        gp = self.gps[c]
        slen = len(segm)

        if len(segm) > 2:
            #plen = self.AVE_LEN**slen * math.exp(-self.AVE_LEN) / math.factorial(slen)
            log_plen = (slen*math.log(self.AVE_LEN) + (-self.AVE_LEN)*math.log(math.e)) - (sum(np.log(np.arange(1,slen+1))))
            p = gp.calc_lik( np.arange(len(segm), dtype=np.float) , segm )
            #return p + math.log(plen)
            return p + log_plen
        else:
            return math.log(1.0e-100)

    def save_model(self, basename ):
        if not os.path.exists(basename):
            os.mkdir( basename )

        for n,segm in enumerate(self.segments):
            classes = []
            cut_points = []
            for s in segm:
                c = self.segmclass[id(s)]
                classes += [ c for i in range(len(s)) ]
                cut_points += [0] * len(s)
                cut_points[-1] = 1
            np.savetxt( basename+"segm%03d.txt" % n, np.vstack([classes,cut_points]).T, fmt=str("%d") )


        # 各クラスに分類されたデータを保存
        for c in range(len(self.gps)):
            for d in range(self.dim):
                plt.clf()
                for data in self.segm_in_class[c]:
                    if self.dim==1:
                        plt.plot( range(len(data)), data, "o-" )
                    else:
                        plt.plot( range(len(data[:,d])), data[:,d], "o-" )
                    plt.ylim( -1, 1 )
                plt.savefig( basename+"class%03d_dim%03d.png" % (c, d) )

        # テキストでも保存
        np.save( basename + "trans.npy" , self.trans_prob  )
        np.save( basename + "trans_bos.npy" , self.trans_prob_bos )
        np.save( basename + "trans_eos.npy" , self.trans_prob_eos )

        for c in range(self.numclass):
            np.save( basename+"class%03d.npy" % c, self.segm_in_class[c] )


    def forward_filtering(self, d ):
        T = len(d)
        log_a = np.log( np.zeros( (len(d), self.MAX_LEN, self.numclass) )  + 1.0e-100 )  # 前向き確率．対数で確率を保持．1.0e-100で確率0を近似的に表現．
        #a = np.zeros( (len(d), self.MAX_LEN, self.numclass) ) + 1.0-e100
        valid = np.zeros( (len(d), self.MAX_LEN, self.numclass) ) # 計算された有効な値可どうか．計算されていない場所の確率を0にするため．
        z = np.ones( T ) # 正規化定数

        for t in range(T):
            for k in range(self.MIN_LEN,self.MAX_LEN,self.SKIP_LEN):
                if t-k<0:
                    break

                segm = d[t-k:t+1]
                for c in range(self.numclass):
                    out_prob = self.calc_emission_logprob( c, segm )
                    foward_prob = 0.0

                    # 遷移確率
                    tt = t-k-1
                    if tt>=0:
                        #for kk in range(self.MAX_LEN):
                        #    for cc in range(self.numclass):
                        #        foward_prob += a[tt,kk,cc] * self.trans_prob[cc, c]
                        #foward_prob = math.log(np.sum( a[tt,:,:] * self.trans_prob[:,c] )) + out_prob
                        foward_prob = logsumexp( log_a[tt,:,:] + z[tt] + np.log(self.trans_prob[:,c]) ) + out_prob
                        #foward_prob = logsumexp( a[tt,:,:] + z[tt] + np.log(self.trans_prob[:,c]) ) + out_prob
                    else:
                        # 最初の単語
                        foward_prob = out_prob + math.log(self.trans_prob_bos[c])

                    if t==T-1:
                        # 最後の単語
                        foward_prob += math.log(self.trans_prob_eos[c])

                    # 正規化を元に戻す
                    log_a[t,k,c] = foward_prob
                    #a[t,k,c] = foward_prob
                    valid[t,k,c] = 1.0
                    if math.isnan(foward_prob):
                        print( "a[t=%d,k=%d,c=%d] became NAN!!" % (t,k,c) )
                        sys.exit(-1)
            # 正規化
            if t-self.MIN_LEN>=0:
                z[t] = logsumexp( log_a[t,:,:] )
                log_a[t,:,:] -= z[t]
                #z[t] = logsumexp( a[t,:,:] )
                #a[t,:,:] -= z[t]

        return np.exp(log_a)*valid
        #return np.exp(a)*valid


    def sample_idx(self, prob ):
        accm_prob = [0,] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i


    def backward_sampling(self, a, d):
        T = a.shape[0]
        t = T-1

        segm = []
        segm_class = []
        c = -1
        while True:
            if t==T-1:
                transp = self.trans_prob_eos
            else:
                transp = self.trans_prob[:,c]

            idx = self.sample_idx( (a[t]*transp).reshape( self.MAX_LEN*self.numclass ))

            k = int(idx/self.numclass)
            c = idx % self.numclass

            if t-k-1<=0:
                #先頭
                s = d[0:t+1]
            else:
                #先頭以外
                s = d[t-k:t+1]

            segm.insert( 0, s )
            segm_class.insert( 0, c )

            t = t-k-1

            if t<=0:
                break

        return segm, segm_class


    def calc_trans_prob( self ):
       self.trans_prob = np.zeros( (self.numclass,self.numclass) )
       self.trans_prob_bos = np.zeros( self.numclass )
       self.trans_prob_eos = np.zeros( self.numclass )
       self.trans_prob += 0.1
       self.trans_prob_bos += 0.1
       self.trans_prob_eos += 0.1
       # 数え上げ
       for n,segm in enumerate(self.segments):
           if id(segm[0]) in self.segmclass:
               c_begin = self.segmclass[ id(segm[0]) ]
               self.trans_prob_bos[c_begin]+=1
           if id(segm[-1]) in self.segmclass:
               c_end = self.segmclass[ id(segm[-1]) ]
               self.trans_prob_eos[c_end]+=1
           for i in range(1,len(segm)):
               try:
                   cc = self.segmclass[ id(segm[i-1]) ]
                   c = self.segmclass[ id(segm[i]) ]
               except KeyError:
                   # gibss samplingで除かれているものは無視
                   continue
               self.trans_prob[cc,c] += 1
       # 正規化
       self.trans_prob = self.trans_prob / self.trans_prob.sum(1).reshape(self.numclass,1)
       self.trans_prob_bos = self.trans_prob_bos / np.sum( self.trans_prob_bos )
       self.trans_prob_eos = self.trans_prob_eos / np.sum( self.trans_prob_eos )


    # list.remove( elem )だとValueErrorになる
    def remove_ndarray(self, lst, elem ):
        l = len(elem)
        for i,e in enumerate(lst):
            if len(e)!=l:
                continue
            if (e==elem).all():
                lst.pop(i)
                return
        raise ValueError( "ndarray is not found!!" )

    def learn(self):
        if self.is_initialized==False:
            # GPの学習
            for i in range(len(self.segments)):
                for s in self.segments[i]:
                    c = self.segmclass[id(s)]
                    self.segm_in_class[c].append( s )

            # 各クラス毎に学習
            for c in range(self.numclass):
                self.update_gp( c )

            self.is_initialized = True

        self.update(True)


    def recog(self):
        self.update(False)


    def update(self, learning_phase=True ):

        for i in range(len(self.segments)):
            d = self.data[i]
            segm = self.segments[i]

            for s in segm:
                c = self.segmclass[id(s)]
                self.segmclass.pop( id(s) )

                if learning_phase:
                    # パラメータ更新
                    self.remove_ndarray( self.segm_in_class[c], s )

            if learning_phase:
                # GP更新
                for c in range(self.numclass):
                    self.update_gp( c )

                # 遷移確率更新
                self.calc_trans_prob()

            start = time.clock()
            print( "forward...", end="")
            a = self.forward_filtering( d )

            print( "backward...", end="" )
            segm, segm_class = self.backward_sampling( a, d )
            print( time.clock()-start, "sec" )

            print( "Number of classified segments: [", end="")
            for s in self.segm_in_class:
                print( len(s), end=" " )
            print( "]" )


            self.segments[i] = segm

            for s,c in zip( segm, segm_class ):
                self.segmclass[id(s)] = c

                # パラメータ更新
                if learning_phase:
                    self.segm_in_class[c].append(s)

            if learning_phase:
                # GP更新
                for c in range(self.numclass):
                    self.update_gp( c )

                # 遷移確率更新
                self.calc_trans_prob()

        print ("gp learn time", self.a_time)
        return


    def calc_lik(self):
        #print ("calc lik")
        lik = 0
        for segm in self.segments:
            for s in segm:
                c = self.segmclass[id(s)]
                #lik += self.gps[c].calc_lik( np.arange(len(s),dtype=np.float) , np.array(s) )
                lik += self.gps[c].calc_lik( np.arange(len(s), dtype=np.float) , s )

        return lik
