#!/usr/bin/env python
"""
QCKTest.py
==================

::

    QCerenkovTest 
    ipython -i tests/QCKTest.py


Hmm largest c2 for different BetaInverse always at 7.6eV 
just to left of rindex peak. 
Possibly there is a one bin shifted issue, that is showing up 
the most in the region where rindex is changing fastest.  

Perhaps could check this with an artifical rindex pattern, 
such as a step function.

Actually just setting BetaInverse to 1.792 just less than rmx 1.793
is informative as then there is only a very small range 
of possible energies. 

Hmm: generating millions of photons just there is a kinda 
extreme test, as in reality will only be 1.

Hmm maybe should exclude BetaInverse where the average number
of photons is less than 1 

::

    In [18]: np.c_[t.s2c[-120:-100,-1],t.bis[-120:-100]]                                                                                                                                                      
    Out[18]: 
    array([[8.45 , 1.113, 1.746],
           [8.448, 1.1  , 1.746],
           [8.446, 1.087, 1.747],
           [8.444, 1.075, 1.747],
           [8.443, 1.062, 1.747],
           [8.441, 1.05 , 1.748],
           [8.439, 1.037, 1.748],
           [8.437, 1.025, 1.749],
           [8.435, 1.012, 1.749],
           [8.433, 1.   , 1.749],
           [8.431, 0.988, 1.75 ],
           [8.429, 0.976, 1.75 ],
           [8.427, 0.963, 1.751],
           [8.425, 0.951, 1.751],
           [8.423, 0.939, 1.751],
           [8.421, 0.927, 1.752],
           [8.419, 0.915, 1.752],
           [8.417, 0.903, 1.753],
           [8.415, 0.891, 1.753],
           [8.413, 0.879, 1.753]])





See also::

    ana/rindex.py 
    ana/ckn.py 

"""
import os, logging, numpy as np

from opticks.ana.nbase import chi2
from opticks.ana.edges import divide_edges

log = logging.getLogger(__name__)


class QCKTest(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/QCerenkovTest") 
    def __init__(self):
        base_path = os.path.join(self.FOLD, "test_makeICDF")
        if not os.path.exists(base_path):
            log.fatal("base_path %s does not exist" % base_path)
            return 
        pass 
        names = os.listdir(base_path)
        log.info("loading from base_path %s " % base_path)
        for name in filter(lambda _:_.endswith(".npy"), names):
            path = os.path.join(base_path, name)
            stem = name[:-4]
            a = np.load(path) 
            print( " t.%5s  %s " % (stem, str(a.shape))) 
            setattr(self, stem, a )
        pass
    pass

    def s2cn_plot(self, istep):
        """
        :param ii: list of first dimension indices, corresponding to BetaInverse values
        """
        s2cn = self.s2cn
        ii = np.arange( 0,len(s2cn), istep )  

        title_ = "QCKTest.py : s2cn_plot : s2cn.shape %s  istep %d " % (str(s2cn.shape), istep) 
        desc_ = "JUNO LS : Cerenkov S2 integral CDF for sample of BetaInverse values" 

        title = "\n".join([title_, desc_])   
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        fig.suptitle(title)
        for i in ii:
            ax.plot( s2cn[i,:,0], s2cn[i,:,1] , label="%d" % i )
        pass
        #ax.legend()
        fig.show()


    def one_s2cn_plot(self, BetaInverse ):
        s2cn = self.s2cn
        ibi = self.getBetaInverseIndex(BetaInverse)
        title_ = "QCKTest.py : one_s2cn_plot BetaInverse %6.4f  ibi %d s2cn[ibi] %s " % (BetaInverse, ibi, str(s2cn[ibi].shape))
        desc_ = " cdf (normalized s2 integral) for single BetaInverse " 

        title = "\n".join([title_, desc_]) ;  
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        fig.suptitle(title)

        ax.plot( s2cn[ibi,:,0], s2cn[ibi,:,1] , label="s2cn[%d]" % ibi )
        ax.legend()
        fig.show()


    def getBetaInverseIndex(self, BetaInverse):
        bis = self.bis
        ibi = np.abs(bis - BetaInverse).argmin()
        return ibi

    def rindex_plot(self):
        ri = self.rindex
        c2 = self.c2
        c2poppy = self.c2poppy
        bi = self.bi
        edges = self.edges
        c2riscale = self.c2riscale

        title = "\n".join(["QCKTest.py : rindex_plot"]) ;  
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        fig.suptitle(title)

        ax.scatter( ri[:,0], ri[:,1], label="ri" )
        ax.plot(    ri[:,0], ri[:,1], label="ri" )
        ax.plot(    edges[:-1], c2*c2riscale,    label="c2", drawstyle="steps-post" )
        ax.plot( [ri[0,0], ri[-1,0]], [bi, bi], label="bi %6.4f " % bi )  

        bi0 = 1.75     
        ax.plot( [ri[0,0], ri[-1,0]], [bi0, bi0], label="bi0 %6.4f " % bi0 )       

        ylim = ax.get_ylim()

        for i in c2poppy:
            ax.plot( [edges[i], edges[i]], ylim , label="edge %d " % i, linestyle="dotted" )
            ax.plot( [edges[i+1], edges[i+1]], ylim , label="edge+1 %d " % (i+1), linestyle="dotted" )
        pass

        ax.legend()
        fig.show() 


    BASE = "$TMP/QCKTest"

    def bislist(self):
        names = sorted(os.listdir(os.path.expandvars(self.BASE)))
        print(names)
        bis = list(map(float, names))
        return bis

    def en_load(self, bi):
        base = os.path.expandvars("%s/%6.4f" % (self.BASE, bi) )
        log.info("load from %s " % base)
        el = np.load(os.path.join(base,"test_energy_lookup_many.npy"))
        es = np.load(os.path.join(base,"test_energy_sample_many.npy"))

        self.base = base
        self.el = el
        self.es = es

    def en_compare(self, bi): 

        ri = self.rindex
        el = self.el
        es = self.es
        
        edges = np.linspace(1.55,15.5,100)  # including rightmost 
        #edges = divide_edges( ri[:,0], mul=4 )  

        hl = np.histogram( el, bins=edges )
        hs = np.histogram( es, bins=edges )

        c2, c2n, c2c = chi2( hl[0], hs[0] )
        ndf = max(c2n - 1, 1)
        c2sum = c2.sum()
        c2p = c2sum/ndf
        c2label = "chi2/ndf %4.2f [%d] %.2f " % (c2p, ndf, c2sum)

        c2amx = c2.argmax()
        rimax = ri[:,1].max()
        c2max = c2.max()
        c2riscale = rimax/c2max
        c2poppy = np.where( c2 > c2max/3. )[0]

        hmax = max(hl[0].max(), hs[0].max())
        c2hscale = hmax/c2max

        cf = " c2max:%4.2f c2amx:%d c2[c2amx] %4.2f edges[c2amx] %5.3f edges[c2amx+1] %5.3f " % (c2max, c2amx, c2[c2amx], edges[c2amx], edges[c2amx+1] ) 

        print("cf", cf)

        #print("c2", c2)
        print("c2n", c2n)
        print("c2c", c2c)

        qq = "hl hs c2 c2label c2n c2c c2riscale c2hscale hmax edges c2max c2poppy cf"
        for q in qq.split():
            globals()[q] = locals()[q]
            setattr(self, q, locals()[q] )
        pass


        t = self
        print("np.c_[t.c2, t.hs[0], t.hl[0]][t.c2 > 0]")
        print(np.c_[t.c2, t.hs[0], t.hl[0]][t.c2 > 0] )

       

    def en_plot(self, bi, c2overlay=False):
        """

        Using divide_edges is good for chi2 checking as it prevents 
        bin migrations or "edge" effects.  But it leads to rather 
        differently sized bins resulting in a strange histogram shape. 

        """
        ri = self.rindex
        s2cn = self.s2cn 
        ibi = self.getBetaInverseIndex(bi)
        c2 = self.c2 
        c2hscale = self.c2hscale
        hmax = self.hmax
        hl = self.hl 
        hs = self.hs 
        edges = self.edges

        icdf_shape = str(s2cn.shape)

        title_ = ["QCKTest.py : en_plot : lookup cf sampled : icdf_shape %s : %s " % ( icdf_shape, self.base ), 
                  "%s : %s " % ( self.c2label, self.cf)
                 ]
 
        title = "\n".join(title_)
        print(title)
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        fig.suptitle(title)

        #ax.scatter( hl[1][:-1], hl[0], label="lookup" )
        #ax.scatter( hs[1][:-1], hs[0], label="sample" )

        ax.plot( edges[:-1], hl[0], drawstyle="steps-post" )
        ax.plot( edges[:-1], hs[0], drawstyle="steps-post" )

        if c2overlay:
            ax.plot( edges[:-1], c2*c2hscale , label="c2", drawstyle="steps-post" )
        pass    

        ax.plot( s2cn[ibi,:,0], s2cn[ibi,:,1]*hmax , label="s2cn[%d]*hmax" % ibi )

        ax.legend()
        fig.show() 
        


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QCKTest()
    #t.s2cn_plot(istep=20)


    for bi in t.bislist():
        t.en_load(bi)
        t.en_compare(bi)
        t.en_plot(bi) 
    pass


    #t.rindex_plot() 
    #t.one_s2cn_plot(bi) 








