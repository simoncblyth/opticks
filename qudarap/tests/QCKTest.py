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

    def s2cn_plot(self, ii):
        """
        :param ii: list of first dimension indices, corresponding to BetaInverse values
        """
        s2cn = self.s2cn
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        for i in ii:
            ax.plot( s2cn[i,:,0], s2cn[i,:,1] , label="%d" % i )
        pass
        #ax.legend()
        fig.show()

    def rindex_plot(self):
        ri = self.rindex
        c2 = self.c2
        c2where = self.c2where
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

        for i in c2where:
            ax.plot( [edges[i], edges[i]], ylim , label="edge %d " % i, linestyle="dotted" )
            ax.plot( [edges[i+1], edges[i+1]], ylim , label="edge+1 %d " % (i+1), linestyle="dotted" )
        pass

        ax.legend()
        fig.show() 

    def en_plot(self, bi):
        ri = self.rindex
        base = os.path.expandvars("$TMP/QCKTest/%6.4f" % bi )
        log.info("load from %s " % base)
        el = np.load(os.path.join(base,"test_energy_lookup_many.npy"))
        es = np.load(os.path.join(base,"test_energy_sample_many.npy"))

        #edges = np.linspace(1.55,15.5,100)  # including rightmost 
        edges = divide_edges( ri[:,0], mul=4 )  

        hl = np.histogram( el, bins=edges )
        hs = np.histogram( es, bins=edges )

        c2, c2n, c2c = chi2( hl[0], hs[0] )
        ndf = max(c2n - 1, 1)
        c2p = c2.sum()/ndf
        c2l = "chi2/ndf %4.2f [%d]" % (c2p, ndf)

        c2amx = c2.argmax()
        ri = self.rindex
        rimax = ri[:,1].max()
        c2max = c2.max()
        c2riscale = rimax/c2max
        c2where = np.where( c2 > c2max/3. )[0]


        hmax = max(hl[0].max(), hs[0].max())
        c2hscale = hmax/c2max

        extra = " c2max:%4.2f c2amx:%d c2[c2amx] %4.2f edges[c2amx] %5.3f edges[c2amx+1] %5.3f " % (c2max, c2amx, c2[c2amx], edges[c2amx], edges[c2amx+1] ) 

        self.c2riscale = c2riscale
        self.bi = bi
        self.edges = edges
        self.c2 = c2
        self.c2max = c2max
        self.c2where = c2where

        icdf_shape = str(self.s2cn.shape)

        title_ = ["QCKTest.py : en_plot : icdf_shape %s : lookup vs sampled : %s " % ( icdf_shape, c2l ), 
                   base, 
                   extra
                 ]
 
        title = "\n".join(title_)
        print(title)
        print("c2", c2)
        print("c2n", c2n)
        print("c2c", c2c)

        for q in "el es hl hs c2 c2n c2c".split(): globals()[q] = locals()[q]

        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        fig.suptitle(title)

        ax.scatter( hl[1][:-1], hl[0], label="lookup" )
        ax.scatter( hs[1][:-1], hs[0], label="sample" )

        ax.plot( edges[:-1], hl[0], drawstyle="steps-post" )
        ax.plot( edges[:-1], hs[0], drawstyle="steps-post" )
        ax.plot( edges[:-1], c2*c2hscale , label="c2", drawstyle="steps-post" )

        ax.legend()
        fig.show() 
        




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QCKTest()
    #ii = np.arange( 0, 1000, 10 )
    #t.s2cn_plot(ii)

    #bi = 1.7920
    bi = 1.5
    t.en_plot(bi) 
    t.rindex_plot() 







