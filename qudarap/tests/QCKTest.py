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


See also::

    ana/rindex.py 
    ana/ckn.py 

"""
import os, logging, numpy as np

from opticks.ana.nbase import chi2
from opticks.ana.edges import divide_bins
from opticks.ana.rsttable import RSTTable

log = logging.getLogger(__name__)


class QCKTest(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/QCerenkovTest") 

    def __init__(self, approach="UpperCut", use_icdf=False):
        assert approach in ["UpperCut", "SplitBin"] 
        self.approach = approach
        self.use_icdf = use_icdf

        base_path = os.path.join(self.FOLD, "test_makeICDF_%s" % approach)
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
        sample_base = os.path.join(base_path, "QCKTest")
        self.sample_base = sample_base
    pass

    def bislist(self):
        names = sorted(os.listdir(os.path.expandvars(self.sample_base)))
        names = filter(lambda n:not n.startswith("bis"), names)
        print(names)
        bis = list(map(float, names))
        return bis

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

    def en_load(self, bi):
        bi_base = os.path.expandvars("%s/%6.4f" % (self.sample_base, bi) )

        use_icdf = self.use_icdf 
        ext = ["s2cn","icdf"][int(use_icdf)] ; 
        log.info("load from bi_base %s ext %s " % (bi_base, ext))

        el = np.load(os.path.join(bi_base,"test_energy_lookup_many_%s.npy" % ext ))
        es = np.load(os.path.join(bi_base,"test_energy_sample_many.npy"))

        tl = np.load(os.path.join(bi_base,"test_energy_lookup_many_tt.npy"))
        ts = np.load(os.path.join(bi_base,"test_energy_sample_many_tt.npy"))

        self.bi_base = bi_base
        self.el = el
        self.es = es
        self.tl = tl
        self.ts = ts


    def check_s2c_monotonic(self):
        s2c = self.s2c 
        for i in range(len(s2c)):  
            w = np.where( np.diff(s2c[i,:,2]) < 0 )[0]  
            print(" %5d : %s " % (i, str(w)))
        pass

    def en_compare(self, bi, num_edges=101): 
        """
        Compare the energy samples created by QCKTest for a single BetaInverse
        """
        ri = self.rindex
        el = self.el
        es = self.es
        s2cn = self.s2cn
        avph = self.avph
        s2c  = self.s2c

        ibi = self.getBetaInverseIndex(bi)

        approach = self.approach
        if approach == "UpperCut":  # see QCerenkov::getS2Integral_UpperCut
            en_slot = 0  
            s2_slot = 1
            cdf_slot = 2
            emn = s2cn[ibi, 0,en_slot] 
            emx = s2cn[ibi,-1,en_slot] 
            avp = s2c[ibi, -1,cdf_slot] 
        elif approach == "SplitBin":  # see QCerenkov::getS2Integral_SplitBin
            en_slot = 0   # en_b
            s2_slot = 5   # s2_b 
            cdf_slot = 7  # s2integral
            emn = avph[ibi, 1]
            emx = avph[ibi, 2]
            avp = avph[ibi, 3]
        else:
            assert 0, "unknown approach %s " % approach
        pass 

        self.en_slot = en_slot
        self.s2_slot = s2_slot
        self.cdf_slot = cdf_slot
        self.emn = emn
        self.emx = emx 
        self.avp = avp 

        edom = emx - emn 
        edif = edom/(num_edges-1)
        edges0 = np.linspace( emn, emx, num_edges )    # across Cerenkov permissable range 
        edges = np.linspace( emn-edif, emx+edif, num_edges + 2 )   # push out with extra bins either side
       
        #edges = np.linspace(1.55,15.5,100)  # including rightmost 
        #edges = np.linspace(1.55,15.5,200)  # including rightmost 
        #edges = divide_bins( ri[:,0], mul=4 )  

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

        qq = "hl hs c2 c2label c2n c2c c2riscale c2hscale hmax edges c2max c2poppy cf bi ibi"
        for q in qq.split():
            globals()[q] = locals()[q]
            setattr(self, q, locals()[q] )
        pass

        t = self
        print("np.c_[t.c2, t.hs[0], t.hl[0]][t.c2 > 0]")
        print(np.c_[t.c2, t.hs[0], t.hl[0]][t.c2 > 0] )

        return [bi, c2sum, ndf, c2p, emn, emx, avp ] 

    LABELS = "bi c2sum ndf c2p emn emx avp".split()   

    def en_plot(self, c2overlay=0., c2poppy_=True):
        """

        Using divide_edges is good for chi2 checking as it prevents 
        bin migrations or "edge" effects.  But it leads to rather 
        differently sized bins resulting in a strange histogram shape. 

        """
        ri = self.rindex
        s2c = self.s2c 
        s2cn = self.s2cn 
        bi = self.bi
        ibi = self.ibi
        c2 = self.c2 
        c2poppy = self.c2poppy 
        c2hscale = self.c2hscale
        hmax = self.hmax
        hl = self.hl 
        hs = self.hs 
        edges = self.edges
        en_slot = self.en_slot
        s2_slot = self.s2_slot
        cdf_slot = self.cdf_slot

        emn = self.emn
        emx = self.emx
        avp = self.avp


        icdf_shape = str(s2cn.shape)

        title_ = ["QCKTest.py : en_plot : lookup cf sampled : icdf_shape %s : %s " % ( icdf_shape, self.bi_base ), 
                  "%s : %s " % ( self.c2label, self.cf), 
                  "approach:%s use_icdf:%s avp %6.2f " % (self.approach, self.use_icdf, avp )
                 ]
 
        title = "\n".join(title_)
        print(title)
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        fig.suptitle(title)

        ax.plot( edges[:-1], hl[0], drawstyle="steps-post", label="lookup" )
        ax.plot( edges[:-1], hs[0], drawstyle="steps-post", label="sampled" )

        if c2overlay != 0.:
            ax.plot( edges[:-1], c2*c2hscale*c2overlay , label="c2", drawstyle="steps-post" )
        pass    

        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        s2max = s2cn[ibi,:,s2_slot].max() 
        ax.plot( s2cn[ibi,:,en_slot], s2cn[ibi,:,s2_slot]*hmax/s2max , label="s2cn[%d,:,%d]*hmax/s2max (s2)" % (ibi,s2_slot) )
        ax.plot( s2cn[ibi,:,en_slot], s2cn[ibi,:,cdf_slot]*hmax ,       label="s2cn[%d,:,%d]*hmax      (cdf)" % (ibi, cdf_slot) )

        ax.set_xlim(xlim) 
        ax.set_ylim(ylim) 

        ax.plot( [emx, emx], ylim, linestyle="dotted" )
        ax.plot( [emn, emn], ylim, linestyle="dotted" )

        if c2poppy_:
            for i in c2poppy:
                ax.plot( [edges[i], edges[i]], ylim ,     label="c2poppy edge   %d " % i    , linestyle="dotted" )
                ax.plot( [edges[i+1], edges[i+1]], ylim , label="c2poppy edge+1 %d " % (i+1), linestyle="dotted" )
            pass
        pass

        ax.legend()
        fig.show() 
        figpath = os.path.join(self.bi_base, "en_plot.png")
        log.info("savefig %s " % figpath)
        fig.savefig(figpath)

    def compare(t, bis):
        res = np.zeros( (len(bis), len(t.LABELS)) )
        for i, bi in enumerate(bis):
            t.en_load(bi)
            res[i] = t.en_compare(bi)
            t.en_plot(c2overlay=0.5, c2poppy_=False) 
            #t.rindex_plot() 
        pass
        t.res = res

    def __str__(self):
        title = "%s use_icdf:%s" % ( self.sample_base, self.use_icdf )
        underline = "=" * len(title)
        rst = RSTTable.Rdr(self.res, self.LABELS )
        return "\n".join(["", title, underline, "", rst]) 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    approach = "UpperCut"
    #approach = "SplitBin"
    use_icdf = False 

    t = QCKTest(approach=approach, use_icdf=use_icdf)
    #t.s2cn_plot(istep=20)
    bis = t.bislist()

    #bis = bis[-2:-1]
    #bis = [1.45,]
    #bis = [1.6,]

    t.compare(bis)
    print(t)



