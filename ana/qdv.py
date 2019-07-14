#!/usr/bin/env python
"""
qdv.py : photon deviation reporting
========================================

This replaces the slow dv.py with a much faster implementation

* https://bitbucket.org/simoncblyth/opticks/commits/d31f4e271ee34a7fe1bf68af3e07f90ceb54565e

fixed slow deviation checking for large numbers of photons by loop inversion
from select-then-deviate to deviate-then-select so only do the expensive
deviation check once : replaces ana/dv.py with ana/qdv.py see
notes/issues/py-analysis-too-slow-for-big-events.rst

"""
import os, sys, logging, numpy as np
#from opticks.ana.log import fatal_, error_, warning_, info_, debug_
from opticks.ana.log import underline_, blink_ 

from opticks.ana.level import Level 

log = logging.getLogger(__name__)



class QDV(object):

   FMT  =       "  %9d %9d : %5d %5d %5d : %6.4f %6.4f %6.4f : %9.4f %9.4f %9.4f  "
   CFMT =       "  %9s %9s : %5s %5s %5s : %6s %6s %6s : %9s %9s %9s    "
   CFMT_CUTS =  "  %9s %9s : %5s %5s %5s : %6.4f %6.4f %6.4f : %9s %9s %9s    "
   CFMT_COLS =  "nitem nelem nwar nerr nfat fwar ferr ffat mx mn avg".split()

   LMT  = " %0.4d %10s : %30s : %7d  %7d "   # labels seqhis line 
   CLMT = " %4s %10s : %30s : %7s  %7s " 

   clabel = CLMT % ( "idx", "msg", "sel", "lcu1", "lcu2" )
   cblank = CLMT % ( "", "", "", "", "" )


   def __init__(self, tab, idx, sel, dv, ndv, lcu, dvmax, msg=""):
       """
       :param tab: QDVTab instance 
       :param idx: unskipped orignal seqhis line index
       :param sel: single line selection eg 'TO BT BT SA'
       :param dv: photon max deviations within seqhis code selection  
       :param ndv: number of elements aggregated in the max
       :param lcu: list of length 3 with (seqhis-bigint, a-count, b-count)
       :param dvmax: triplet of floats for warn/error/fatal deviation levels

       Access an Dv instance in ipython::

            In [12]: ab.ox_dv.dvs[2]
            Out[12]:  0002            :                 TO BT BR BT SA :     561      527  :          27       324/       14: 0.043  mx/mn/av   0.00238/        0/3.823e-05  eps:0.0002    

       Get at the values::

           In [16]: av,bv = ab.ox_dv.dvs[2].av, ab.ox_dv.dvs[2].bv   

        
       Change in meaning of fractions
       -------------------------------- 

       The transition from dv.py to qdv.py changes the meaning of the 
       warning fractions : it used to be fraction of compared elements with deviation exceeding cut
       it is now fraction of photons with aggregated max deviation exceeding cut

       This tends to reduce the fractions, but its more meaningful as single elements dont go wrong 
       on there own : usually its the photon that goes wrong with many elements deviating together.
       So a fraction of deviant photons is more pertinent.
       """
       label = self.LMT % ( idx, msg, sel, lcu[1], lcu[2] )
       assert len(dvmax) == 3 

       nitem = len(dv)
       npoi = len(sel.split())
       nelem = ndv*npoi*nitem

       if nelem>0:
           mx = dv.max()
           mn = dv.min()
           avg = dv.sum()/float(nelem)
           disc=[ dv[dv>dvmax[0]], dv[dv>dvmax[1]], dv[dv>dvmax[2]] ]
           ndisc = map(len, disc)     # elements, not items
           fdisc =  map(lambda _:float(_)/float(nelem), ndisc ) 
       else:
           mx = None
           mn = None
           avg = None
           ndisc = None
           fdisc = None
       pass

       self.tab = tab 
       self.label = label
       self.nitem = nitem
       self.nelem = nelem

       self.mx = mx
       self.mn = mn
       self.avg = avg
       self.ndisc = ndisc
       self.fdisc = fdisc
       self.ismax = False # set from DvTab
       
       self.dv = dv
       self.ndv = ndv
       self.lcu  = lcu
       self.dvmax = dvmax 
       self.msg = msg

       if self.mx > self.dvmax[2]:
           lev = Level.FromName("FATAL")
           lmsg = "  > dvmax[2] %.4f " % self.dvmax[2] 
       elif self.mx > self.dvmax[1]:
           lev = Level.FromName("ERROR")
           lmsg = "  > dvmax[1] %.4f " % self.dvmax[1] 
       elif self.mx > self.dvmax[0]:
           lev = Level.FromName("WARNING")
           lmsg = "  > dvmax[0] %.4f " % self.dvmax[0] 
       else:
           lev = Level.FromName("INFO")
           lmsg = ""
       pass
       self.fn_ = lev.fn_
       self.lev = lev
       self.lmsg = lmsg
 

   @classmethod  
   def columns(cls):
       cdesc = cls.CFMT % tuple(cls.CFMT_COLS)
       clabel = cls.clabel 
       return "%s : %s  " % (clabel, cdesc )

   def columns2(self, tdisc):
       cdesc2 = self.CFMT_CUTS % ("","",tdisc[0],tdisc[1],tdisc[2], self.dvmax[0], self.dvmax[1], self.dvmax[2], "", "", "") 
       return "%s : %s  " % (self.cblank, cdesc2 )


   def __repr__(self):
       if self.nelem>0:
           desc =  self.FMT % ( self.nitem, self.nelem, self.ndisc[0], self.ndisc[1], self.ndisc[2], self.fdisc[0], self.fdisc[1], self.fdisc[2], self.mx, self.mn, self.avg )
       else:
           desc = ""
       pass

       if self.ismax:
           #pdesc = self.fn_(desc)
           pdesc = underline_(self.fn_(desc))
       else: 
           pdesc = self.fn_(desc)
       pass
       return "%s : %s : %s : %s " % (self.label, pdesc, self.fn_("%20s"%self.lev.name), self.fn_(self.lmsg)  )




class QDVTab(object):
    """
    A much quicker way to analyse deviations, avoiding selection jumping 
    by inverting the loop ordering : a single deviation calc first 
    and then aggregate into categories. 

    Skips eg "SC AB RE" only relevant for non-aligned "accidental" comparisons

    """
    def __init__(self, name, ab, skips="" ):
        """
        :param name: ox_dv, rpost_dv, rpol_dv 
        :param ab:
        :param skips: 
        """
        log.debug("[ sel %s  " % (name) )
        self.name = name
        self.ab = ab 
        self.seqtab = ab.ahis   # SeqTable
        self.a = ab.a
        self.b = ab.b
        self.dirty = False
        self.skips = skips.split()
        #self.sli = slice(None)
        self.sli = slice(0,10)

        labels = self.seqtab.labels       # eg list of length 17 : ['TO BT BT SA', 'TO BR SA', ... ]

        cu = self.seqtab.cu               # eg with shape (17,3)  the 3 columns being (seqhis, a-count, b-count ) 
        assert len(labels) == len(cu)
        nsel = len(labels)

        self.labels = labels
        self.cu = cu 
        self.nsel = nsel 
        self._ndisc = None

        self.aligned = self.a.seqhis == self.b.seqhis 
        self.init_qdv()
        self.dvs = self.make_selection_dvs()
        self.findmax()
        log.debug("] %s " % name )

    def init_qdv(self):
        """
        Full array deviation comparisons and max aggregation, with no selection 
        """
        a = self.a
        b = self.b

        if self.name == "rpost_dv": 
            qdv = np.abs(a.rposta - b.rposta).max(axis=(1,2))   # maximal photon deviates 
            ndv = a.rposta.shape[a.rposta.ndim-1]               # 4, items in last dimension
        elif self.name == "rpol_dv": 
            qdv = np.abs(a.rpola - b.rpola).max(axis=(1,2))   
            ndv = 3  
        elif self.name == "ox_dv": 
            aox = a.ox[:,:3,:]  
            box = b.ox[:,:3,:]  
            sox = aox.shape
            assert len(sox) == 3
            qdv = np.abs(aox - box).max(axis=(1,2))
            ndv = sox[-2]*sox[-1]                      # 3*4 items in last two dimension
        else:
            assert self.name
        pass 
        self.qdv = qdv
        self.ndv = ndv

    def make_selection_dvs(self):
        """
        Slice and dice the full per photon aggregated max deviation into 
        the single line selections 
        """
        dvmax = self.dvmax
        dvs = []
        for i in range(self.nsel)[self.sli]:
            sel = self.labels[i]
            if self.is_skip(sel):
                continue
            pass
            lcu = self.cu[i]
            code = lcu[0]
            cqdv = self.cqdv(code)     # array of photon max deviations within selection 

            dv = QDV(self, i, sel, cqdv, self.ndv, lcu, dvmax )
            if len(dv.dv) == 0:
                log.debug("dv.dv empty for i:%d sel:%s lcu:%r " % ( i, sel, lcu))
            else:
                dvs.append(dv)
            pass
        pass
        return dvs
 

    def csel(self, code):
        """
        :param seqhis code:
        :return boolean selection array:
        """  
        return np.logical_and( self.aligned, self.a.seqhis == code ) 

    def cqdv(self, code):
        """
        :return array of deviations within selection: 
        """
        csel = self.csel( code )
        return self.qdv[np.where(csel)]


    def _get_float(self, att):
        return map(lambda dv:float(getattr(dv, att)), filter(None,self.dvs))

    maxdv = property(lambda self:self._get_float("mx"))  
    def _get_maxdvmax(self):  
        maxdv_ = self.maxdv
        return max(maxdv_) if len(maxdv_) > 0 else -1 
    maxdvmax   = property(_get_maxdvmax)  

    def _get_ndvp(self):
        """total number of deviant (ERROR or FATAL) photons"""
        return self.ndisc[1]  
    ndvp = property(_get_ndvp)


    def findmax(self):
        maxdv = map(lambda _:float(_.mx), self.dvs) 
        mmaxdv = max(maxdv) if len(maxdv) > 0 else -1
        for dv in self.dvs:
            if dv.mx == mmaxdv and dv.lev.level > Level.INFO:
                dv.ismax = True
            pass
        pass 

    level = property(lambda self:self.maxlevel.name)    

    def _get_maxlevel(self):
        """
        Overall level of the table : INFO, WARNING, ERROR or FATAL 
        based on the maximum level of the lines
        """
        levs = map(lambda dv:dv.lev.level, self.dvs)
        mxl = max(levs) if len(levs) > 0 else None
        return Level.FromLevel(mxl) if mxl is not None else None
    maxlevel = property(_get_maxlevel)  

    def _get_ndisc(self):
        if self._ndisc is None:
            ndisc = np.zeros(3, dtype=np.int)
            for dv in self.dvs:
                ndisc += dv.ndisc
            pass
            self._ndisc = ndisc
        pass
        return self._ndisc
    ndisc = property(_get_ndisc) 


    def _get_RC(self):
        maxlevel = self.maxlevel
        return 1 if maxlevel is not None and  maxlevel.level > Level.WARNING else 0
    RC = property(_get_RC)

    fdiscreps = property(lambda self:self._get_float("fdiscrep"))  
    def _get_fdiscmax(self):
        fdiscreps_ = self.fdiscreps
        return max(fdiscreps_) if len(fdiscreps_) > 0 else -1 
    fdiscmax   = property(_get_fdiscmax)  


    def _get_smry(self):
        return "%s fdiscmax:%s fdiscreps:%r maxdvmax:%s " % ( self.name, self.fdiscmax, self.fdiscreps, self.maxdvmax  )
    smry = property(_get_smry)

    def _get_brief(self):
        skips = " ".join(self.skips)
        gfmt_ = lambda _:"%.4f" % float(_) 

        if self.maxlevel is None:
            return "maxlevel None"   
        else:
            return "maxdvmax:%s  ndvp:%4d  level:%s  RC:%d       skip:%s" % ( gfmt_(self.maxdvmax), self.ndvp, self.maxlevel.fn_(self.maxlevel.name), self.RC,  skips )
        pass
  

    brief = property(_get_brief)

    def __repr__(self):
        if len(self.dvs) == 0:
            return "\n".join(["ab.%s" % self.name, "no dvs" ])
        else: 
            return "\n".join( ["ab.%s" % self.name, self.brief, self.dvs[0].columns2(self.ndisc), QDV.columns()] + map(repr, filter(None,self.dvs[self.sli]) ) + ["."] )
        pass

    def __getitem__(self, sli):
         self.sli = sli
         return self

    def is_skip(self, sel):
        """
        SC AB RE 
          see notes/issues/sc_ab_re_alignment.rst
        
        When comparing non-aligned events it is necessary to
        avoid accidental history alignment causing deviation fails
        by skipping any selection that includes SC AB or RE, assuming 
        reflect cheat is used.
        """
        sqs = sel.split() 
        skip = False
        for skp in self.skips:
             with_sk = skp in sqs
             if with_sk:
                 skip = True 
             pass
        pass
        return skip 

    def _get_dvmax(self): 
        """
        :return dvmax: list of three values corresponding to warn/error/fatal deviation cuts  

        rpost 
            extent*2/65535 is the size of shortnorm compression bins 
            so its the closest can expect values to get.

            TODO: currently the time cuts are not correct, a for "t" 
            should be using different domain 

        rpol 
             is extremely compressed, so bins are big : 1*2/255 
        ox
             not compressed stored as floats : other things will limit deviations

        """
        ab = self.ab
        if self.name == "rpost_dv": 
            eps = ab.fdom[0,0,3]*2.0/((0x1 << 16) - 1)
            dvmax = [eps, 1.5*eps, 2.0*eps] 
        elif self.name == "rpol_dv": 
            eps = 1.0*2.0/((0x1 << 8) - 1)
            dvmax = [eps, 1.5*eps, 2.0*eps] 
        elif self.name == "ox_dv":
            dvmax = ab.ok.pdvmax    ## adhoc input guess at appropriate cuts
        else:
            assert self.name 
        pass 
        return dvmax 
    dvmax = property(_get_dvmax)


    def _get_dvmaxt(self): 
        """ 
        ta 19::

            In [1]: ab.rpost_dv.dvmax
            Out[1]: [0.022827496757457846, 0.03424124513618677, 0.04565499351491569]

            In [2]: ab.rpost_dv.dvmaxt
            Out[2]: [0.0003043666242088853, 0.00045654993631332797, 0.0006087332484177706]

        Depends on the rule-of-thumb Opticks::setupTimeDomains but time cuts should be much 
        more stringent than position ones::

            In [3]: ab.fdom[0,0,3]
            Out[3]: 748.0

            In [4]: ab.fdom[1,0,1]
            Out[4]: 9.973333

            In [5]: ab.fdom[0,0,3]/ab.fdom[1,0,1]
            Out[5]: 75.0

        """
        ab = self.ab
        if self.name == "rpost_dv": 
            eps_t = ab.fdom[1,0,1]*2.0/((0x1 << 16) - 1)
            dvmaxt = [eps_t, 1.5*eps_t, 2.0*eps_t]
        else:
            assert self.name  
        pass
        return dvmaxt 
    dvmaxt = property(_get_dvmaxt)



if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    from opticks.ana.ab import AB
    ok = opticks_main()
    ab = AB(ok)
    #ab.dump()

    print(ab.ahis)

    a = ab.a 
    b = ab.b 

    qdv = QDVTab("rpost_dv", ab)
    print(qdv)



