#!/usr/bin/env python
"""

dv.py
======


Non-aligned deviation checking
---------------------------------

::

    tp() { tboolean-;PROXYLV=18 tboolean-proxy-ip $* ; }

    ab.rpost_dv maxdvmax: 0.02283 maxdv: 0.02283        0  0.02283  skip:SC AB RE
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA :    8794     8794  :        7710    123360/       17: 0.000  mx/mn/av   0.02283/        0/1.705e-06  eps:0.0002    
     0001            :                       TO BR SA :     580      617  :          33       396/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :     561      527  :          27       540/        1: 0.002  mx/mn/av   0.02283/        0/4.227e-05  eps:0.0002    
    ab.rpol_dv maxdvmax:       0 maxdv:       0        0        0  skip:SC AB RE
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA :    8794     8794  :        7710     92520/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0001            :                       TO BR SA :     580      617  :          33       297/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :     561      527  :          27       405/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
    ab.ox_dv maxdvmax: 0.00238 maxdv:0.0001221        0  0.00238  skip:SC AB RE
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA :    8794     8794  :        7710     92520/        0: 0.000  mx/mn/av 0.0001221/        0/3.652e-06  eps:0.0002    
     0001            :                       TO BR SA :     580      617  :          33       396/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :     561      527  :          27       324/       14: 0.043  mx/mn/av   0.00238/        0/3.823e-05  eps:0.0002    



Observations:

* nitem much lower than photon counts when you have a BR in history 
* "--reflectcheat" is not enabled by default : enabling it can increase the stats of the comparison


Non-random-aligned deviation checking relies on 

1. same input photons for both simulations A and B
2. "accidental" history alignment between A and B : helped by BR reflectcheat 

Need to find some corroboration:

    For histories like "TO BT BT SA" what happens is purely geometry, so 
    for those photons in A and B that follow this history can directly compare.
    But some fraction of BT in simulation A will be BR in the simulation B as 
    the random numbers are not aligned.  To reduce this the reflectcheat technique 
    is used to decide which fraction of sample to reflect and which to transmit 
    based on the ratio of record_id to total   


Examining the deviation::

    In [15]: ab.rpost_dv
    Out[15]: 
    rpost_dv
     0000            :                          TO SA :   55321    55303  :     55249/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
     0001            :                    TO BT BT SA :   39222    39231  :     34492/      8: 0.000  mx/mn/av 0.0138/0.0000/3.192e-06    
     0002            :                       TO BR SA :    2768     2814  :       188/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
     0003            :                 TO BT BR BT SA :    2425     2369  :       125/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
     0004            :              TO BT BR BR BT SA :     151      142  :         1/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    

    In [12]: dv = ab.rpost_dv.dvs[1].dv
    In [16]: av = ab.rpost_dv.dvs[1].av
    In [17]: bv = ab.rpost_dv.dvs[1].bv

    In [14]: np.where( dv > 0 )
    Out[14]: 
    (
        A([ 8019,  8019,  8019,  8019, 13879, 13879, 13879, 13879]),
        A([0, 1, 2, 3, 0, 1, 2, 3]),
        A([1, 1, 1, 1, 0, 0, 0, 0])
    )

    In [18]: wdv = np.where( dv > 0 )

    In [19]: av[wdv]
    Out[19]: 
    A([  30.4181,   30.4181,   30.4181,   30.4181,  116.2219,  116.2219,  116.2219,  116.2219])

    In [20]: bv[wdv]
    Out[20]: 
    A([  30.4319,   30.4319,   30.4319,   30.4319,  116.2357,  116.2357,  116.2357,  116.2357])

    In [21]: av[wdv] - bv[wdv]
    Out[21]: 
    A([-0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138])


"""
import os, sys, logging, numpy as np
from opticks.ana.log import fatal_, error_, warning_, info_, debug_
from opticks.ana.log import underline_, blink_ 
log = logging.getLogger(__name__)

assert 0, "dont use this : use qdv.py : its much faster "

class Level(object):
    FATAL = 20
    ERROR = 10
    WARNING = 0 
    INFO = -10
    DEBUG = -20

    level2name = { FATAL:"FATAL", ERROR:"ERROR", WARNING:"WARNING", INFO:"INFO", DEBUG:"DEBUG" }
    name2level = { "FATAL":FATAL, "ERROR":ERROR, "WARNING":WARNING, "INFO":INFO, "DEBUG":DEBUG  }
    level2func = { FATAL:fatal_, ERROR:error_, WARNING:warning_, INFO:info_, DEBUG:debug_ }


    @classmethod
    def FromName(cls, name):
        level = cls.name2level[name] 
        return cls(name, level) 
    @classmethod
    def FromLevel(cls, level):
        name = cls.level2name[level] 
        return cls(name, level) 

    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.fn_ = self.level2func[level]


class Dv(object):

   FMT  =       "  %9d %9d : %5d %5d %5d : %6.4f %6.4f %6.4f : %9.4f %9.4f %9.4f  "
   CFMT =       "  %9s %9s : %5s %5s %5s : %6s %6s %6s : %9s %9s %9s    "
   CFMT_CUTS =  "  %9s %9s : %5s %5s %5s : %6.4f %6.4f %6.4f : %9s %9s %9s    "
   CFMT_COLS =  "nitem nelem nwar nerr nfat fwar ferr ffat mx mn avg".split()

   LMT  = " %0.4d %10s : %30s : %7d  %7d "   # labels seqhis line 
   CLMT = " %4s %10s : %30s : %7s  %7s " 

   clabel = CLMT % ( "idx", "msg", "sel", "lcu1", "lcu2" )
   cblank = CLMT % ( "", "", "", "", "" )


   def __init__(self, tab, idx, sel, av, bv, lcu, dvmax, msg=""):
       """
       :param tab: DvTab instance 
       :param idx: unskipped orignal seqhis line index
       :param sel: single line selection eg 'TO BT BT SA'
       :param av: evt a values array within selection  
       :param bv: evt b values array within selection
       :param lcu: list of length 3 with (seqhis-bigint, a-count, b-count)
       :param dvmax: triplet of floats for warn/error/fatal deviation levels

       Access an Dv instance in ipython::

            In [12]: ab.ox_dv.dvs[2]
            Out[12]:  0002            :                 TO BT BR BT SA :     561      527  :          27       324/       14: 0.043  mx/mn/av   0.00238/        0/3.823e-05  eps:0.0002    

       Get at the values::

           In [16]: av,bv = ab.ox_dv.dvs[2].av, ab.ox_dv.dvs[2].bv   

       """
       label = self.LMT % ( idx, msg, sel, lcu[1], lcu[2] )
       assert len(dvmax) == 3 

       dv = np.abs( av - bv )
       nitem = len(dv)
       nelem = dv.size   

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
       self.ndisc = ndisc
       self.fdisc = fdisc
       self.mx = mx 
       self.mn = mn 
       self.avg = avg 
       self.ismax = False # set from DvTab
       
       self.av = av
       self.bv = bv
       self.dv = dv
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

   def columns2(self):
       cdesc2 = self.CFMT_CUTS % ("","","","","", self.dvmax[0], self.dvmax[1], self.dvmax[2], "", "", "") 
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


class DvTab(object):
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

    def __init__(self, name, seqtab, ab, skips="SC AB RE", selbase=None ):
        """
        :param name: ox_dv, rpost_dv, rpol_dv 
        :param seqtab: ab.ahis SeqTable
        :param ab:
        :param skips: 
        :param selbase: either None or "ALIGN" 

        """
        log.info("[ sel %s selbase %s  " % (name, selbase) )
        self.name = name
        self.seqtab = seqtab
        self.ab = ab 
        self.dirty = False
        self.skips = skips.split()
        #self.sli = slice(None)
        self.sli = slice(0,10)
        self.selbase = selbase

        labels = self.seqtab.labels       # eg list of length 17 : ['TO BT BT SA', 'TO BR SA', ... ]

        cu = self.seqtab.cu               # eg with shape (17,3)  the 3 columns being (seqhis, a-count, b-count ) 
        assert len(labels) == len(cu)
        nsel = len(labels)

        self.labels = labels
        self.cu = cu 
        self.nsel = nsel 

        ab.aselhis = selbase

        self.dvs = self.make_dvs_slowly()

        self.findmax()
        log.info("] %s " % name )


    def make_dvs_slowly(self):
        """
        Changing selection for every line like this 
        is too slow for more than 100k photons
        """
        cu = self.cu
        ab = self.ab 
        dvs = []
        for i in range(self.nsel)[self.sli]:
            sel = self.labels[i]

            if self.is_skip(sel):
                continue
            pass

            lcu = cu[i]
            assert len(lcu) == 3
            _, na, nb = lcu     

            ab.aselhis = sel              # set selection to just this history sequence
            assert True == ab.checkrec()  # checking that both evt a and b give same number of record points

            dv = self.dv_(i, sel, lcu)    # Dv instance comparing values within single line selection eg all 'TO BT BT SA' 

            if dv is None:
                log.debug("dv None for i:%d sel:%s lcu:%r " % ( i, sel, lcu))
            else:
                dvs.append(dv)
            pass

            ab.aselhis = self.selbase
        pass
        return dvs



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


    def dv_(self, i, sel, lcu):
        """
        :param i: unskipped orignal seqhis line index
        :param sel: seqhis label eg 'TO BT BT SA'
        :param lcu: count unique for single line eg [36045,  8794, 8794] 

        Accessing av and bv values within the active single line selection ab.aselhis, 
        and creating a Dv instance from them. 
        """ 
        log.info("[  %s " % sel )
 
        ab = self.ab
        if self.name == "rpost_dv": 
            av = ab.a.rpost()
            bv = ab.b.rpost()
        elif self.name == "rpol_dv": 
            av = ab.a.rpol()
            bv = ab.b.rpol()
        elif self.name == "ox_dv": 
            av = ab.a.ox[:,:3,:]
            bv = ab.b.ox[:,:3,:]
        else:
            assert self.name
        pass 
        dvmax = self.dvmax

        assert ab.a.sel == ab.b.sel 
        sel = ab.a.sel 
        dv = Dv(self, i, sel, av, bv, lcu, dvmax )
        ret = dv if len(dv.dv) > 0 else None
        log.info("]")
        return ret 

    def _get_float(self, att):
        return map(lambda dv:float(getattr(dv, att)), filter(None,self.dvs))

    maxdv = property(lambda self:self._get_float("mx"))  
    def _get_maxdvmax(self):  
        maxdv_ = self.maxdv
        return max(maxdv_) if len(maxdv_) > 0 else -1 
    maxdvmax   = property(_get_maxdvmax)  

    def findmax(self):
        maxdv = map(lambda _:float(_.mx), self.dvs) 
        mmaxdv = max(maxdv) if len(maxdv) > 0 else -1
        for dv in self.dvs:
            if dv.mx == mmaxdv and dv.lev.level > Level.INFO:
                dv.ismax = True
            pass
        pass 

    def _get_maxlevel(self):
        """
        Overall level of the table : INFO, WARNING, ERROR or FATAL 
        based on the maximum level of the lines
        """
        levs = map(lambda dv:dv.lev.level, self.dvs)
        mxl = max(levs) if len(levs) > 0 else None
        return Level.FromLevel(mxl) if mxl is not None else None
    maxlevel = property(_get_maxlevel)  

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
            return "maxdvmax:%s  level:%s  RC:%d       skip:%s" % ( gfmt_(self.maxdvmax), self.maxlevel.fn_(self.maxlevel.name), self.RC,  skips )
        pass
  

    brief = property(_get_brief)

    def __repr__(self):
        if len(self.dvs) == 0:
            return "\n".join(["ab.%s" % self.name, "no dvs" ])
        else: 
            return "\n".join( ["ab.%s" % self.name, self.brief, self.dvs[0].columns2(), Dv.columns()] + map(repr, filter(None,self.dvs[self.sli]) ) + ["."] )
        pass

    def __getitem__(self, sli):
         self.sli = sli
         return self



if __name__ == '__main__':
    pass
pass





