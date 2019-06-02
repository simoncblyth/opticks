#!usr/bin/env python
"""
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

class Dv(object):

   FMT  = "  %7d %7d/%7d:%6.3f  mx/mn/av %6.4g/%6.4g/%6.4g  eps:%g  "
   CFMT = "  %7s %7s/%7s:%6s  mx/mn/av %6s/%6s/%6s  eps:%s  "

   LMT  = " %0.4d %10s : %30s : %7d  %7d " 
   CLMT = " %4s %10s : %30s : %7s  %7s " 

   clabel = CLMT % ( "idx", "msg", "sel", "lcu1", "lcu2" )

   def __init__(self, idx, sel, av, bv, lcu, eps, msg=""):
       """
       :param idx: 
       :param sel: 
       :param av: 
       :param bv: 
       :param lcu: 
       :param eps: 
       """
       label = self.LMT % ( idx, msg, sel, lcu[1], lcu[2] )

       dv = np.abs( av - bv )
       nitem = len(dv)
       nelem = dv.size   

       if nelem>0:

           mx = dv.max()
           mn = dv.min()
           avg = dv.sum()/float(nelem)

           discrep = dv[dv>eps]
           ndiscrep = len(discrep)  # elements, not items
           fdiscrep = float(ndiscrep)/float(nelem) 
       else:
           mx = None
           mn = None
           avg = None
           ndiscrep = None
           fdiscrep = None
       pass

       self.label = label
       self.nitem = nitem
       self.nelem = nelem
       self.ndiscrep = ndiscrep
       self.fdiscrep = fdiscrep
       self.mx = mx 
       self.mn = mn 
       self.avg = avg 
       
       self.av = av
       self.bv = bv
       self.dv = dv
       self.lcu  = lcu
       self.eps = eps
       self.msg = msg

   @classmethod  
   def columns(cls):
       cdesc = cls.CFMT % ( "nitem", "nelem", "ndisc", "fdisc", "mx", "mn", "avg", "eps" )
       clabel = cls.clabel ; 
       return "%s : %s  " % (clabel, cdesc )


   def __repr__(self):
       if self.nelem>0:
           desc =  self.FMT % ( self.nitem, self.nelem, self.ndiscrep, self.fdiscrep, self.mx, self.mn, self.avg, self.eps )
       else:
           desc = ""
       pass
       return "%s : %s  " % (self.label, desc )


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

    def __init__(self, name, seqtab, ab, skips="SC AB RE" ):
        self.name = name
        self.seqtab = seqtab
        self.ab = ab 
        self.dirty = False
        self.eps = ab.dveps
        self.skips = skips.split()

        labels = self.seqtab.labels
        cu = self.seqtab.cu
        assert len(labels) == len(cu)
        nsel = len(labels)


        dvs = []
        for i in range(nsel):
            sel = labels[i]

            #if self.is_skip(sel):
            #    continue
            #pass

            lcu = cu[i]
            assert len(lcu) == 3
            _, na, nb = lcu 

            ab.aselhis = sel 
            ab.checkrec()

            dv = self.dv_(i, sel, lcu)
            dvs.append(dv)
        pass
        self.dvs = dvs 

    def dv_(self, i, sel, lcu):
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
        assert ab.a.sel == ab.b.sel 
        sel = ab.a.sel 
        dv = Dv(i, sel, av, bv, lcu, eps=self.eps)
        return dv if len(dv.dv) > 0 else None


    def _get_float(self, att):
        return map(lambda dv:float(getattr(dv, att)), filter(None,self.dvs))

    maxdv = property(lambda self:self._get_float("mx"))  
    def _get_maxdvmax(self):  
        maxdv_ = self.maxdv
        return max(maxdv_) if len(maxdv_) > 0 else -1 
    maxdvmax   = property(_get_maxdvmax)  

    fdiscreps = property(lambda self:self._get_float("fdiscrep"))  
    def _get_fdiscmax(self):
        fdiscreps_ = self.fdiscreps
        return max(fdiscreps_) if len(fdiscreps_) > 0 else -1 
    fdiscmax   = property(_get_fdiscmax)  


    def _get_smry(self):
        return "%s fdiscmax:%s fdiscreps:%r maxdvmax:%s maxdv:%r " % ( self.name, self.fdiscmax, self.fdiscreps, self.maxdvmax, self.maxdv )
    smry = property(_get_smry)

    def _get_brief(self):
        return "%s maxdvmax:%s maxdv:%r " % ( self.name, self.maxdvmax, self.maxdv )
    brief = property(_get_brief)

    def __repr__(self):
        return "\n".join( [self.brief, Dv.columns()] + map(repr, filter(None,self.dvs) ))



if __name__ == '__main__':
    pass
pass





