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

class Dv(object):

   FMT  = "  %9d %9d/%9d:%6.3f  mx/mn/av %9.4g/%9.4g/%9.4g  eps:%g  "
   CFMT = "  %9s %9s/%9s:%6s  mx/mn/av %9s/%9s/%9s  eps:%s  "

   LMT  = " %0.4d %10s : %30s : %7d  %7d "   # labels seqhis line 
   CLMT = " %4s %10s : %30s : %7s  %7s " 

   clabel = CLMT % ( "idx", "msg", "sel", "lcu1", "lcu2" )

   def __init__(self, idx, sel, av, bv, lcu, eps, msg=""):
       """
       :param idx: unskipped orignal seqhis line index
       :param sel: single line selection eg 'TO BT BT SA'
       :param av: evt a values array within selection  
       :param bv: evt b values array within selection
       :param lcu: list of length 3 with (seqhis-bigint, a-count, b-count)
       :param eps: epsilon passed down from ab.eps


       Maximum and minimum 

       Access an Dv instance in ipython::

            In [12]: ab.ox_dv.dvs[2]
            Out[12]:  0002            :                 TO BT BR BT SA :     561      527  :          27       324/       14: 0.043  mx/mn/av   0.00238/        0/3.823e-05  eps:0.0002    


       Get at the values::

           In [16]: av,bv = ab.ox_dv.dvs[2].av, ab.ox_dv.dvs[2].bv   

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
        """
        :param name:
        :param seqtab:
        :param ab:
        :param skips: 
        """
        self.name = name
        self.seqtab = seqtab
        self.ab = ab 
        self.dirty = False
        self.eps = ab.dveps
        self.skips = skips.split()

        labels = self.seqtab.labels       # eg list of length 17 : ['TO BT BT SA', 'TO BR SA', ... ]

        cu = self.seqtab.cu               # eg with shape (17,3)  the 3 columns being (seqhis, a-count, b-count ) 
        assert len(labels) == len(cu)
        nsel = len(labels)

        dvs = []
        for i in range(nsel):
            sel = labels[i]

            if self.is_skip(sel):
                continue
            pass

            lcu = cu[i]
            assert len(lcu) == 3
            _, na, nb = lcu     

            ab.aselhis = sel              # set selection to just this history sequence
            assert True == ab.checkrec()  # checking that both evt a and b give same number of record points

            dv = self.dv_(i, sel, lcu)    # Dv instance comparing values within single line selection eg all 'TO BT BT SA' 
            dvs.append(dv)
        pass
        self.dvs = dvs 



    def dv_(self, i, sel, lcu):
        """
        :param i: unskipped orignal seqhis line index
        :param sel: seqhis label eg 'TO BT BT SA'
        :param lcu: count unique for single line eg [36045,  8794, 8794] 

        Accessing av and bv values within the active single line selection ab.aselhis, 
        and creating a Dv instance from them. 
        """ 
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
        return "%s fdiscmax:%s fdiscreps:%r maxdvmax:%s maxdv:%r  " % ( self.name, self.fdiscmax, self.fdiscreps, self.maxdvmax, self.maxdv  )
    smry = property(_get_smry)

    def _get_brief(self):
        skips = " ".join(self.skips)
        gfmt_ = lambda _:"%.4g" % float(_) 
        return "ab.%s maxdvmax:%s maxdv:%s  skip:%s" % ( self.name, gfmt_(self.maxdvmax), " ".join(map(gfmt_,self.maxdv)), skips )
    brief = property(_get_brief)

    def __repr__(self):
        return "\n".join( [self.brief, Dv.columns()] + map(repr, filter(None,self.dvs) ))



if __name__ == '__main__':
    pass
pass





