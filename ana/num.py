#!/usr/bin/env python
"""
"""

import numpy as np
import os, logging, re
from collections import OrderedDict as odict 

log = logging.getLogger(__name__) 

class Num(object):
    """
    ::

        In [19]: np.power( 10, np.arange(10) )
        Out[19]: array([         1,         10,        100,       1000,      10000,     100000,    1000000,   10000000,  100000000, 1000000000])

        In [69]: map(Num.String, a )
        Out[69]: ['1', '10', '100', '1k', '10k', '100k', '1M', '10M', '100M', '1B']

    """
    Ptn0 = re.compile("^(?P<nz>[\d]+?)(?P<zz>[0]*)$")   # less greedy frount ints, before contiguous zeros
    Ptn = re.compile("^(?P<num>\d+)(?P<unit>[kM]*)$")
    Units = { "":1, "k":1000, "M":1000000 } 
    Pow10 = None    

    @classmethod
    def OrigShape(cls, a):
        return a.origshape() if hasattr(a, 'origshape') else "-"

    @classmethod
    def Init(cls):
        r = odict()
        r[            "0"] = (0,"")  
        r[            "1"] = (1,"")  
        r[           "10"] = (10,"")
        r[          "100"] = (100,"")
        r[        "1,000"] = (1,"k")
        r[       "10,000"] = (10,"k")
        r[      "100,000"] = (100,"k")
        r[    "1,000,000"] = (1,"M")
        r[   "10,000,000"] = (10,"M")
        r[  "100,000,000"] = (100,"M")
        r["1,000,000,000"] = (1,"B")

        d = odict()
        for k,v in r.items():
            d[int(k.replace(",",""))] = v
        pass
        return d 

    @classmethod
    def Int(cls, s):
        """
        :param s: string representation of integer eg "1","10","100","1k","10k","100k","1M",...
        :return i: the integer
        """
        if s.find(",") > -1:
            ret = tuple(map(cls.Int, s.split(",")))
        else: 
            m = cls.Ptn.match(s)
            if m is None:
                return int(s)
            pass
            d = m.groupdict() 
            n = int(d["num"])      
            u = cls.Units[d["unit"]]      
            i = n*u  
            ret = i
        pass
        return ret         
     

    @classmethod
    def String(cls, i):
        """
        :param i: integer or tuple of integers
        :return s: summary string 

        Summarize large power of 10 numbers to make them easy to read without counting zeros 
        """ 
        if cls.Pow10 is None: cls.Pow10 = cls.Init()

        if type(i) in [ int, np.int64, np.int32 ]:

            m0 = cls.Ptn0.match(str(i))
            d0 = m0.groupdict() if m0 else None
            if d0: 
                nz = int(d0["nz"])      # integers at front of contiguous zeros
                zz = int("1"+d0["zz"])  # pull out contiguous zeros to the right 
                n,u = cls.Pow10[zz] 
                nnz = n*nz 
                ret = "%d%s" % (nnz, u)
            else:
                ret = str(i)
            pass
            return ret   

        elif type(i) is tuple:
            ret =  ",".join( map(lambda _:cls.String(_), list(i) ))
        elif type(i) is type(None):
            ret = "-" 
        else:
            assert 0, (i, type(i)) 
            ret = "-"
        pass    
        return ret 



slice_ = lambda s:slice(*map(lambda _:Num.Int(_) if len(_) > 0 else None,s.split(":")))                     # string to slice
_slice = lambda s:":".join(map(lambda _:Num.String(_) if not _ is None else "", (s.start,s.stop,s.step)))   # slice to string


if __name__ == '__main__':

    for i in [(20000000, 4, 4), 200000000, 2, 20, 30,300,3000,30000,100000,1000000,10000000,101000]:
        s = Num.String(i)
        i2 = Num.Int(s)
        print(" i %20r Num.String %20r Num.Int %10r   " % ( i, s, i2 )) 
        assert i == i2, ( i, s, i2)
    pass



