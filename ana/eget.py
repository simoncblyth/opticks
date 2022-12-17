#!/usr/bin/env python

import os, numpy as np

efloat_ = lambda ekey, fallback:float(os.environ.get(ekey,fallback))
efloatlist_ = lambda ekey,fallback="":list(map(float, filter(None, os.environ.get(ekey,fallback).split(","))))
efloatarray_ = lambda ekey,fallback="":np.array( efloatlist_(ekey, fallback)) 

def eint_(ekey, fallback):
    """
    A blank value is special cased to return zero 
    """
    val = os.environ.get(ekey, fallback)
    if val == "":
        val = "0"
    pass
    return int(val)


def eintlist_(ekey, fallback):
    """ 
    empty string envvar yields None
    """
    slis = os.environ.get(ekey,fallback)
    if slis is None or len(slis) == 0: return None
    slis = slis.split(",")
    return list(map(int, filter(None, slis)))


def intlist_(s):
    """
    :param s: string of form 10:21,30,31,32,40:51
    :return ii: list of integers 

    Uses np.arange one past the end convention, so "10:20" would not include 20::

        In [10]: np.arange(10,20)
        Out[10]: array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    """
    ii = []
    if not s is None and len(s) > 0:
        ee = s.split(",")
        for e in ee:
            if ":" in e:
                ab = e.split(":")
                assert len(ab) == 2 
                a, b = list(map(int,ab))
                for i in range(a,b):
                    ii.append(i)
                pass
            else:
                ii.append(int(e))
            pass
        pass 
    pass
    return ii

def intarray_(s):
    ii = intlist_(s)
    return np.array(ii, dtype=np.int32 )

def eintarray_(ekey, fallback=None):
    s = os.environ.get(ekey,fallback)
    return intarray_(s) 


def elook_epsilon_(epsilon, ekey="LOOK"):
    if not ekey in os.environ:
         return None
    pass
    look = efloatlist_(ekey, "0,0,0")
    ce = np.zeros( (4,), dtype=np.float32 )     
    ce[:3] = look 
    ce[3] = epsilon
    return ce 

def elookce_(extent="10.", ekey="LOOK"):
    """
    :param extent: float, is overridden by LOOKCE envvar
    :param ekey: typically "LOOK" specifting envvar that contains look coordinates eg "10.5,10.5,10.5" 
    """
    if not ekey in os.environ:
         return None
    pass
    look = efloatlist_(ekey, "0,0,0")
    extents = efloatlist_("LOOKCE", extent)
    ce = np.zeros( (len(extents), 4,), dtype=np.float32 )     
    for i in range(len(extents)):
        ce[i,:3] = look 
        ce[i, 3] = extents[i]
    pass 
    return ce 


if __name__ == '__main__':

    tmin0 = efloat_("TMIN",0.5)
    tmin1 = efloat_("TMIN","0.5")
    assert tmin0 == tmin1


    eye0 = efloatlist_("EYE", "1,-1,1")
    print("%10.4f %10.4f %10.4f " % tuple(eye0) )

    
