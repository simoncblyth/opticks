#!/usr/bin/env python
"""
intersect_boolean.py
=======================

This script prepares the boolean lookup tables 
packed into uint4 integers used by: 

* CUDA/OptiX header oxrap/cu/intersect_boolean.h 
* python mimmick of that dev/csg/slavish.py 

It also dumps the boolean lookup tables allowing 
checking of the original acts table and the acloser, 
bcloser variants and the packed lookup 
equivalents.


It imports python classes generated from C enum headers::

    cd ~/opticks/sysrap
    c_enums_to_python.py OpticksCSG.h > OpticksCSG.py 

    cd ~/opticks/optixrap/cu
    c_enums_to_python.py boolean_solid.h > boolean_solid.py 


"""
import sys, os, datetime, collections, numpy as np
ndict = lambda:collections.defaultdict(ndict) 

# generated enum classes
from opticks.sysrap.OpticksCSG import CSG_
from opticks.optixrap.cu.boolean_solid import Act_, State_, CTRL_
from opticks.optixrap.cu.boolean_solid import Union_, Difference_, Intersection_
from opticks.optixrap.cu.boolean_solid import ACloser_Union_, ACloser_Difference_, ACloser_Intersection_
from opticks.optixrap.cu.boolean_solid import BCloser_Union_, BCloser_Difference_, BCloser_Intersection_

def _init_table(pfx="", vrep_=lambda v:Act_.descmask(v)):
    """
    Build CSG boolean lookup tables from the python translation of the C enums
    """
    tab = ndict()
    Case_ = lambda s:s[0].upper() + s[1:].lower() 
    for op in [CSG_.UNION, CSG_.INTERSECTION, CSG_.DIFFERENCE]:
        kln = Case_(CSG_.desc(op)) + "_"
        kls = sys.modules[__name__].__dict__.get(pfx+kln)
        #print kls.__name__
        for k,v in kls.enum():  
            kk = k.split("_")
            assert len(kk) == 2
            al, bl = kk
            a = State_.fromdesc(al[:-1])
            b = State_.fromdesc(bl[:-1])
            #print " %d:%5s   %d:%5s :  %s " % ( a, State_.desc(a), b, State_.desc(b),  vrep_(v) ) 
            tab[op][a][b] = v
        pass
    pass
    return tab

_act_table = _init_table(vrep_=lambda v:Act_.descmask(v))
_acloser_table = _init_table("ACloser_", vrep_=lambda v:CTRL_.desc(v))
_bcloser_table = _init_table("BCloser_", vrep_=lambda v:CTRL_.desc(v))

def _pack_ctrl_tables():
    """
    Note that 32 bits can only fit 8*4 bits, but there are 
    9 values in the (stateA, stateB) table.
    Thus the (Miss,Miss) corner which always yields Miss, 
    is trimmed and this is special cased at lookup.    
    """
    assert (State_.Enter, State_.Exit, State_.Miss) == (0,1,2) 
    assert (CSG_.UNION, CSG_.INTERSECTION, CSG_.DIFFERENCE) == (1,2,3)

    pak = np.zeros( (2, 4), dtype=np.uint32 ) 
    for op in [CSG_.UNION,CSG_.INTERSECTION,CSG_.DIFFERENCE]:
        for a in [State_.Enter, State_.Exit, State_.Miss]:
            for b in [State_.Enter, State_.Exit, State_.Miss]:
                ctrlACloser = _acloser_table[op][a][b] 
                ctrlBCloser = _bcloser_table[op][a][b] 
                off = (a*3+b)
                if off < 8:
                    pak[0,op - CSG_.UNION] |=  ctrlACloser << (off*4) 
                    pak[1,op - CSG_.UNION] |=  ctrlBCloser << (off*4) 
                pass
            pass
        pass               
    pass
    return pak

_pak = _pack_ctrl_tables() 
_pak0 = np.array([[0x22121141, 0x14014, 0x141141, 0x0],
                  [0x22115122, 0x22055, 0x133155, 0x0]], dtype=np.uint32)

assert np.all(_pak == _pak0), _pak

def pak_lookup(op, a, b, ACloser):
    ac = 1-int(ACloser)  # 0:ACloser 1:BCloser
    lut0 = _pak0[ac][op-CSG_.UNION]
    lut  = _pak[ac][op-CSG_.UNION]
    assert lut0 == lut
    off = (a*3+b)
    return ((lut >> (off*4)) & 0xf) if off < 8 else CTRL_.RETURN_MISS

class OptiXPak(object):
    def __init__(self, a ):
        self.a = a 
    def __repr__(self):
        x8fmt_ = lambda _:"0x%.8x" % _
        fmt_ = lambda a:", ".join(map(x8fmt_, a)) 
        fmt = "rtDeclareVariable(uint4, packed_boolean_lut_%s, , ) = { %s } ;"
        now = datetime.datetime.now().strftime("%c")
        return "\n".join([
             "// generated from %s by %s on %s " % (os.getcwd(), sys.argv[0],  now ),
             fmt % ("ACloser", fmt_(_pak[0]) ),
             fmt % ("BCloser", fmt_(_pak[1]) )])


def boolean_table(op, a, b):
    assert op in [CSG_.UNION, CSG_.INTERSECTION, CSG_.DIFFERENCE]
    assert a in [State_.Enter, State_.Exit, State_.Miss]
    assert b in [State_.Enter, State_.Exit, State_.Miss]
    return _act_table[op][a][b]


def boolean_decision(acts, ACloser):
    """
    :param acts: actions mask 
    :return act:

    Decide on the single action to take from
    potentially multiple in the actions mask 
    based on which intersection is closer 
    """
    if (acts & Act_.ReturnMiss):
        act = Act_.ReturnMiss

    elif (acts & Act_.ReturnA):
        act = Act_.ReturnA
    elif (acts & Act_.ReturnAIfCloser) and ACloser:
        act = Act_.ReturnAIfCloser
    elif (acts & Act_.ReturnAIfFarther) and (not ACloser):
        act = Act_.ReturnAIfFarther

    elif (acts & Act_.ReturnB):
        act = Act_.ReturnB
    elif (acts & Act_.ReturnBIfCloser) and (not ACloser):
        act = Act_.ReturnBIfCloser
    elif (acts & Act_.ReturnFlipBIfCloser) and (not ACloser):
        act = Act_.ReturnFlipBIfCloser
    elif (acts & Act_.ReturnBIfFarther) and ACloser:
        act = Act_.ReturnBIfFarther
    
    elif (acts & Act_.AdvanceAAndLoop):
        act = Act_.AdvanceAAndLoop
    elif (acts & Act_.AdvanceAAndLoopIfCloser) and ACloser:
        act = Act_.AdvanceAAndLoopIfCloser

    elif (acts & Act_.AdvanceBAndLoop):
        act = Act_.AdvanceBAndLoop
    elif (acts & Act_.AdvanceBAndLoopIfCloser) and (not ACloser):
        act = Act_.AdvanceBAndLoopIfCloser
    else:
        assert 0, (acts, Act_.descmask(acts), ACloser)
    pass 
    return act 


def compare_tables():
    """
    The _acloser and _bcloser tables avoid branchy decision
    making at runtime of the original _act_table, 
    instead simply lookup the ctrl from the appropriate table.
    """
    fmt = "   %7s  %7s  ->   %35s       %35s         %35s " 
    for op in [CSG_.UNION, CSG_.INTERSECTION, CSG_.DIFFERENCE]:
        print "%s : boolean_table " % CSG_.desc(op)
        print fmt % ( "","","", "tA < tB", "tB < tA" )
        for a in [State_.Enter, State_.Exit, State_.Miss]:
            for b in [State_.Enter, State_.Exit, State_.Miss]:

                acts = boolean_table(op, a, b) 
                actACloser = boolean_decision(acts, ACloser=True )
                actBCloser = boolean_decision(acts, ACloser=False )

                ctrlACloser = _acloser_table[op][a][b] 
                ctrlACloser_pak = pak_lookup(op,a,b,ACloser=True)
                assert ctrlACloser == ctrlACloser_pak, (ctrlACloser, ctrlACloser_pak)

                ctrlBCloser = _bcloser_table[op][a][b] 
                ctrlBCloser_pak = pak_lookup(op,a,b,ACloser=False)
                assert ctrlBCloser == ctrlBCloser_pak, (ctrlBCloser, ctrlBCloser_pak)
 
                aextra = " " + CTRL_.desc(ctrlACloser)  
                bextra = " " + CTRL_.desc(ctrlBCloser)  

                print fmt % ( State_.desc(a), State_.desc(b), Act_.descmask(acts), Act_.descmask(actACloser)+aextra, Act_.descmask(actBCloser)+bextra )
            pass
        pass
    pass
    print OptiXPak(_pak0)




if __name__ == '__main__':
    pass
    compare_tables()


