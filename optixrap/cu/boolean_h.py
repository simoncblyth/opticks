#!/usr/bin/env python
"""
The boolean_solid_h module is manually 
generated from the C enum header with::

    simon:cu blyth$ c_enums_to_python.py boolean-solid.h > boolean_solid_h.py    

This module uses the enum values and key strings from that generated module 
to provide the functions needed for the python prototype. 
In this way the critical parts of the python boolean state tables 
are effectively generated from the C enum.

"""

import os, sys, datetime
import numpy as np
import boolean_solid_h as h 
enum_kv = filter(lambda kv:kv[0][0] != "_",h.__dict__.items())

from boolean_solid_h import ReturnMiss, ReturnA, ReturnAIfCloser, ReturnAIfFarther
from boolean_solid_h import ReturnB, ReturnBIfCloser, ReturnBIfFarther, ReturnFlipBIfCloser
from boolean_solid_h import AdvanceAAndLoop, AdvanceAAndLoopIfCloser
from boolean_solid_h import AdvanceBAndLoop, AdvanceBAndLoopIfCloser

from boolean_solid_h import Enter, Exit, Miss
from boolean_solid_h import CTRL_RETURN_MISS

## boolean operation, hmm these need to match okc- OpticksShape.h CSG_UNION, CSG_INTERSECTION, CSG_DIFFERENCE
UNION  = 0
INTERSECTION = 1
DIFFERENCE = 2
_op_label = { UNION:"Union", INTERSECTION:"Intersection", DIFFERENCE:"Difference" }
def desc_op(op):
    return _op_label[op]

## intersection state of a ray with a primitive

_st_label = { Enter:"Enter", Exit:"Exit", Miss:"Miss" }
_st = dict(zip(_st_label.values(), _st_label.keys()))


def desc_state(st):
    return _st_label(st)

_ctrl = dict(filter(lambda _:_[0].startswith("CTRL"), enum_kv))
_ctrl_label = dict(zip(_ctrl.values(), _ctrl.keys()))

def desc_ctrl(ct):
    return _ctrl_label[ct]


ffs_ = lambda x:(x&-x).bit_length()

def ffs(x):
    """  
    http://stackoverflow.com/questions/5520655/return-index-of-least-significant-bit-in-python
    Pythons 2.7 and 3.1 and up

    Hmm works with very big numbers in python, restrict to 32 for C::

        map(ffs_, [0x1 << n for n in range(100)]) == range(100)

        In [60]: [0x1 << n for n in range(16)]
        Out[60]: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

        In [80]: map(ffs_,[0x1 << n for n in range(16)])
        Out[80]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    """
    return ffs_(x)


_act = {}        # label -> 0x1 << n
_act_label = {}  # 0x1 << n --> label 
_act_index = {}  # 0x1 << n --> n 

def _init_acts(ekv):
    """
    # from the generated header, values not starting with "_" and 
    # with keys without "_" are the action enum values, like ReturnMiss = 0x1 << 0
    """
    global _act
    global _act_label
    global _act_index

    print "_init_acts"
    for k,v in filter(lambda kv:kv[0].find("_") == -1, ekv):
        if k.startswith("Return") or k.startswith("Advance"):
            print "%30s : %s " % (k, v )
            _act[k] = v 
            _act_label[v] = k 
            _act_index[v] = ffs(v)   # CAUTION THIS WAS OFF-BY-ONE
        pass
_init_acts(enum_kv)



def _init_enum(ekv, pfxs=["CTRL,ERROR"]):
    enum = {}
    print "_init_enum"
    for k,v in ekv:
        print "%30s : %s " % (k, v )
        if k.find("_") == -1:continue

        p = k.index("_") 

        kpfx = k[:p]
        if kpfx in pfxs:
            if kpfx not in enum:
                enum[kpfx] = {} 
            pass
            enum[kpfx][kpfx] = v 
        pass
        return enum
    pass
_enum = _init_enum(enum_kv)

    





def desc_acts(acts):
    l = []
    for iact,label in _act_label.items():
        if acts & iact: l.append(label)
    pass
    return " ".join(l)

def act_index(v):
    """
    v (0x1 << n)  --> n
    """
    return _act_index[v]

def dump_acts():
    print "dump_acts"
    for v in sorted(_act_label.keys()):
        print " %6d : %2d : %30s " % ( v, act_index(v), desc_acts(v) )



def _init_table(ekv, pfx=""):
    if len(pfx)>0:assert pfx[-1] == "_", "pfx must end with _"
    tab = {}
    for op in [UNION, INTERSECTION, DIFFERENCE]:
        ol = desc_op(op)
        for k,v in sorted(filter(lambda kv:kv[0].startswith(pfx+ol),ekv), key=lambda kv:kv[1]):
            #print "%30s : %s " % (k,v )

            elem = k.split("_")

            if len(pfx) == 0:
                ol2,al,bl = elem
            else:
                chk, ol2, al,bl = elem
                assert chk == pfx[:-1]
            pass
            assert ol2 == ol
            assert al[:-1] in ["Enter","Exit","Miss"], al
            assert bl[:-1] in ["Enter","Exit","Miss"], bl
 
            a = _st[al[:-1]]
            b = _st[bl[:-1]]

            if op not in tab:        
                tab[op] = {}
            pass
            if a not in tab[op]:
                tab[op][a] = {}
            pass
            if b not in tab[op][a]:
                tab[op][a][b] = {}
            pass
            tab[op][a][b] = v
        pass
    return tab
_table = _init_table(enum_kv)


_acloser = _init_table(enum_kv, "ACloser_")
_bcloser = _init_table(enum_kv, "BCloser_")




def boolean_table(op, a, b):
    assert op in [UNION, INTERSECTION, DIFFERENCE], op
    assert a in [Enter,Exit,Miss], a
    assert b in [Enter,Exit,Miss], b
    return _table[op][a][b]


def boolean_decision(acts, ACloser):
    """
    :param acts: actions mask 
    :param tA: distance to first intersect of ray with A
    :param tB: distance to first intersect of ray with B
    :return act:

    Decide on the single action to take from
    potentially multiple in the actions mask 
    based on which intersection is closer 
    """


    if (acts & ReturnMiss):
        act = ReturnMiss

    elif (acts & ReturnA):
        act = ReturnA
    elif (acts & ReturnAIfCloser) and ACloser:
        act = ReturnAIfCloser
    elif (acts & ReturnAIfFarther) and (not ACloser):
        act = ReturnAIfFarther

    elif (acts & ReturnB):
        act = ReturnB
    elif (acts & ReturnBIfCloser) and (not ACloser):
        act = ReturnBIfCloser
    elif (acts & ReturnFlipBIfCloser) and (not ACloser):
        act = ReturnFlipBIfCloser
    elif (acts & ReturnBIfFarther) and ACloser:
        act = ReturnBIfFarther
    
    elif (acts & AdvanceAAndLoop):
        act = AdvanceAAndLoop
    elif (acts & AdvanceAAndLoopIfCloser) and ACloser:
        act = AdvanceAAndLoopIfCloser

    elif (acts & AdvanceBAndLoop):
        act = AdvanceBAndLoop
    elif (acts & AdvanceBAndLoopIfCloser) and (not ACloser):
        act = AdvanceBAndLoopIfCloser
    else:
        assert 0, (acts, desc_acts(acts), ACloser)
    pass 
    return act 


def _pack_ctrl_tables():
    """
    """
    ops = [UNION,INTERSECTION,DIFFERENCE]
    pak = np.zeros( (2, 4), dtype=np.uint32 ) 

    for op in [UNION,INTERSECTION,DIFFERENCE]:
        for a in [Enter, Exit, Miss]:
            for b in [Enter, Exit, Miss]:
                ctrlACloser = _acloser[op][a][b] 
                ctrlBCloser = _bcloser[op][a][b] 
                off = (a*3+b)
                if off < 8:
                    pak[0,op] |=  ctrlACloser << (off*4) 
                    pak[1,op] |=  ctrlBCloser << (off*4) 
                pass
            pass
        pass               
    pass
    return pak
                 
_pak = _pack_ctrl_tables() 
_pak0 = np.array([[0x22121141, 0x14014, 0x141141, 0x0],
                  [0x22115122, 0x22055, 0x133155, 0x0]], dtype=np.uint32)

assert np.all(_pak == _pak0)

def optix_pak():
    x8fmt_ = lambda _:"0x%.8x" % _
    fmt_ = lambda a:", ".join(map(x8fmt_, a)) 
    fmt = "rtDeclareVariable(uint4, packed_boolean_lut_%s, , ) = { %s } ;"
    now = datetime.datetime.now().strftime("%c")
    print "// generated from %s by %s on %s " % (os.getcwd(), sys.argv[0],  now )
    print fmt % ("ACloser", fmt_(_pak[0]) )
    print fmt % ("BCloser", fmt_(_pak[1]) )


def pak_lookup(op, a, b, ACloser):
    ac = 1-int(ACloser)  # 0:ACloser 1:BCloser
    lut0 = _pak0[ac][op]
    lut  = _pak[ac][op]
    assert lut0 == lut
    off = (a*3+b)
    return ((lut >> (off*4)) & 0xf) if off < 8 else CTRL_RETURN_MISS
    


if __name__ == '__main__':

    fmt = "   %7s  %7s  ->   %35s       %35s         %35s " 
    for op in [UNION,INTERSECTION,DIFFERENCE]:
        print "%s : boolean_table " % _op_label[op]

        print fmt % ( "","","", "tA < tB", "tB < tA" )

        for a in [Enter, Exit, Miss]:
            for b in [Enter, Exit, Miss]:
                acts = boolean_table(op, a, b )
                actACloser = boolean_decision(acts, ACloser=True )
                actBCloser = boolean_decision(acts, ACloser=False )

                aextra = ""
                if op in _acloser:
                   ctrlACloser = _acloser[op][a][b] 
                   ctrlACloser_pak = pak_lookup(op,a,b,ACloser=True)
                   assert ctrlACloser == ctrlACloser_pak, (ctrlACloser, ctrlACloser_pak)
                   aextra = " " + desc_ctrl(ctrlACloser_pak)  
  
                bextra = ""
                if op in _bcloser:
                   ctrlBCloser = _bcloser[op][a][b] 
                   ctrlBCloser_pak = pak_lookup(op,a,b,ACloser=False)
                   assert ctrlBCloser == ctrlBCloser_pak
                   bextra = " " + desc_ctrl(ctrlBCloser_pak)

                print fmt % ( _st_label[a], _st_label[b], desc_acts(acts), desc_acts(actACloser)+aextra, desc_acts(actBCloser)+bextra )
            pass
        pass 
    pass
    dump_acts()            
    optix_pak()




