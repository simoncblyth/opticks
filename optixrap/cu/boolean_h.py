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
import boolean_solid_h as h 
enum_kv = filter(lambda kv:kv[0][0] != "_",h.__dict__.items())

from boolean_solid_h import ReturnMiss, ReturnA, ReturnAIfCloser, ReturnAIfFarther
from boolean_solid_h import ReturnB, ReturnBIfCloser, ReturnBIfFarther, ReturnFlipBIfCloser
from boolean_solid_h import AdvanceAAndLoop, AdvanceAAndLoopIfCloser
from boolean_solid_h import AdvanceBAndLoop, AdvanceBAndLoopIfCloser


## boolean operation  

UNION  = 1
INTERSECTION = 2
DIFFERENCE = 3
_op_label = { UNION:"Union", INTERSECTION:"Intersection", DIFFERENCE:"Difference" }
def desc_op(op):
    return _op_label[op]

## intersection state of a ray with a primitive

Enter = 1
Exit = 2
Miss = 3
_st_label = { Enter:"Enter", Exit:"Exit", Miss:"Miss" }
_st = dict(zip(_st_label.values(), _st_label.keys()))

def desc_state(st):
    return _st_label(st)


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
    for k,v in filter(lambda kv:kv[0].find("_") == -1, ekv):
        #print "%30s : %s " % (k, v )
        _act[k] = v 
        _act_label[v] = k 
        _act_index[v] = ffs(v)   # CAUTION THIS WAS OFF-BY-ONE
    pass
_init_acts(enum_kv)


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
    for v in sorted(_act_label.keys()):
        print " %6d : %2d : %30s " % ( v, act_index(v), desc_acts(v) )


_table = {}

def _init_table(ekv):
    global _table
    for op in [UNION, INTERSECTION, DIFFERENCE]:
        ol = desc_op(op)
        for k,v in sorted(filter(lambda kv:kv[0].startswith(ol),ekv), key=lambda kv:kv[1]):
            #print "%30s : %s " % (k,v )

            ol2,al,bl = k.split("_")
            assert ol2 == ol
        
            a = _st[al[:-1]]
            b = _st[bl[:-1]]

            if op not in _table:        
                _table[op] = {}
            pass
            if a not in _table[op]:
                _table[op][a] = {}
            pass
            if b not in _table[op][a]:
                _table[op][a][b] = {}
            pass
            _table[op][a][b] = v
        pass
    pass
_init_table(enum_kv)


def boolean_table(op, a, b):
    assert op in [UNION, INTERSECTION, DIFFERENCE], op
    assert a in [Enter,Exit,Miss], a
    assert b in [Enter,Exit,Miss], b
    return _table[op][a][b]


def boolean_decision(acts, tA, tB):
    """
    :param acts: actions mask 
    :param tA: distance to first intersect of ray with A
    :param tB: distance to first intersect of ray with B
    :return act:

    Decide on the single action to take from
    potentially multiple in the actions mask 
    based on which intersection is closer 
    """

    ACloser = tA <= tB
    BFarther = ACloser
    BCloser = not ACloser
    AFarther = not ACloser

    if (acts & ReturnMiss):
        act = ReturnMiss

    elif (acts & ReturnA):
        act = ReturnA
    elif (acts & ReturnAIfCloser) and ACloser:
        act = ReturnAIfCloser
    elif (acts & ReturnAIfFarther) and AFarther:
        act = ReturnAIfFarther

    elif (acts & ReturnB):
        act = ReturnB
    elif (acts & ReturnBIfCloser) and BCloser:
        act = ReturnBIfCloser
    elif (acts & ReturnFlipBIfCloser) and BCloser:
        act = ReturnFlipBIfCloser
    elif (acts & ReturnBIfFarther) and BFarther:
        act = ReturnBIfFarther
    
    elif (acts & AdvanceAAndLoop):
        act = AdvanceAAndLoop
    elif (acts & AdvanceAAndLoopIfCloser) and ACloser:
        act = AdvanceAAndLoopIfCloser

    elif (acts & AdvanceBAndLoop):
        act = AdvanceBAndLoop
    elif (acts & AdvanceBAndLoopIfCloser) and BCloser:
        act = AdvanceBAndLoopIfCloser
    else:
        assert 0, (acts, desc_acts(acts), tA, tB, ACloser, BFarther, BCloser, AFarther)
    pass 
    return act 


if __name__ == '__main__':

    fmt = "   %7s  %7s  ->   %35s       %35s         %35s " 
    for op in [UNION,INTERSECTION,DIFFERENCE]:
        print "%s : boolean_table " % _op_label[op]

        print fmt % ( "","","", "tA < tB", "tB < tA" )

        for a in [Enter, Exit, Miss]:
            for b in [Enter, Exit, Miss]:
                acts = boolean_table(op, a, b )
                actACloser = boolean_decision(acts, 1, 2 )
                actBCloser = boolean_decision(acts, 2, 1 )
                print fmt % ( _st_label[a], _st_label[b], desc_acts(acts), desc_acts(actACloser), desc_acts(actBCloser) )
            pass
        pass 
    pass
    dump_acts()            




