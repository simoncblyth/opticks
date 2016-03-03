#!/bin/env python
import os, datetime, logging
log = logging.getLogger(__name__)
import numpy as np

from env.numerics.npy.base import Abbrev, ListFlags
from env.numerics.npy.seq import SeqType, SeqAna
from env.numerics.npy.PropLib import PropLib
from env.numerics.npy.nload import A

def test_roundtrip(mt):
    s = "MO Py MO MO Py OpaqueVacuum Vm MO"
    i = mt.code(s)
    l = mt.label(i)
    print "%s : %x : %s" % (s, i, l )
    assert l == s 

class MatType(SeqType):
    def __init__(self):
        flags = ListFlags("GMaterialLib")
        abbrev = Abbrev("~/.opticks/GMaterialLib/abbrev.json")
        SeqType.__init__(self, flags, abbrev)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    af = MatType()
    test_roundtrip(af)

    src = "torch"
    tag = "1"
    det = "PmtInBox"  
    ph = A.load_("ph"+src,tag,det)
    seqmat = ph[:,0,1]

    ma = SeqAna.for_evt(af, tag, src, det, offset=1)
    print ma.table









