#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

print("[from opticks.sysrap.sevt import SEvt, SAB")
from opticks.sysrap.sevt import SEvt, SAB
print("]from opticks.sysrap.sevt import SEvt, SAB")


TEST = os.environ.get("TEST","")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    a = SEvt.Load("$AFOLD", symbol="a")
    print(repr(a))

    if "BFOLD" in os.environ:   
        b = SEvt.Load("$BFOLD", symbol="b") 
        print(repr(b))
        ab = SAB(a,b) 
        print(repr(ab))
    pass


    e = a 

    qtab = e.minimal_qtab()
    print("qtab")
    print(qtab)
 

