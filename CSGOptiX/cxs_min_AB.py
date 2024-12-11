#!/usr/bin/env python
"""

~/o/cxs_min.sh AB 

"""

import os, logging, textwrap, numpy as np
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


    EXPR = filter(None,textwrap.dedent(r"""
    np.all( a.f.genstep == b.f.genstep )
    np.all( a.f.hit == b.f.hit )
    a.f.hit.shape
    b.f.hit.shape
    """).split("\n"))

    for expr in EXPR:
        val = str(eval(expr)) if not expr.startswith("#") else ""
        fmt = "%-80s \n%s\n" if len(val.split("\n")) > 1 else "%-80s : %s"
        print(fmt % (expr, val))
    pass

    

    e = a 
    qtab = e.minimal_qtab()
    print("qtab")
    print(qtab)
 

