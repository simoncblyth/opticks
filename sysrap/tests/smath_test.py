#!/usr/bin/env python




import os, textwrap, numpy as np
from opticks.ana.fold import Fold, EXPR_
from opticks.sysrap.smath import rotateUz
import matplotlib.pyplot as mp

SIZE=np.array([1280, 720])
TEST=os.environ["TEST"]




def rotateUz_test(f):
    u = f.rotateUz[0,0]
    assert np.all( f.rotateUz[:,0] == u )

    d = f.rotateUz[:,1]
    d1 = f.rotateUz[:,2]
    d1p = rotateUz(d, u)

    for expr in EXPR_(r"""
u
d      # original direction from C++
d1     # C++ rotateUz
d1p    # py rotateUz
d1 - d1p
(d1 - d1p).min()
(d1 - d1p).max()
"""):
        print(expr)
        if expr == "" or expr[0] == "#": continue
        print(repr(eval(expr)))
    pass
pass


def log_cu_test(f):
    pass
    fig, ax = mp.subplots(figsize=SIZE/100.)
    title= f"smath_test.sh : {f.base} : {TEST}  "
    fig.suptitle(title)
    ax.plot( f.dom, f.val, label="val,dom" )
    ax.legend()
    fig.show()



if __name__ == '__main__':
    f = Fold.Load("$FOLD/$TEST", symbol="f")
    print(repr(f))

    if TEST == "rotateUz":
        rotateUz_test(f)
    elif TEST == "log_cu_float":
        log_cu_test(f)
    elif TEST == "log_cu_double":
        log_cu_test(f)
    else:
        pass
    pass



