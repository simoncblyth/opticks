#!/usr/bin/env python
"""
opticks_CSGOptiX_test.py
=========================

::

   ~/o/CSGOptiX/tests/opticks_CSGOptiX_test.py


"""
import functools, operator, numpy as np
import opticks_CSGOptiX as cx

def test_CSGOptiXService_ctor():
    print("[test_CSGOptiXService_ctor")
    svc = cx.CSGOptiXService()
    print("repr(svc):[%s]" % repr(svc))
    print("]test_CSGOptiXService_ctor")


def main():
    test_CSGOptiXService_ctor()
pass

if __name__ == '__main__':
    #main()
    svc = cx.CSGOptiXService()

    gs = np.arange(10*6*4, dtype=np.float32).reshape(10,6,4)
    ht = svc.simulate(gs)

    print("gs\n",gs)
    print("ht\n",ht)
pass

