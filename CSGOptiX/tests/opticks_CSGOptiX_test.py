#!/usr/bin/env python
"""
opticks_CSGOptiX_test.py
=========================

::

   ~/o/CSGOptiX/tests/opticks_CSGOptiX_test.py


"""
import functools, operator, numpy as np
import opticks_CSGOptiX as cx

def test_inspect():
    print("[test_inspect")
    shape = (10,6,4)
    sz = functools.reduce(operator.mul,shape)
    a = np.arange(sz, dtype=np.float32).reshape(*shape)
    print("a\n",a)
    cx.inspect(a)
    print("]test_inspect")

def test_Dog():
    print("[test_Dog")
    d = cx.Dog("max")
    print(d)
    d.name = "maxine"
    print(d)
    print("]test_Dog")

def test_CSGOptiXService():
    print("[test_CSGOptiXService")
    s = cx.CSGOptiXService()
    print("repr(s):[%s]" % repr(s))
    print("]test_CSGOptiXService")

def test_create():
    #s = cx.create_2d(4,4)
    s = cx.create_3d(4,4,4)
    print("repr(s):\n%s\n" % repr(s))

def test_create_from_NP():
    for i in range(6):
        s = cx.create_from_NP(i)
        print("repr(s):\n%s\n" % repr(s))
    pass



def main():
    test_inspect()
    test_Dog()
    test_CSGOptiXService()
    test_create()
    test_create_from_NP()
pass

if __name__ == '__main__':
    #main()
    a = np.arange(10*6*4, dtype=np.float32).reshape(10,6,4)
    b = cx.roundtrip_numpy_array_via_NP(a)
    print("a\n",a)
    print("b\n",b)
pass

