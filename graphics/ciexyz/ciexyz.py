#!/usr/bin/env/python
"""
Access simple C float-float functions with ctypes
"""
import ctypes

class CLib(object):
    def __init__(self, path):
        self.lib = ctypes.cdll.LoadLibrary(path)

    def make_ff(self, fn):
        f = getattr(self.lib,fn) 
        f.argtypes = [ctypes.c_float]
        f.restype = ctypes.c_float 
        return f 

if __name__ == '__main__':
    clib = CLib("/tmp/ciexyz.dylib")
    xFit = clib.make_ff("xFit_1931") 
    yFit = clib.make_ff("yFit_1931") 
    zFit = clib.make_ff("zFit_1931") 

    clib.lib.dump()

    for w in range(400,800,20):
        print " %s  %10.4f %10.4f %10.4f " % (w, xFit(w), yFit(w), zFit(w) )




