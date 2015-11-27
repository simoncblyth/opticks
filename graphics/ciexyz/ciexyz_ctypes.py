#!/usr/bin/env/python
"""
Access simple C float-float functions with ctypes
"""
import os, ctypes

class CLib(object):
    def __init__(self, path):
        self.lib = ctypes.cdll.LoadLibrary(path)

    def make_ff(self, fn):
        f = getattr(self.lib,fn) 
        f.argtypes = [ctypes.c_float]
        f.restype = ctypes.c_float 
        return f 


class CIEXYZ(object):
    def __init__(self):
        path = os.path.expandvars("$LOCAL_BASE/env/graphics/ciexyz/ciexyz.dylib")
        clib = CLib(path)
        x = clib.make_ff("xFit_1931") 
        y = clib.make_ff("yFit_1931") 
        z = clib.make_ff("zFit_1931") 
        self.x = x 
        self.y = y 
        self.z = z
        self.clib = clib  

    def dump(self):
        self.clib.lib.dump()


if __name__ == '__main__':

    xyz = CIEXYZ()
    for w in range(400,800,20):
        print " %s  %10.4f %10.4f %10.4f " % (w, xyz.x(w), xyz.y(w), xyz.z(w) )




