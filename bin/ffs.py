#!/usr/bin/env python

import ctypes

cpp = ctypes.cdll.LoadLibrary('libc++.1.dylib')
cppffs_ = lambda _:cpp.ffs(_)

ffs_ = lambda x:(x&-x).bit_length()


if __name__ == '__main__':

    for i in range(-1,16):
        n = 0x1 << i if i > -1 else 0
        print " i %2d n:%5d  n:0x%4x cpp.ffs_: %2d ffs_: %2d " % (i, n, n, cppffs_(n), ffs_(n) )   



