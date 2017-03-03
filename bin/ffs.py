#!/usr/bin/env python
"""

Huh, the ffs and clz symbols appear not to be there with nm, but ctypes finds it::

    simon:opticks blyth$ nm /usr/lib/libc++.1.dylib | grep ffs
    simon:opticks blyth$ nm /usr/lib/libc++.1.dylib | c++filt | grep ffs
    simon:opticks blyth$ nm /usr/lib/libc.dylib | grep ffs
    simon:opticks blyth$ 

From one of the many libs of libSystem::

    simon:opticks blyth$ otool -L /usr/lib/libSystem.B.dylib
    /usr/lib/libSystem.B.dylib:
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)
        /usr/lib/system/libcache.dylib (compatibility version 1.0.0, current version 62.0.0)
        /usr/lib/system/libcommonCrypto.dylib (compatibility version 1.0.0, current version 60049.0.0)
        /usr/lib/system/libcompiler_rt.dylib (compatibility version 1.0.0, current version 35.0.0)
        /usr/lib/system/libcopyfile.dylib (compatibility version 1.0.0, current version 103.92.1)
        /usr/lib/system/libcorecrypto.dylib (compatibility version 1.0.0, current version 1.0.0)
        /usr/lib/system/libdispatch.dylib (compatibility version 1.0.0, current version 339.92.1)
        ...



::

    In [77]: map(cpp.fls, [0x1 << n for n in range(22)])
    Out[77]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

::

    In [97]: map(cpp.fls,[0x1,0x1f,0x1ff,0x1fff,0x1ffff,0x1fffff,0x1ffffff,0x1fffffff,0x1ffffffff,0x1fffffffff,0x1fffffffffffffff])
    Out[97]: [1, 5, 9, 13, 17, 21, 25, 29, 32, 32, 32]


::

    simon:opticks blyth$ nm /usr/lib/system/libcompiler_rt.dylib | grep clz
    0000000000005ded S $ld$hide$os10.4$___clzdi2
    0000000000005def S $ld$hide$os10.4$___clzsi2
    0000000000005df1 S $ld$hide$os10.4$___clzti2
    0000000000005dee S $ld$hide$os10.5$___clzdi2
    0000000000005df0 S $ld$hide$os10.5$___clzsi2
    0000000000005df2 S $ld$hide$os10.5$___clzti2
    0000000000001fc5 T ___clzdi2
    0000000000001fe2 T ___clzsi2
    000000000000205c T ___clzti2
    simon:opticks blyth$ 
    simon:opticks blyth$ 
    simon:opticks blyth$ nm /usr/lib/system/libcompiler_rt.dylib | grep ffs
    0000000000005e0b S $ld$hide$os10.4$___ffsdi2
    0000000000005e0d S $ld$hide$os10.4$___ffsti2
    0000000000005e0c S $ld$hide$os10.5$___ffsdi2
    0000000000005e0e S $ld$hide$os10.5$___ffsti2
    0000000000002d77 T ___ffsdi2
    0000000000002d94 T ___ffsti2
    simon:opticks blyth$ 



"""
import ctypes

cpp = ctypes.cdll.LoadLibrary('libc++.1.dylib')
rt = ctypes.cdll.LoadLibrary('/usr/lib/system/libcompiler_rt.dylib')

cppffs_ = lambda _:cpp.ffs(_)
cppfls_ = lambda _:cpp.fls(_)

ffs_ = lambda x:(x&-x).bit_length()

def clz_(x):
   """
   https://en.wikipedia.org/wiki/Find_first_set
   """
   n = 0 ;
   if x == 0: return 32
   while (x & 0x80000000) == 0:
       n += 1
       x <<= 1  
   pass
   return n

def test_clz():
    print " %10s : %6s %6s %6s  " % ("test_clz", "clz", "32-clz",  "fls")
    for i in [0x0,0xf,0xff,0xfff,0xffff,0xfffff,0xffffff,0xfffffff,0xffffffff]:
        c = clz_(i)
        f = cppfls_(i)
        print " %10x : %6u %6u %6u " % (i, c, 32-c, f) 

def test_ffs():
    for i in range(-1,16):
        n = 0x1 << i if i > -1 else 0
        print " i %2d n:%5d  n:0x%4x cpp.ffs_: %2d ffs_: %2d " % (i, n, n, cppffs_(n), ffs_(n) )   



if __name__ == '__main__':
    test_ffs()
    test_clz()
