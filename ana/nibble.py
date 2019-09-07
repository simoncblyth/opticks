#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""
::

    In [14]: np.where( seqhis & 0xf << (4*5) != 0 )
    Out[14]: (array([5, 6, 7, 8, 9]),)

"""
from collections import OrderedDict as odict 
import numpy as np


msk_ = lambda n:(1 << 4*n) - 1  # msk_(0)=0x0 msk_(1)=0xf msk_(2)=0xff msk_(3)=0xfff 

def make_msk(n):
   msk = np.zeros( n, dtype=np.uint64  )
   for i in range(n): 
       msk[i] = msk_(i)
   pass
   return msk 


nib_ = lambda n:( 0xf << 4*(n-1) ) if n > 0 else 0    # nib_(0)=0xf nib_(1)=0xf0 nib_(2)=0xf00 nib_(3)=0xf000

def make_nib(n):
    """
    :return nib: array of n uint64 with nibble masks of 4 bits each for slots from 0 to n-1 
    """  
    nib = np.zeros( n, dtype=np.uint64  )
    for i in range(n): 
        nib[i] = nib_(i)
    pass
    return nib


def count_nibbles(x):
    """
    https://stackoverflow.com/questions/38225571/count-number-of-zero-nibbles-in-an-unsigned-64-bit-integer
    """

    ## gather the zeroness (the OR of all 4 bits)
    x |= x >> 1               # or-with-1bit-right-shift-self is or-of-each-consequtive-2-bits 
    x |= x >> 2               # or-with-2bit-right-shift-self is or-of-each-consequtive-4-bits in the lowest slot 
    x &= 0x1111111111111111   # pluck the zeroth bit of each of the 16 nibbles

    x = (x + (x >> 4)) & 0xF0F0F0F0F0F0F0F    # sum occupied counts of consequtive nibbles, and pluck them 
    count = (x * 0x101010101010101) >> 56     #  add up byte totals into top byte,  and shift that down to pole 64-8 = 56 

    return count 



tst_ = lambda i:reduce(lambda a,b:a|b, map(lambda n:0x1 << (4*n), range(i) ),0)  # marching 1s to mimic seqhis nibbles 
tst0_ = lambda n:(1 << 4*n)   # leaves unoccupied nibbles to the right, so not valid standin for seqhis nibbles

def test_tst(n):
    """
     0 : 0000000000000000
     1 : 0000000000000001
     2 : 0000000000000011
     3 : 0000000000000111
     4 : 0000000000001111
     5 : 0000000000011111
     6 : 0000000000111111
     7 : 0000000001111111
     8 : 0000000011111111
     9 : 0000000111111111
    10 : 0000001111111111
    11 : 0000011111111111
    12 : 0000111111111111
    13 : 0001111111111111
    14 : 0011111111111111
    15 : 0111111111111111
    """
    print("\n".join(map(lambda i:"%2d : %0.16x" % (i, tst_(i) ), range(n))))


def make_tst(n):
   """
   Create array of 64bit uints as standin for seqhis
   photon histories with 4 bits per recorded step point
   """
   tst = np.zeros( n, dtype=np.uint64 )
   for i in range(n):   
       tst[i] = tst_(i)
   pass
   xpo = np.arange( len(tst), dtype=np.int32 )
   return tst, xpo


def make_tst2(n):
   """
   Create array of 64bit uints as standin for seqhis
   photon histories with 4 bits per recorded step point
   """
   tst = np.zeros( 2*n, dtype=np.uint64 )
   xpo = np.zeros( 2*n, dtype=np.uint64 )
   for i in range(n):   
       tst[i] = tst_(i)
       xpo[i] = i 
   for i in range(n):   
       tst[n+i] = tst_(n-1-i)
       xpo[n+i] = n-1-i 
   pass
   return tst, xpo


def dump_msk_nib(msk, nib):
    print(" %16s : %16s : %16s " % ( "nib", "msk", "~msk" )) 
    for i in range(len(msk)):
        print(" %16x : %16x : %16x " % ( nib[i], msk[i], ~msk[i]  ))
    pass




if __name__ == '__main__':
    n = 16 
    msk = make_msk(n)
    nib = make_nib(n)
    dump_msk_nib(msk, nib)

    #test_tst(n)
    tst,xpo = make_tst2(n)

    ## number of occupied nibbles in the seqhis
    npo = np.zeros(len(tst), dtype=np.int32)
    qpo = odict() 
    wpo = odict() 

    for i in range(16):
        wpo[i] = np.where( tst >> (4*i) != 0 )
        npo[wpo[i]] = i+1    
        """
        wpo[i]
           right shift by i nibbles and see if any bits remain 
        
        Notice the overwriting into npo, this works because  
        are using ascending order

        """
        #qpo[i] = np.where(np.logical_and(np.logical_and( tst & ~msk[i] == 0, tst & msk[i] == tst ), tst & nib[i] != 0  ))[0]
        #qpo[i] = np.where(reduce(np.logical_and, (tst & ~msk[i] == 0, tst & msk[i] == tst, tst & nib[i] != 0  )))
        #qpo[i] = np.where(reduce(np.logical_and, (tst & ~msk[i] == 0, tst & nib[i] != 0  )))
        #qpo[i] = np.where(np.logical_and(tst & ~msk[i] == 0, tst & nib[i] != 0  ))
        qpo[i] = np.where( (tst & ~msk[i] == 0) & (tst & nib[i] != 0) )
    pass
    """
    qpo[i] (i in 1..16)  gives tst indices having each of the possible number of occupied nibbles 

    tst & ~msk[i] == 0
        and-ing with anti-mask[i] removes high side nibbles, 
        but leaves all those below 
         
    tst & nib[i] != 0
        requiring something in nibble i cuts out all those below   

    All the indices selected have the same number of occupied nibbles 

    This may be specialized to the seqhis case where never get 
    zero nibbles on the low side. 
    """ 

    print(" %2s : %16s : %2s : %2s " % ( "i", "tst", "xp", "np" )) 
    for i in range(len(tst)):
        print(" %2d : %16x : %2d : %2d " % ( i, tst[i],  xpo[i], npo[i]  ))
    pass

    assert np.all( npo == xpo )


    print(" %2s : %16s : %s " % ( "i", "tst", "qpo" )) 
    for i in range(16):
        print(" %2d : %16x : %s " % (  i, tst[i],  repr(qpo[i]) ))
    pass

 
