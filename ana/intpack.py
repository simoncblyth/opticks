#!/usr/bin/env python
"""
intpack.py
------------

Thinking about twos-complement representation of signed integers 
and bit packing. See SSys::unsigned_as_int 

When packing signed ints have to be more careful with the masking
to avoid getting "leaking" bits as -1:0xffffffff 

::

    In [15]: ( 0 << 16 )    
    Out[15]: 0
    In [16]: ( 0 << 16 ) | -1
    Out[16]: -1
    In [17]: pack = ( 0 << 16 ) | -1
    In [18]: pack >> 16
    Out[18]: -1
    In [19]: pack = ( ( 0 & 0xffff)  << 16 ) | ( -1 & 0xffff )
    In [20]: "%x" % pack
    Out[20]: 'ffff'
    In [21]: pack >> 16
    Out[21]: 0

"""
import sys, binascii as ba, numpy as np
x_ = lambda _:ba.hexlify(_)

num_bytes = 4
signed_max = (0x1 << (num_bytes*8 - 1)) - 1 

ii = list(range(-10,10)) + list(range(signed_max - 10, signed_max+1)) + list(range(-(signed_max+1), -signed_max+10))

assert sys.byteorder == 'little'


i2big_ = lambda _:ba.hexlify(_.to_bytes(2, "big", signed=True )).decode()   
i4big_ = lambda _:ba.hexlify(_.to_bytes(4, "big", signed=True )).decode()   




for i in ii:
   u = np.uint32(i)
   v = u.view(np.int32)
   assert v == i 

   uhi = ( u & 0xffff0000 ) >> 16 
   ulo = ( u & 0x0000ffff ) >> 0

   l = i.to_bytes(num_bytes, "little", signed=True )  
   b = i.to_bytes(num_bytes, "big", signed=True )  

   print("i:%12d u:%12d x:%8x xhi:%4x xlo:%4x        little:%10s  big:%10s " % (i,u,u,uhi,ulo,x_(l), x_(b)))
pass

print("(endianness is implementation detail that only very rarely need to think about even when masking and packing, only relevant when doing very low level debug eg with xxd)")
print("(splitting a 16 bit value across two adjacent 8 bit fields and then reinterpreting them as 16 bit is one example where would need to consider endianness)") 
print("little: least significant byte is first in memory")
print("   big: least significant byte is last in memory")
print("     x: hex string presentation looks like big-endian, the less common endianness" )
print(" sys.byteorder : %s " % sys.byteorder)  



# packing signed ints in unsigned array 

n0 = np.arange(0, -100, -1, dtype=np.int32)  
n1 = -1000 + n0 

nn = np.zeros(len(n0), dtype=np.uint32)

nn[:] = (( n0 & 0xffff ) << 16 ) | ((n1 & 0xffff) << 0 )
 
## with negative ints have to downsize the container and pluck the evens
## because of twos-complement rep of signed integers
n0chk = ( n0 & 0xffff ).view(np.int16)[::2]    
assert np.all( n0chk == n0 )

n1chk = ( n1 & 0xffff ).view(np.int16)[::2] 
assert np.all( n1chk == n1 )

nn_0 = (nn >> 16).view(np.int16)[::2] 
assert np.all( n0 == nn_0 ) 

nn_1 = (nn & 0xffff).view(np.int16)[::2] 
assert np.all( n1 == nn_1 ) 


# compare with packing unsigned ints in unsigned array
p0 = np.arange(0, 100, dtype=np.uint32)
p1 = 1000 + p0 
pp = np.zeros( len(p0), dtype=np.uint32) 

pp[:] = (( p0 & 0xffff ) << 16 ) | ((p1 & 0xffff) << 0 )

p0chk = p0 & 0xffff   
assert np.all( p0chk == p0 )

p0chk_ = ( p0 & 0xffff ).view(np.int16)[::2]  ## works, but not needed for +ve 
assert np.all( p0chk_ == p0 )

p1chk = p1 & 0xffff   
assert np.all( p1chk == p1 )

p1chk_ = ( p1 & 0xffff ).view(np.int16)[::2]   ## works, but not needed for +ve 
assert np.all( p1chk_ == p1 )


# cover the whole 16 bit range and one extra that clocks it at each end
r0 = np.arange(-0x7fff-1-1, 0x7fff+1+1, dtype=np.int32) 
r1 = r0[::-1]

rr = np.zeros( len(r0), dtype=np.uint32 )
rr[:] = ((r0 & 0xffff) << 16 ) | ((r1 & 0xffff) << 0)

rr_0 = (rr >> 16).view(np.int16)[::2] 
rr_1 = (rr & 0xffff).view(np.int16)[::2] 

# just the one steps beyond dont match 
assert len(np.where( rr_0 != r0 )[0]) == 2  
assert len(np.where( rr_1 != r1 )[0]) == 2  




