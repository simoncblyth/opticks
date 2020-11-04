#!/usr/bin/env python
"""
intpack.py
------------

Thinking about twos-complement representation of signed integers 
and bit packing. See SSys::unsigned_as_int 


"""
import sys, binascii as ba, numpy as np

num_bytes = 4
signed_max = (0x1 << (num_bytes*8 - 1)) - 1 

ii = list(range(-10,10)) + list(range(signed_max - 10, signed_max+1)) + list(range(-(signed_max+1), -signed_max+10))

assert sys.byteorder == 'little'

for i in ii:
   u = np.uint32(i)
   v = u.view(np.int32)
   assert v == i 

   l = i.to_bytes(num_bytes, "little", signed=True )  
   b = i.to_bytes(num_bytes, "big", signed=True )  
   print("i:%12d u:%12d x:%12x little:%10s  big:%10s " % (i,u,u, ba.hexlify(l), ba.hexlify(b)))
pass

print("")
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


