#/usr/bin/env python

import numpy as np

# see tests/GColorsTest.cc

if __name__ == '__main__':

   a = np.load("/tmp/colors_GBuffer.npy") 
   b = np.load("/tmp/colors_NPY.npy")

   assert a.shape == (256, 1)
   assert b.shape == (256, 4)

   aa = a.view(np.uint8)

   assert aa.shape == (256, 4)

   assert np.all( b == aa ) 
 

   print "a[:10]", a[:10]
   print "b", b
   print "aa", aa


