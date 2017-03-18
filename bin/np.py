#!/usr/bin/env python
import sys, numpy as np
np.set_printoptions(suppress=True, precision=4, linewidth=200)

if __name__ == '__main__':
    a = np.load(sys.argv[1])
    print a.shape
    print "f32\n",a.view(np.float32)
    print "u32\n",a.view(np.uint32)
    #print "i32\n",a.view(np.int32)

