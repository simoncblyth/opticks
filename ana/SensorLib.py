#!/usr/bin/env python
"""
::

    run ~/opticks/ana/SensorLib.py 

    In [18]: sl.sensorData[:,3].view(np.int32)[:100]                                                                                                                                                          
    Out[18]: 
    array([0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x1000000, 0x1000100,
           0x1000200, 0x1000300, 0x1000400, 0x1000500, 0x1000600, 0x1000700,
           0x8, 0x9, 0x1000800, 0x1000900, 0x1000a00, 0x1000b00, 0x1000c00,
           0x1000d00, 0x1000e00, 0x1000f00, 0xa, 0xb, 0x1001000, 0x1001100,
           0x1001200, 0x1001300, 0x1001400, 0x1001500, 0x1001600, 0x1001700,
           0xc, 0xd, 0x1001800, 0x1001900, 0x1001a00, 0x1001b00, 0x1001c00,
           0x1001d00, 0x1001e00, 0x1001f00, 0xe, 0xf, 0x10, 0x11, 0x1002000,
           0x1002100, 0x1002200, 0x1002300, 0x1002400, 0x1002500, 0x1002600,
           0x1002700, 0x12, 0x13, 0x1002800, 0x1002900, 0x1002a00, 0x1002b00,
           0x1002c00, 0x1002d00, 0x1002e00, 0x1002f00, 0x14, 0x15, 0x1003000,
           0x1003100, 0x1003200, 0x1003300, 0x1003400, 0x1003500, 0x1003600,
           0x1003700, 0x16, 0x17, 0x1003800, 0x1003900, 0x1003a00, 0x1003b00,
           0x1003c00, 0x1003d00, 0x1003e00, 0x1003f00, 0x18, 0x19, 0x1a, 0x1b,
           0x1c, 0x1004000, 0x1004100, 0x1004200, 0x1004300, 0x1004400,
           0x1004500, 0x1004600], dtype=int32)

    In [39]: ae[:,0]                                                                                                                                                                                          
    Out[39]: 
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)


"""
np.set_printoptions(suppress=True) 
np.set_printoptions(formatter={'int':hex})
np.set_printoptions(edgeitems=10, linewidth=200) 

import os, numpy as np

class SensorLib(object):
   def __init__(self, fold):
       self.sensorData = np.load(os.path.expandvars(os.path.join(fold, "sensorData.npy")))
       self.angularEfficiency = np.load(os.path.expandvars(os.path.join(fold, "angularEfficiency.npy")))

if __name__ == '__main__':
    sl = SensorLib("$TMP/G4OKTest/SensorLib")

    assert np.all( sl.sensorData[:,0] == 0.5 ) 
    assert np.all( sl.sensorData[:,1] == 1.0 ) 
    assert np.all( sl.sensorData[:,2] == 0. ) 

    i = sl.sensorData[:,3].view(np.int32)  

    ae = sl.angularEfficiency[0].reshape(-1,360) 


