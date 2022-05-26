#!/usr/bin/env python

import numpy as np

X,Y,Z = 0,1,2 


class XZ(object):
    """

            +------+
            |      |  nz     ->  ( X, Z )    nx_over_nz > 1 
            +------+
               nx

               Z  Y                    
               | /
               |/
               +----- X
              .
            -Y

    """
    up = np.array(  [0,0,1], dtype=np.float32 )
    off = np.array( [0,-1,0], dtype=np.float32 )


class ZX(object):
    """

            +------+
            |      |  nx     ->  ( Z, X )    nx_over_nz < 1 
            +------+
               nz

               X  -Y                    
               | .
               |.
               +----- Z
              /
             Y

    HMM: X4IntersectSolidTest.py was using up (-1,0,0) ?
    """
    up = np.array(  [1,0,0], dtype=np.float32 )
    off = np.array( [0,1,0], dtype=np.float32 )


class YZ(object):
    """

            +------+
            |      |  nz     ->  ( Y, Z )    ny_over_nz > 1 
            +------+
               ny

               Z                     
               |
               |
               +----- Y
              /
            X

    """
    up = np.array( [0,0,1], dtype=np.float32 )
    off = np.array( [1,0,0], dtype=np.float32 )


class ZY(object):
    """

            +------+
            |      |  ny     ->  ( Z, Y )    ny_over_nz > 1 
            +------+
               nz

               Y  X                    
               | /
               |/
               +----- Z
              .
           - X

    """
    up = np.array( [0,1,0], dtype=np.float32 )
    off = np.array( [-1,0,0], dtype=np.float32 ) 



class XY(object):
    """

            +------+
            |      |  ny     ->  ( X, Y )    nx_over_ny > 1 
            +------+
               nx

               Y  -Z                     
               | .
               |.
               +----- X
              /
             Z

    """
    up = np.array( [0,1,0], dtype=np.float32 )
    off = np.array( [0,0,1], dtype=np.float32 ) 


class YX(object):
    """

            +------+
            |      |  nx     ->  ( Y, X )    nx_over_ny < 1 
            +------+
               ny

               X  -Z                     
               | .
               |.
         Y ----+. . .  -Y
              /
             Z

    """
    up = np.array( [1,0,0], dtype=np.float32 )
    off = np.array( [0,0,1], dtype=np.float32 ) 


class XYZ(object):
    """

            Z  
            | 
            |
            +------ Y
           /
          /
         X

    """ 
    up = np.array( [0,0,1], dtype=np.float32 )
    off = np.array( [1,0,0], dtype=np.float32 )


class Axes(object):
    ups = {}
    ups["XZ"] = XZ.up
    ups["ZX"] = ZX.up

    ups["YZ"] = YZ.up
    ups["ZY"] = ZY.up

    ups["XY"] = XY.up
    ups["YX"] = YX.up

    ups["XYZ"] = XYZ.up

    offs = {}
    offs["XZ"] = XZ.off
    offs["ZX"] = ZX.off

    offs["YZ"] = YZ.off
    offs["ZY"] = ZY.off

    offs["XY"] = XY.off
    offs["YX"] = YX.off

    offs["XYZ"] = XYZ.off


    @classmethod
    def HV_(cls, H, V, axes="XYZ"):
        return "%s%s" % (axes[H], axes[V] ) 
 
    @classmethod
    def Up(cls, H, V):
        HV = cls.HV_(H,V) 
        up = cls.ups.get(HV, None)
        return up 

    @classmethod
    def Off(cls, H, V):
        HV = cls.HV_(H,V) 
        off = cls.offs.get(HV, None)
        return off 



if __name__ == '__main__':
    up = Axes.Up(0, 1)




