#!/usr/bin/env python

import logging, os 
import numpy as np
log = logging.getLogger(__name__)

eary_ = lambda ekey, edef:np.array( list(map(float, os.environ.get(ekey,edef).split(","))) )


X,Y,Z = 0,1,2

_axes = {}
_axes[X] = "X"
_axes[Y] = "Y"
_axes[Z] = "Z"

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



class Axes(object):
    ups = {}
    ups["XZ"] = XZ.up
    ups["ZX"] = ZX.up

    ups["YZ"] = YZ.up
    ups["ZY"] = ZY.up

    ups["XY"] = XY.up
    ups["YX"] = YX.up

    offs = {}
    offs["XZ"] = XZ.off
    offs["ZX"] = ZX.off

    offs["YZ"] = YZ.off
    offs["ZY"] = ZY.off

    offs["XY"] = XY.off
    offs["YX"] = YX.off

    @classmethod
    def Up(cls, HV):
        up = cls.ups.get(HV, None)
        return up 

    @classmethod
    def Off(cls, HV):
        off = cls.offs.get(HV, None)
        return off 


class GridSpec(object):
    @classmethod
    def DemoPeta(cls, ix0=-2, ix1=2, iy0=-2, iy1=2, iz0=0, iz1=0, photons_per_genstep=100):
        peta = np.zeros( (1,4,4), dtype=np.float32 )
        peta.view(np.int32)[0,0] = [ix0,ix1,iy0,iy1]
        peta.view(np.int32)[0,1] = [iz0,iz1,photons_per_genstep, 0]
        ce = [ -492.566,  -797.087, 19285.   ,   264.   ]
        peta[0,2] = ce
        return peta 


    def __init__(self, peta):

        ix0,ix1,iy0,iy1 = peta[0,0].view(np.int32)
        iz0,iz1,photons_per_genstep,zero = peta[0,1].view(np.int32)

        ce = tuple(peta[0,2])
        sce = ("%7.2f" * 4 ) % ce

        assert photons_per_genstep > 0
        assert zero == 0
        nx = (ix1 - ix0)//2
        ny = (iy1 - iy0)//2
        nz = (iz1 - iz0)//2

        log.info(" ix0 %d ix1 %d nx %d  " % (ix0, ix1, nx)) 
        log.info(" iy0 %d iy1 %d ny %d  " % (iy0, iy1, ny)) 
        log.info(" iz0 %d iz1 %d nz %d  " % (iz0, iz1, nz)) 

        # below default from envvars are overridden for planar data
        eye = eary_("EYE","1.,1.,1.")
        look = eary_("LOOK","0.,0.,0.")
        up  = eary_("UP","0.,0.,1.")
        off  = eary_("OFF","0.,0.,1.")

        axes = self.determine_planar_axes(nx, ny, nz)
        planar = not axes is None

        if planar:
            H, V = axes
            axlabels =  _axes[H], _axes[V]

            if   H == X and V == Z: HV = "XZ"
            elif H == Z and V == X: HV = "ZX"
            elif H == Y and V == Z: HV = "YZ"
            elif H == Z and V == Y: HV = "ZY"
            elif H == X and V == Y: HV = "XY"
            elif H == Y and V == X: HV = "YX"
            else: HV = None

            up  = Axes.Up(HV)
            off = Axes.Off(HV)
            eye = look + 50.*off
        else:
            HV = None
            axlabels = None          
            pass
        pass

        self.eye = eye
        self.look = look 
        self.up  = up
        self.off = off
        self.HV = HV 
        self.peta = peta 
        self.axes = axes
        self.planar = planar
        self.axlabels = axlabels

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ce = ce
        self.sce = sce
        self.thirdline = " ce: " + sce 
        self.photons_per_genstep = photons_per_genstep

    def __str__(self):
        return "GridSpec (nx ny nz) (%d %d %d) axes %s axlabels %s " % (self.nx, self.ny, self.nz, str(self.axes), str(self.axlabels) ) 

    def determine_planar_axes(self, nx, ny, nz):
        """
        :param nx:
        :param nx:
        :param nx:

        Planar axes order is arranged to make the longer axis the first horizontal one 
        followed by the shorter axis as the vertical one.  

            +------+
            |      |  nz     ->  ( Y, Z )    ny_over_nz > 1 
            +------+
               ny

            +------+
            |      |  ny     ->  ( Z, Y )    ny_over_nz < 1 
            +------+
               nz

        """
        if nx == 0 and ny > 0 and nz > 0:
            ny_over_nz = float(ny)/float(nz)
            axes = (Y,Z) if ny_over_nz > 1 else (Z,Y)
        elif nx > 0 and ny == 0 and nz > 0:
            nx_over_nz = float(nx)/float(nz)
            axes = (X,Z) if nx_over_nz > 1 else (Z,X)
        elif nx > 0 and ny > 0 and nz == 0:
            nx_over_ny = float(nx)/float(ny)
            axes = (X,Y) if nx_over_ny > 1 else (Y,X)
        else:
            axes = None
        pass
        return axes



if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)
     peta = GridSpec.DemoPeta()
     grid = GridSpec(peta)
     print(grid)


