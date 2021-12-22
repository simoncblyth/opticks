#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

class Scan(object):
    @classmethod
    def Create(cls, n, modes):
        scan = cls()
        scan.ipos = np.zeros( (n*len(modes), 3) ) 
        scan.iray = np.zeros( (n*len(modes), 2, 3) ) 
        scan.isec = np.zeros( (n*len(modes), 4) ) 
        scan.topline = "created"
        scan.botline = ""
        scan.thirdline = ""
        scan.fold = None
        return scan 

    @classmethod
    def Load(cls, fold):
        scan = cls()
        log.info("loading from %s " % fold )
        scan.ipos = np.load(os.path.join(fold, "ipos.npy"))
        scan.iray = np.load(os.path.join(fold, "iray.npy"))
        scan.isec = np.load(os.path.join(fold, "isec.npy"))
        scan.topline = fold 
        scan.botline = ""
        scan.thirdline = ""
        scan.fold = fold
        return scan 
 
    @classmethod
    def XY(cls, geom, n=100, modes=[0,1,2,3], t_min=0, shifted=True ):
        scan = cls.Create(n, modes)
        offset = 0 
        for mode in modes: 
            for i in range(n):
                j = i - n/2 
                if mode == 0:  # shoot upwards from X axis, or shifted line
                    dx = 0
                    dy = 1
                    ox = j*0.1
                    oy = -10. if shifted else 0. 
                elif mode == 1: # shoot downwards from X axis, or shifted line
                    dx = 0
                    dy = -1 
                    ox = j*0.1
                    oy = 10. if shifted else 0.  
                elif mode == 2: # shoot to right from Y axis, or shifted line 
                    dx = 1
                    dy = 0 
                    ox = -10. if shifted else 0. 
                    oy = j*0.1 
                elif mode == 3: # shoot to left from Y axis, or shifted line
                    dx = -1
                    dy = 0 
                    ox = 10. if shifted else 0.
                    oy = j*0.1 
                pass

                ray_origin    = np.array( [ox, oy, 0 ] )    
                ray_direction = np.array( [dx, dy, 0 ] )    
                isect = geom.intersect( ray_origin, ray_direction, t_min )

                scan.iray[i+offset,0] = ray_origin
                scan.iray[i+offset,1] = ray_direction

                if not isect is None:
                    scan.ipos[i+offset] = isect[3]*ray_direction + ray_origin 
                    scan.isec[i+offset] = isect 
                pass
            pass
            offset += n 
        pass
        return scan

    @classmethod
    def Plot(cls, scan):
    
        size = np.array([1280, 720])
        pl = pv.Plotter(window_size=size*2 )
        pl.view_xy() 

        pl.add_text(scan.topline, position="upper_left")
        pl.add_text(scan.botline, position="lower_left")
        pl.add_text(scan.thirdline, position="lower_right")


        limit = 100. 
        mask = np.logical_and( np.abs(scan.ipos[:,1]) < limit , np.abs(scan.ipos[:,0]) < limit )   

        isect_normals = True 
        if isect_normals:
            pl.add_arrows( scan.ipos[mask], scan.isec[mask, :3],  color="white" )  
        else:
            pl.add_points( scan.ipos[mask],  color="white" )  
        pass
        pl.add_points( scan.iray[:,0],  color="red" )  

        look = (0,0,0)
        up = (0,1,0)
        eye = (0,0, 10)

        pl.set_focus(    look )
        pl.set_viewup(   up )
        pl.set_position( eye, reset=True )   ## for reset=True to succeed to auto-set the view, must do this after add_points etc.. 
        pl.camera.Zoom(1)
        pl.show_grid()

        outpath = None
        if not scan.fold is None:
            outpath = os.path.join(scan.fold, "figs", "scanplot.png")
            outfold = os.path.dirname(outpath)
            if not os.path.isdir(outfold):
                os.makedirs(outfold)
            pass
        pass
        if not outpath is None:
            log.info("screenshot outpath %s " % outpath)  
            cp = pl.show(screenshot=outpath)
        else:
            cp = pl.show()
        pass
        return cp 


