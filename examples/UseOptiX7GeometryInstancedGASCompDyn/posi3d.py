#!/usr/bin/env python
import os, numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

def plot3d(pos, title):
    pl = pv.Plotter()
    #pl.add_title(title)
    pl.add_text(title, position="upper_edge")
    pl.add_points(pos, color='#FFFFFF', point_size=2.0 )  
    pl.show_grid()
    cp = pl.show()
    return cp


np.set_printoptions(suppress=True)

dir_ = lambda:os.environ.get("OUTDIR", os.getcwd())
path_ = lambda name:os.path.join(dir_(), name)
load_ = lambda name:np.load(path_(name))


if __name__ == '__main__':

    base = dir_()
    print(base)
    title = "/".join(base.split("/")[-1:])  
    print(title)

    posi = load_("posi.npy")
    hposi = posi[posi[:,:,3] != 0 ]  
    iposi = hposi[:,3].view(np.uint32)  

    plot3d( hposi[:,:3], title )
    print(dir_())



