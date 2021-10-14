#!/usr/bin/env python
"""
tests/CSGOptiXSimulate.py
============================

This is useful as it allows interactive visualization of workstation 
generated intersect data fphoton.npy on remote machines such as 
user laptops that support pyvista. 
 
But that raises the strong possibility that the geocache+CSGFoundry 
used on the laptop does not match that used on the workstation to generate
the fphoton.npy intersects

* this will cause incorrect identity information for intersects. 
* the problem is especially acute when multiple geometries are in use
  and geometries are changed frequently 


AVOIDING INCORRECT IDENTITY INFO ?
------------------------------------

* digest info json metadata accompanying the intersects and foundry would allows the 
  situation to be detected.  

* transfering the CSGFoundry folder from workstation, so the identitity backing information is 
  guaranteed to be consistent with the intersects even as geometry is frequently changed...
  with this in mind think about the size of the CSGFoundry/geocache what can be skipped 
  and what is essential for identity info

::

    cg ; ./grab.sh 



FOUNDRY IDENTITY INFO
------------------------

Currently the only names CSGFoundry holds are mesh names


::

    cx ; ipython -i tests/CSGOptiXSimulate.py


__closesthit__ch::

    331     unsigned instance_idx = optixGetInstanceId() ;    // see IAS_Builder::Build and InstanceId.h 
    332     unsigned prim_idx  = optixGetPrimitiveIndex() ;  // see GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    333     unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_idx & 0xffff ) ;

    prim_idx = ( i >> 16 )      ## index of bbox within within the GAS 
    instance_idx = i & 0xffff   ## flat 

NB getting zero for the flat instance_idx (single IAS, all transforms in it) 
**DOES** tell you that its a global intersect 

Now how to lookup what a prim_id corresponds to ?

"""
import os, numpy as np
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)
from opticks.CSG.CSGFoundry import CSGFoundry 
import matplotlib.pyplot as plt

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

class CSGOptiXSimulate(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/CSGOptiX/CSGOptiXSimulate")
    def __init__(self, fold=FOLD):
        names = os.listdir(fold)
        for name in filter(lambda n:n.endswith(".npy"),names):
            path = os.path.join(fold, name)
            stem = name[:-4]
            a = np.load(path)
            print(" %10s : %15s : %s " % (stem, str(a.shape), path )) 
            globals()[stem] = a
        pass



def look_photon(p):
    np.all( p[:,0,1] == p[0,0,1] )   # all positions expected to be at same y are using planar x-z gensteps 

    i = p[:,3,3].view(np.uint32)
    ui,ui_counts = np.unique(i, return_counts=True)

    print(ui)
    print(ui_counts)

    prim_idx = ( i >> 16 ) 
    instance_id = i & 0xffff
 
    print("prim_idx")
    print(prim_idx)
    print("instance_id")
    print(instance_id)

    boundary = p[:,2,3].view(np.uint32)

    print("boundary")
    print(boundary)




if __name__ == '__main__':

    cf = CSGFoundry()
    cxs = CSGOptiXSimulate()

    f = fphoton
    print(f)

    b = f.view(np.uint32)[:,:,2,3]    # boundary   
    i = f.view(np.uint32)[:,:,3,3]    # identity  

    uis, uis_counts = np.unique(i, return_counts=True)    
    ii = i[i>0]     

    prim_idx = i >> 16    
    instance_id = i & 0xffff   

    pick_b = b[b>0] 
    #pick_b = b[instance_id == 0]    # global boundaries
    #pick_b = b[instance_id > 0]    # instance boundaries


    ubs, ubs_counts = np.unique(pick_b, return_counts=True)   
    # hmm b=0 is meaningful, but is swamped by no-hits : need to make it 1-based 

    print("ubs",ubs)
    print("ubs_counts",ubs_counts)
    colors = ["red","green","blue","cyan","magenta","yellow","pink","purple"]


    fpos = f[b>0][:,0,:3]

    
    size = np.array( [1024, 768] )*2
    eye =  (94006.38845416412, 86640.65749713287, 95402.39480182037)
    look = (7365.73095703125, 0.0, 8761.7373046875)
    up = (0.0, 0.0, 1.0)

    pl = pv.Plotter(window_size=size )
    pl.view_xz() 
   
    pl.camera.ParallelProjectionOn()  

    eye = (17700, -17700, 0 )
    look = (17700, 0, 0 )
    up = (0,0,1)
    #scale = 100 

    pl.set_position( eye, reset=False )
    pl.set_focus(    look )
    pl.set_viewup(   up )
    #pl.set_scale( scale )


    # genstep grid
    #pl.add_points( g[:,1,:3] , color="white" )


    #pl.camera.position = (0, -17000, 0.0)
    #pl.camera.focal_point = (0, 0, 0)
    #pl.camera.up = (0.0, 0.0, 1.0)


    for ub, ub_count in zip(ubs, ubs_counts):
        color = colors[ub % len(colors)]
        bname = cf.bndname[ub]
        #if bname != "Water///Acrylic": continue

        print( " %4d : %6d : %10s : %40s " % (ub, ub_count, color, bname ))            
        fpos = f[b==ub][:,0,:3]

        pl.add_points( fpos, color=color )
    pass

    #pl.show_grid()
    cp = pl.show()




