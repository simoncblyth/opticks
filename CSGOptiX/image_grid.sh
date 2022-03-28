#!/bin/bash -l 

usage(){ cat << EOU
image_grid.sh
===============

Create list of jpg using eg::

    cx
    ./grabsnap.sh --jpg --out --outpath /tmp/ana_snap.txt 
    ./grabsnap.sh --jpg --out
         ## uses default outpath of /tmp/ana_snap.txt 

    ./grabsnap.sh --jpg --reverse --out --selectspec not_elv_t --outpath /tmp/ana_snap.txt

    SNAP_LIMIT=64 ./grabsnap.sh --jpg --reverse --out --selectspec not_elv_t --outpath /tmp/ana_snap.txt


Note that default maximum open files on macOS appears to be only 256 


This makes an enormous image in memory before scaling back to the size of the input images::

    epsilon:CSGOptiX blyth$ ./image_grid.sh 
    INFO:__main__:all_paths 137 all_anno 0 grid.shape (12, 12) 
    INFO:__main__:IMG.downsize factor 12 from width 23040 height 12960 to dwidth 1920 dheight 1080 
    INFO:__main__:save to /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGOptiXRenderTest/cvd1/70000/cxr_overview/cam_0_tmin_0.4/image_grid_elv_scan.jpg 
    epsilon:CSGOptiX blyth$ 


The save path is based on the path of the first image.

EOU
}


make_image_grid()
{
    local outstem=$1
    local pathlist=$2
    local annolist=$3

    export ANNOTATE=1    
    export OUTSTEM=$outstem

    ${IPYTHON:-ipython} -- ~/env/doc/image_grid.py $pathlist $annolist
}


make_image_grid "image_grid_elv_scan" /tmp/ana_snap.txt 




