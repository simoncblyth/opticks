#!/bin/bash -l 
usage(){ cat << EOU
cxs.sh : hybrid rendering/simulation machinery, eg creating 2D ray trace cross sections
========================================================================================

::

    ISEL=0,1,3,4,5 ./cxs.sh py  # select which boundaries to include in plot 


Two envvars MOI and CEGS configure the gensteps.

The MOI string has form meshName:meshOrdinal:instanceIdx 
and is used to lookup the center extent from the CSGFoundry 
geometry. Examples::


    MOI=Hama
    MOI=Hama:0:0   

    CEGS=16:0:9:200                 # nx:ny:nz:num_photons

    CEGS=16:0:9:200:17700:0:0:200   # nx:ny:nz:num_photons:cx:cy:cz:ew

The CEGS envvar configures an *(nx,ny,nz)* grid from -nx->nx -ny->ny -nz->nz
of integers which are used to mutiply the extent from the MOI center-extent.
The *num_photons* is the number of photons for each of the grid gensteps.

* as the gensteps are currently xz-planar it makes sense to use *ny=0*
* to get a non-distorted jpg the nx:nz should follow the aspect ratio of the frame 

::

    In [1]: sz = np.array( [1920,1080] )
    In [5]: 9*sz/1080
    Out[5]: array([16.,  9.])

Instead of using the center-extent of the MOI selected solid, it is 
possible to directly enter the center-extent in integer mm for 
example adding "17700:0:0:200"

As the extent determines the spacing of the grid of gensteps, it is 
good to set a value of slightly less than the extent of the smallest
piece of geometry to try to get a genstep to land inside. 
Otherwise inner layers can be missed. 

EOU
}

msg="=== $BASH_SOURCE : "
cxs=${CXS:-100}         # collect sets of config underneath CXS
cfbase=$TMP/CSG_GGeo   # default CSGFoundry dir is within cfbase 
isel=

if [ "$cxs" == "1" ]; then
    moi=Hama
    #cegs=16:0:9:1000:18700:0:0:100
    cegs=16:0:9:500
    gridscale=0.05
elif [ "$cxs" == "2" ]; then
    moi=uni_acrylic3
    cegs=16:0:9:100
    #cegs=0:0:0:1000
    #cegs=16:4:9:100
    gridscale=0.05
elif [ "$cxs" == "4" ]; then
    moi=uni_acrylic3
    cegs=32:0:18:100
    gridscale=0.025
elif [ "$cxs" == "20" ]; then
    note="very tight grid to get into close corners"
    moi=uni_acrylic3
    cegs=16:0:9:100
    gridscale=0.025
elif [ "$cxs" == "25" ]; then
    cfbase=$TMP/CSGDemoTest/dcyl    
    moi=0
    cegs=16:0:9:100
    gridscale=0.025
    isel=0                           # setting isel to zero, prevents skipping bnd 0 
elif [ "$cxs" == "30" ]; then
    note="HMM : box minus sub-sub cylinder NOT showing the spurious intersects"
    cfbase=$TMP/CSGDemoTest/bssc    
    moi=0
    cegs=16:0:9:100
    gridscale=0.025
    isel=0                           # setting isel to zero, prevents skipping bnd 0 
elif [ "$cxs" == "100" ]; then
    cfbase=$TMP/GeoChain/AdditionAcrylicConstruction  
    moi=0
    cegs=16:0:9:100
    gridscale=0.1
    isel=0
fi 

if [ ! -d "$cfbase/CSGFoundry" ]; then
   echo $msg : ERROR : cfbase directory $cfbase MUST contain CSGFoundry subfolder 
   exit 1 
fi 


export MOI=${MOI:-$moi}
export CEGS=${CEGS:-$cegs}
export GRIDSCALE=${GRIDSCALE:-$gridscale}
export CXS=${CXS:-$cxs}
export TOPLINE="cxs.sh CSGOptiXSimulateTest CXS $CXS MOI $MOI CEGS $CEGS GRIDSCALE $GRIDSCALE ISEL $ISEL ZZ $ZZ"
export BOTLINE="ZOOM $ZOOM LOOK $LOOK"
export CFBASE=${CFBASE:-$cfbase}
export ISEL=${ISEL:-$isel}

unset OPTICKS_KEY 

if [ "$1" == "run" ]; then
    $GDB CSGOptiXSimulateTest
elif [ "$1" == "ana" -o "$(uname)" == "Darwin" ]; then 
    ${IPYTHON:-ipython} --pdb -i tests/CSGOptiXSimulateTest.py 
else
    $GDB CSGOptiXSimulateTest
fi 

