#!/bin/bash -l 
usage(){ cat << EOU
cxs0.sh : hybrid rendering/simulation machinery
===================================================

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

cxs=${CXS:-2}   # collect sets of config underneath CXS

if [ "$cxs" == "1" ]; then
    moi=Hama
    cegs=16:0:9:1000:18700:0:0:100
    gridscale=1.0
elif [ "$cxs" == "2" ]; then
    moi=uni_acrylic3
    cegs=16:0:9:1000
    #cegs=0:0:0:1000
    gridscale=0.05
fi 

export MOI=${MOI:-$moi}
export CEGS=${CEGS:-$cegs}
export GRIDSCALE=${GRIDSCALE:-$gridscale}
export CXS=${CXS:-$cxs}
export TOPLINE="CSGOptiXSimulateTest CXS $CXS MOI $MOI CEGS $CEGS GRIDSCALE $GRIDSCALE ISEL $ISEL"

if [ "$1" == "py" ]; then 
    ipython --pdb -i tests/CSGOptiXSimulateTest.py 
else
    CSGOptiXSimulateTest
fi 

