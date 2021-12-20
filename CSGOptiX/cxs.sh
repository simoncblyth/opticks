#!/bin/bash -l 
usage(){ cat << EOU
cxs.sh : hybrid rendering/simulation machinery, eg creating 2D ray trace cross sections
========================================================================================

::

    ISEL=0,1,3,4,5 ./cxs.sh ana       # select which boundaries to include in plot 

    XX=-208,208 ZZ=-15.2,15.2 ./cxs.sh 

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

#geom=Hama_1
#geom=Hama_2
#geom=Hama_4
geom=Hama_8
#geom=Hama_16


#geom=uni_acrylic3_0
#geom=uni_acrylic1_0
export GEOM=${GEOM:-$geom}

isel=
cfbase=
gsplot=1

if [ "$GEOM" == "Hama_1" ]; then

    moi=Hama
    cegs=16:0:9:500
    gridscale=0.10
    gsplot=1

elif [ "$GEOM" == "Hama_2" ]; then

    moi=Hama
    cegs=32:0:18:500
    gridscale=0.10
    gsplot=1

elif [ "$GEOM" == "Hama_4" ]; then

    moi=Hama
    cegs=64:0:36:100
    #gridscale=0.10
    gridscale=0.20
    gsplot=0

elif [ "$GEOM" == "Hama_8" ]; then

    moi=Hama
    cegs=128:0:72:100
    gridscale=0.40
    gsplot=0

elif [ "$GEOM" == "Hama_16" ]; then

    ##  CUDA error on synchronize with error 'an illegal memory access was encountered' (/data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:342)
    moi=Hama
    cegs=256:0:144:100
    gridscale=0.20
    gsplot=0


elif [ "$GEOM" == "uni_acrylic1_0" ]; then
    moi=uni_acrylic1
    cegs=16:0:9:100
    gridscale=0.05
elif [ "$GEOM" == "uni_acrylic3_0" ]; then
    ## when use the option --additionacrylic-simplify-csg the uni_acrylic3 is not present 
    ## instead get uni_acrylic1 : is that OK? 
    moi=uni_acrylic3
    cegs=16:0:9:100
    #cegs=0:0:0:1000
    #cegs=16:4:9:100
    gridscale=0.05
elif [ "$GEOM" == "uni_acrylic3_wide" ]; then
    moi=uni_acrylic3
    cegs=32:0:18:100
    gridscale=0.025
elif [ "$GEOM" == "uni_acrylic3_tight" ]; then
    note="very tight grid to get into close corners"
    moi=uni_acrylic3
    cegs=16:0:9:100
    gridscale=0.025

elif [ "$GEOM" == "25" ]; then
    cfbase=$TMP/CSGDemoTest/dcyl    
    moi=0
    cegs=16:0:9:100
    gridscale=0.025
    isel=0                           # setting isel to zero, prevents skipping bnd 0 
elif [ "$GEOM" == "30" ]; then
    note="HMM : box minus sub-sub cylinder NOT showing the spurious intersects, maybe nice round demo numbers effect"
    cfbase=$TMP/CSGDemoTest/bssc    
    moi=0
    cegs=16:0:9:100
    gridscale=0.025
    isel=0                           # setting isel to zero, prevents skipping bnd 0 
elif [ "$GEOM" == "100" ]; then
    cfbase=$TMP/GeoChain/AdditionAcrylicConstruction  
    moi=0
    cegs=16:0:9:100
    gridscale=0.1
    isel=0
elif [ "$GEOM" == "101" ]; then
    cfbase=$TMP/GeoChain/BoxMinusTubs1
    moi=0
    cegs=16:0:9:100
    gridscale=0.1
    isel=0

elif [ "$GEOM" == "SphereWithPhiSegment" ]; then
    cfbase=$TMP/GeoChain/$GEOM
    num_pho=100
    cegs=9:16:0:0:0:$dz:$num_pho
    gridscale=0.10

else
    # everything else assume single PMT dimensions
    cfbase=$TMP/GeoChain/$GEOM
    moi=0
    dz=-4
    num_pho=100
    #cegs=16:0:9:0:0:$dz:$num_pho
    cegs=9:0:16:0:0:$dz:$num_pho
    #gridscale=0.15
    gridscale=0.10
    isel=
    unset CXS_OVERRIDE_CE
    export CXS_OVERRIDE_CE=0:0:-130:320   ## fix at the full uncut ce 

    #zz=190,0,-5,-162,-195,-210,-275,-350,-365,-400,-420,-450
    #xx=-254,254,-190,190
    zz=190,-450         #  450+190 = 640 m_pmt_h     190 = m_z_equator     640-190 = 450 
    xx=254,-254         #  m_pmt_r

    # 640/2 = 320
    #  190 - 320 = -130  

fi 

if [ "$(uname)" == "Linux" ]; then
    if [ -n "$cfbase" -a ! -d "$cfbase/CSGFoundry" ]; then
       echo $msg : ERROR : cfbase directory $cfbase MUST contain CSGFoundry subfolder 
       echo $msg : TIPS : run GeoChain first to create the geometry and use b7 to build CSGOptiX 
       exit 1 
    fi 
fi

export MOI=${MOI:-$moi}
export CXS_CEGS=${CXS_CEGS:-$cegs}
export GRIDSCALE=${GRIDSCALE:-$gridscale}
export TOPLINE="cxs.sh CSGOptiXSimulateTest CXS $CXS MOI $MOI CXS_CEGS $CXS_CEGS GRIDSCALE $GRIDSCALE ISEL $ISEL"
export BOTLINE="ZOOM $ZOOM LOOK $LOOK ZZ $ZZ XX $XX GEOM $GEOM "
export GSPLOT=${GSPLOT:-$gsplot}

if [ -n "$cfbase" ]; then 
    echo $msg cfbase $cfbase defined setting CFBASE to override standard geometry default 
    export CFBASE=${CFBASE:-$cfbase}   ## setting CFBASE only appropriate for non-standard geometry 
fi 

export ISEL=${ISEL:-$isel}
export XX=${XX:-$xx}
export ZZ=${ZZ:-$zz}

opticks_keydir_grabbed_default=.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1
opticks_keydir_grabbed=${OPTICKS_KEYDIR_GRABBED:-$opticks_keydir_grabbed_default} 
export FOLD=$HOME/$opticks_keydir_grabbed/CSG_GGeo



if [ "$(uname)" == "Linux" ]; then 

    if [ "$1" == "run" ]; then
        $GDB CSGOptiXSimulateTest
    elif [ "$1" == "ana" ]; then 
        NOGUI=1 ${IPYTHON:-ipython} tests/CSGOptiXSimulateTest.py 
    else
        $GDB CSGOptiXSimulateTest
        NOGUI=1 ${IPYTHON:-ipython} tests/CSGOptiXSimulateTest.py 
    fi

elif [ "$(uname)" == "Darwin" ]; then

    if [ "$1" == "bat" ]; then
        NOGUI=1 ${IPYTHON:-ipython} --pdb -i tests/CSGOptiXSimulateTest.py 
    else
        ${IPYTHON:-ipython} --pdb -i tests/CSGOptiXSimulateTest.py 
    fi 

fi 




