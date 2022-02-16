#!/bin/bash -l 
msg="=== $BASH_SOURCE :"
usage(){ cat << EOU
csg_geochain.sh : CPU Opticks equivalent to cxs_geochain.sh with OptiX on GPU using test/CSGIntersectSolidTest.cc tests/CSGIntersectSolidTest.py
==================================================================================================================================================

The idea behind this is to provide a convenient way to test code primarily intended to run on GPU 
in the more friendly debugging environment of the CPU.::

    IXYZ=4,0,0 ./csg_geochain.sh  ana
        selecting single genstep with IXYZ 

    IXYZ=5,0,0 ./csg_geochain.sh  ana

    IXYZ=0,6,0 ./csg_geochain.sh  ana      


    SPURIOUS=1 GEOM=CylinderFourBoxUnion_YX ./csg_geochain.sh ana
         56/20997 spurious intersects all exiting the cylinder 
         within the +Y and -X boxes

         Curious that are no such issues with the other two boxes
         despite the symmetry. 

         TODO: test with CSG balancing disabled

    SPURIOUS=1 GEOM=AnnulusFourBoxUnion_YX ./csg_geochain.sh ana
          181/20997 spurious all exiting the cylinder with all 4 boxes

    SPURIOUS=1 IXYZ=-6,0,0 GEOM=AnnulusFourBoxUnion_YX ./csg_geochain.sh ana
          select single genstep   

    SPURIOUS=1 IXYZ=-6,0,0 IW=17 GEOM=AnnulusFourBoxUnion_YX SAVE_SELECTED_ISECT=1 ./csg_geochain.sh ana

    SELECTED_ISECT=/tmp/selected_isect.npy GEOM=AnnulusFourBoxUnion_YX ./csg_geochain.sh run
          run again just with selected isects, with CSGRecord enabled 
          will need to use same geometry to get same results     

    SPURIOUS=1 IXYZ=-6,0,0 IW=17 GEOM=AnnulusFourBoxUnion_YX  ./csg_geochain.sh ana


    TMIN=50 ./csg_geochain.sh 
        NB can only change TMIN at C++ level not ana level 

        Note that TMIN is currently absolute, it is not extent relative like with cxr_geochain.sh  
        rendereing   


    IXYZ=8,0,0 TMIN=50 ./csg_geochain.sh ana
        at python analysis level the highlighted genstep can be changed using IXYZ 
        TMIN does nothing at analysis level the intersects from the prior run are loaded and plotted
           
    SXYZW=4,4,0,80 ./csg_geochain.sh run
        re-run with a single photon 
        do this after recompiling with DEBUG flag allows to see the details of the single photon 



     


EOU
}

#geom=AltXJfixtureConstruction_YZ
#geom=AltXJfixtureConstruction_XZ
#geom=AltXJfixtureConstruction_XY

#geom=AnnulusBoxUnion_XY
#geom=AnnulusTwoBoxUnion_XY
#geom=AnnulusOtherTwoBoxUnion_XY

#geom=AnnulusFourBoxUnion_XY
#geom=AnnulusFourBoxUnion_YX
#geom=CylinderFourBoxUnion_YX
#geom=AnnulusCrossTwoBoxUnion_YX

#geom=BoxFourBoxUnion_YX
#geom=BoxFourBoxContiguous_YX
#geom=SphereWithPhiCutDEV_YX

#geom=BoxCrossTwoBoxUnion_YX
#geom=BoxThreeBoxUnion_YX

#geom=XJfixtureConstruction_YZ
#geom=XJfixtureConstruction_XZ
#geom=XJfixtureConstruction_XY

#geom=iphi_YX

#geom=ithe_XZ
#geom=ithl_XZ

#geom=ithe_YZ
#geom=ithl_YZ

#geom=ithe_XYZ
#geom=ithl_XYZ

#geom=GeneralSphereDEV_XZ
#geom=GeneralSphereDEV_XYZ
#geom=GeneralSphereDEV_XZ
#geom=GeneralSphereDEV_YZ
#geom=GeneralSphereDEV_XY

#geom=OverlapBoxSphere_XY
#geom=ContiguousBoxSphere_XY
#geom=DiscontiguousBoxSphere_XY
#geom=IntersectionBoxSphere_XY

geom=OverlapThreeSphere_XY
#geom=ContiguousThreeSphere_XY



#catgeom=$(cat ~/.opticks/GEOM.txt 2>/dev/null | grep -v \#) && [ -n "$catgeom" ] && echo $msg catgeom $catgeom override of default geom $geom && geom=${catgeom} 

export GEOM=${GEOM:-$geom}
gcn=${GEOM%%_*}   ## name up to the first underscore, assuming use of axis suffix  _XZ _YZ _XY _ZX _ZY _YX 

if [ "$(uname)" == "Darwin" ] ; then
   cfbase=$TMP/GeoChain_Darwin/$gcn 
else
   cfbase=$TMP/GeoChain/$gcn 
fi 

case $gcn in 
   AnnulusFourBoxUnion)     gridscale=0.1  ;; 
   AnnulusCrossTwoBoxUnion) gridscale=0.1  ;; 
  CylinderFourBoxUnion)     gridscale=0.1  ;; 
   BoxCrossTwoBoxUnion)     gridscale=0.07 ;; 
   BoxThreeBoxUnion)        gridscale=0.07 ;; 
   BoxFourBoxUnion)         gridscale=0.07 ;; 
                     *)     gridscale=0.15 ;;
esac

case $GEOM in 
       AnnulusFourBoxUnion_YX) note="see spurious with IXYZ=5,0,0 and 0,6,0 " ;;
   AnnulusCrossTwoBoxUnion_YX) note="no spurious despite same apparent geom on ray path" ;;
      CylinderFourBoxUnion_YX) note="see spurious" ;;
           BoxThreeBoxUnion_YX) note="no spurious" ;; 
           BoxFourBoxUnion_YX) note="see spurious in the +Y and -X small boxes " ;; 
          GeneralSphereDEV_XY) note="phicut bug from \"t_cand <  t_min\" that should be \"t_cand <= t_min \" : SPHI selects spurious in bad phi range "  ;;
                            *) note="" ;;
esac

dx=0
dy=0
dz=0
pho=${PHO:--100} 

case $pho in
  -*)  echo $msg using regular bicycle spoke photon directions ;; 
   *)  echo $msg using random photon directions                ;;
esac

case $GEOM in  
   *_XZ) cegs=16:0:9:$dx:$dy:$dz:$pho  ;;  
   *_YZ) cegs=0:16:9:$dx:$dy:$dz:$pho  ;;  
   *_XY) cegs=16:9:0:$dx:$dy:$dz:$pho  ;;  
   *_ZX) cegs=9:0:16:$dx:$dy:$dz:$pho  ;;  
   *_ZY) cegs=0:9:16:$dx:$dy:$dz:$pho  ;;  
   *_YX) cegs=9:16:0:$dx:$dy:$dz:$pho  ;;  
   *_XYZ) cegs=9:16:9:$dx:$dy:$dz:$pho ;;  
       *) echo $msg UNEXPECTED SUFFIX FOR GEOM $GEOM WHICH DOES NOT END WITH ONE OF : _XZ _YZ _XY _ZX _ZY _YX _XYZ  && exit 1   ;; 
esac

echo $msg GEOM $GEOM gcn $gcn gridscale $gridscale ixiyiz $ixiyiz


topline="GEOM=$GEOM ./csg_geochain.sh "
[ -n "$SPHI" ] && topline="SPHI=$SPHI $topline" 
[ -n "$IXYZ" ] && topline="IXYZ=$IXYZ $topline" 

cmdline="GEOM=$GEOM ./csg_geochain.sh "
[ -n "$SPHI" ] && cmdline="SPHI=$SPHI $cmdline" 
[ -n "$IXYZ" ] && cmdline="IXYZ=$IXYZ $cmdline" 
[ -n "$SPURIOUS" ] && cmdline="SPURIOUS=$SPURIOUS $cmdline" 

export CMDLINE=$cmdline
export NOTE=$note 
export GRIDSCALE=${GRIDSCALE:-$gridscale}
export CEGS=${CEGS:-$cegs}
export CFBASE=${CFBASE:-$cfbase}
export CEGS=${CEGS:-$cegs}
export TOPLINE="$topline"
export BOTLINE="$note"
export THIRDLINE="CEGS=$CEGS"

check_cegs()
{
    local msg="=== $FUNCNAME :"
    IFS=: read -a cegs_arr <<< "$CEGS"
    local cegs_elem=${#cegs_arr[@]}

    case $cegs_elem in
       4) echo $msg 4 element CEGS $CEGS ;;
       7) echo $msg 7 element CEGS $CEGS ;;
       *) echo $msg ERROR UNEXPECTED $cegs_elem element CEGS $CEGS && return 1  ;;
    esac
    return 0 
}


check_cfbase_how_to_create(){ cat << EOH

How to Create CSGFoundry Geometry
======================================

A. From converted G4VSolid 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply GeoChain conversion to a named geometry::

    b7  # when using OptiX 7
    cd ~/opticks/GeoChain
    GEOM=${GEOM%%_*} ./run.sh 

B. Directly from CSGSolid/CSGPrim/CSGNode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and persists into CSGFoundry folders one OR all CSGMaker solids::

    cd ~/opticks/CSG

    CSGMakerTest     
    GEOM=${GEOM%%_*} CSGMakerTest


EOH
}


check_cfbase()
{
    local msg="=== $FUNCNAME :"
    if [ -n "$cfbase" -a ! -d "$cfbase/CSGFoundry" ]; then

       echo $msg : ERROR cfbase $cfbase is defined signalling to use a non-standard CSGFoundry geometry 
       echo $msg : BUT no such CSGFoundry directory exists 
       echo $msg :
       echo $msg : Possibilities: 
       echo $msg :
       echo $msg : 1. you intended to use the standard geometry but the GEOM $GEOM envvar does not match any of the if branches 
       echo $msg : 2. you want to use a non-standard geometry but have not yet created it
       echo $msg :
       check_cfbase_how_to_create 
       return 1
    fi
    return 0 
}

check_cfbase_file()
{
    local msg="=== $FUNCNAME :"
    if [ -n "$cfbase" -a -d "$cfbase/CSGFoundry" -a ! -f "$cfbase/CSGFoundry/meshname.txt" ]; then
       echo $msg : ERROR cfbase $cfbase is defined and the $cfbase/CSGFoundry directory exists
       echo $msg : BUT it misses expected files such as $cfbase/CSGFoundry/meshname.txt 
       echo $msg :
       check_cfbase_how_to_create 
       return 1 
    fi 
    return 0 
}

dumpvars(){ for var in $* ; do printf "%20s : %s \n" $var "${!var}" ; done ; }

check_cegs        || exit 1 
check_cfbase      || exit 1 
check_cfbase_file || exit 1 
dumpvars GEOM CEGS GRIDSCALE TOPLINE BOTLINE CFBASE NOTE IXYZ 

bin=CSGIntersectSolidTest
script=tests/CSGIntersectSolidTest.py 

arg=${1:-run_ana}

if [ "${arg/dump}" != "$arg" ]; then 
   echo $msg CSGGeometryTest dump
   CSGGeometryTest
   exit 0 

elif [ "${arg/run}" != "$arg" ]; then

    echo $msg running binary $bin
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 1

elif [ "${arg/dbg}" != "$arg" ]; then 

    echo $msg running binary $bin under debugger
    if [ "$(uname)" == "Darwin" ]; then
        lldb__ $bin
    else
        gdb $bin
    fi 
fi

if [ "${arg/ana}" != "$arg" ]; then
    echo $msg running script $script
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $msg script error && exit 2

    figdir=$cfbase/$bin/$GEOM/figs
    ls -l $figdir 

    if [ -n "$PUB" ]; then 

       reldir=/env/presentation/CSG/$bin/$GEOM/figs 
       pubdir=$HOME/simoncblyth.bitbucket.io$reldir

       figname=out.png 
       pubname=${GEOM}_${PUB}.png       

       echo $msg figdir $figdir
       echo $msg reldir $reldir
       echo $msg pubdir $pubdir
       echo $msg pubname $pubname

       if [ ! -d "$pubdir" ]; then 
          mkdir -p $pubdir
       fi  

       cmd="cp $figdir/$figname $pubdir/$pubname"
       echo $msg cmd $cmd
       eval $cmd
       echo $msg rel $reldir/$pubname
    fi 

fi

exit 0
