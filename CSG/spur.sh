#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

export CSGRecord_ENABLED=1

catgeom=$(cat ~/.opticks/GEOM.txt 2>/dev/null | grep -v \#) && [ -n "$catgeom" ] && geom=$(echo ${catgeom})

case $geom in 
   #JustOrbOrbDifference_XY)        sxyzw=-16,-3,0,99 ;;
   JustOrbOrbDifference_XY)         sxyzw=0,9,0,74    ;;
   #AnnulusTwoBoxUnionContiguous_YZ) sxyzw=0,8,8,61    ;; 
   #AnnulusTwoBoxUnionContiguous_YZ) sxyzw=0,-3,-1,4   ;;
   AnnulusTwoBoxUnionContiguous_YZ) sxyzw=0,-2,0,50   ;;
   AltXJfixtureConstruction_XY) sxyzw=-5,0,0,85       ;; 
esac

if [ -n "$sxyzw" ]; then
    export SXYZW=$sxyzw
    echo $msg defining SXYZW $SXYZW for geom $geom : selecting individual rays for rerunning with CSGRecord collection and presentation  
fi 


#export DEBUG=1

#./csg_geochain.sh run 
#./csg_geochain.sh ana
./csg_geochain.sh 


