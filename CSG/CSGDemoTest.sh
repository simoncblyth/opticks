#!/bin/bash -l 

usage(){ cat << EOU
CSGDemoTest.sh 
================




EOU
}


bin=CSGDemoTest 

#geometry=parade
#geometry=sphere_containing_grid_of_spheres
#geometry=layered_sphere
#geometry=layered_zsphere
#geometry=clustered_sphere
#geometry=scaled_box3
#geometry=sphe
#geometry=zsph
#geometry=cone
#geometry=hype
#geometry=box3 
#geometry=plan 
#geometry=slab  
#geometry=cyli
#geometry=disc 
#geometry=vcub
#geometry=vtet
#geometry=elli
#geometry=ubsp
#geometry=ibsp 
#geometry=dbsp
#geometry=rcyl
#geometry=dcyl
geometry=bssc

#clusterspec=-3:4:1,-3:4:1,-3:4:1
clusterspec=-1:2:1,-1:2:1,-1:2:1

clusterunit=500

gridmodulo=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
#gridmodulo=12,13,14
#gridmodulo=9,10
#gridmodulo=5,6
#gridmodulo=10
#gridmodulo=2

#gridsingle=2
gridsingle=""

#gridspec=-10:11:2,-10:11:2,-10:11:2
#gridspec=-10:11:2,-10:11:2,0:8:2
gridspec=-10:11:2,-10:11:2,0:6:3
#gridspec=-40:41:4,-40:41:4,-40:41:4
#gridspec=-40:41:10,-40:41:10,-40:41:10
#gridspec=-40:41:10,-40:41:10,0:1:1

gridscale=200.0

# number of concentric layers in compound shapes
#layers=1     
#layers=2
#layers=3
layers=7
#layers=20

numgas=2
#numgas=5


# make sensitive to calling environment
export GEOMETRY=${GEOMETRY:-$geometry}
export CLUSTERSPEC=${CLUSTERSPEC:-$clusterspec}
export CLUSTERUNIT=${CLUSTERUNIT:-$clusterunit}
export GRIDMODULO=${GRIDMODULO:-$gridmodulo}
export GRIDSINGLE=${GRIDSINGLE:-$gridsingle}
export GRIDSPEC=${GRIDSPEC:-$gridspec}
export GRIDSCALE=${GRIDSCALE:-$gridscale}
export LAYERS=${LAYERS:-$layers}
export NUMGAS=${NUMGAS:-$numgas}

export CFBASE=/tmp/$USER/opticks/CSGDemoTest/$GEOMETRY
cfdir=$CFBASE/CSGFoundry
mkdir -p $cfdir

vars="bin GEOMETRY CLUSTERSPEC CLUSTERUNIT GRIDMODULO GRIDSINGLE GRIDSPEC GRIDSCALE LAYERS CFBASE cfdir"
for var in $vars ; do printf "%-20s : %s \n" $var ${!var} ; done

$GDB $bin $* 
[ $? -ne 0 ] && exit 1 

echo ls -l $cfdir/
ls -l $cfdir/

exit 0
