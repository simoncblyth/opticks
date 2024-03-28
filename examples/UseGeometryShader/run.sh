#!/bin/bash -l 
usage(){ cat << EOU
run.sh 
======

With REC=1 find have to adhoc scale down input position coordinates 
by factor 1000 (from mm to m) to make the propagation visible ?::

    REC=1 ADHOC=0.001 T0=0 T1=200 SHADER=rec_flying_point ~/o/examples/UseGeometryShader/run.sh
    REC=1 ADHOC=0.001 T0=0 T1=200 SHADER=pos              ~/o/examples/UseGeometryShader/run.sh

With REC=0 ADHOC=0.5 the expected concentric circles are half the default size. 
This means the frame targetting is not working, it is intended to adjust 
view matrix to fill frame as needed for the provided center_extent::

    REC=0 ADHOC=0.5 T0=0 T1=10 SHADER=rec_flying_point ~/o/examples/UseGeometryShader/run.sh
    REC=0                      SHADER=rec_flying_point ~/o/examples/UseGeometryShader/run.sh

Attempt to investigate this in the below inconclusive so far::

   ~/o/sysrap/tests/SGLM_frame_targetting_test.sh


What is the output of SGLM actually used by OpenGL ?




TODO: polz colored line output, like with old Opticks 


EOU
}

DIR=$(dirname $(realpath $BASH_SOURCE))

rec=2
REC=${REC:-$rec}
case $REC in
 0) fold=/tmp/sphoton_test ;;
 1) fold=/data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL/p001 ;;
 2) fold=/data/blyth/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000 ;;
esac

echo $BASH_SOURCE REC $REC fold $fold

export RECORD_FOLD=$fold
$DIR/build.sh run

