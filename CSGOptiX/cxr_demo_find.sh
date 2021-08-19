#!/bin/bash -l 

geometry=layered_sphere
export GEOMETRY=${1:-$geometry}

outbase=/tmp/$USER/opticks/CSGOptiX/CSGOptiXRender/CSGDemoTest
find $outbase -name "cxr_demo_${GEOMETRY}_*.jpg" 

