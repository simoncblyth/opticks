#!/bin/bash -l 

adir=/tmp/$USER/opticks/snap 
bdir=/tmp/$USER/opticks/CSGOptiX/CSGOptiXRender/70000/render/CSG_GGeo/1

q=${1:-t8,}

find $adir -name "lLowerChimney_phys__*${q}__00000.jpg" 
find $bdir -name "*${q}_sWaterTube.jpg" 

