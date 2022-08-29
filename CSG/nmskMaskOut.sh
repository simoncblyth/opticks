#!/bin/bash -l 
usage(){ cat << EOU
nmskMaskOut.sh
==================
To temporatily look at a GEOM::    

    GEOM=nmskSolidMask ./gxt.sh ana 

EOU
}

echo $BASH_SOURCE 

#selection=294748

GEOM=nmskMaskOut SELECTION=$selection ./CSGSimtraceRerunTest.sh $* 

