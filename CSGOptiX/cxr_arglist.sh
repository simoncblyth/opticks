#!/bin/bash -l 

usage(){ cat << EON
cxr_arglist.sh
===============

Use the CSGOptiXRenderTest --arglist option 
to create multiple snapshot views of a geometry  
from a single geometry load. 

Note that with repeated global geometry that 
is currently not instanced such as solidXJfixture
the apparent viewpoints are all over the place despite a fixed
eye, look, up because there is no instance transform
dedicated to the target geometry instead there is
only the global identity transform.  

EON
}


moi=solidXJfixture
min=0
max=63

MOI=${MOI:-$moi}
MIN=${MIN:-$min}
MAX=${MAX:-$max}

global_repeat_arglist(){     for i in $(seq $MIN $MAX ) ; do echo $MOI:$i    ; done ; }
tangential_repeat_arglist(){ for i in $(seq $MIN $MAX ) ; do echo $MOI:$i:-3 ; done ; }

all_mname_arglist()
{
    local mname=$(Opticks_getFoundryBase)/CSGFoundry/meshname.txt 
    #cat $mname | grep -v Flange | grep -v _virtual | sort | uniq | perl -ne 'm,(.*0x).*, && print "$1\n" ' -  
    cat $mname | grep -v Flange | grep -v _virtual | sort | uniq 
}


path=$TMP/cxr_global/$MOI.txt
mkdir -p $(dirname $path)

if [ "$MOI" == "ALL" ]; then
    all_mname_arglist > $path
else
    tangential_repeat_arglist  > $path
    #global_repeat_arglist  > $path
fi

echo path $path 
cat $path


EYE=2,-1,-1 UP=1,0,0 TMIN=0.1 CAM=1 MOI=$MOI ARGLIST=$path ./cxr.sh 

