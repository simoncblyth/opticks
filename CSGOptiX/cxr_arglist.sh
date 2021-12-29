#!/bin/bash -l 

usage(){ cat << EON
cxr_arglist.sh
===============

Use the CSGOptiXRenderTest --arglist option 
to create multiple snapshot views of a geometry  
from a single geometry load. 

EON
}


moi=solidXJfixture
min=0
max=63

MOI=${MOI:-$moi}
MIN=${MIN:-$min}
MAX=${MAX:-$max}

global_repeat_arglist(){ for i in $(seq $MIN $MAX ) ; do echo $MOI:$i ; done ; }

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
    global_repeat_arglist  > $path
fi

echo path $path 
cat $path


MOI=$MOI ARGLIST=$path ./cxr.sh 

