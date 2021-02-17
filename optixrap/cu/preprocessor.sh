#!/bin/bash -l

pp-usage(){ cat << EOU
preprocessor.sh
=================

This is just for testing the preprocessor.py script, 
actual usage directly uses the python script via  add_custom_command 
in oxrap/CMakeLists.txt 

EOU
}

pp-flags(){ cat << EOF
+ANGULAR_ENABLED,+WAY_ENABLED
+ANGULAR_ENABLED,-WAY_ENABLED
-ANGULAR_ENABLED,+WAY_ENABLED
-ANGULAR_ENABLED,-WAY_ENABLED
EOF
}

pp-cmd(){ cat << EOC
python preprocessor.py $1 --flags="$2" --out $3
EOC
}

pp--()
{
    local dir=/tmp/$USER/opticks/optixrap/cu/preprocessor
    local flags=$(pp-flags)
    local f
    local cu=generate.cu
    local stem=${cu/.cu}
   
    mkdir -p $dir  
    local cpc="cp $cu $dir/$cu"
    eval $cpc

    for f in $flags ; do
        local out=$dir/${stem}_$f.cu
        local cmd=$(pp-cmd $cu $f $out)
        echo $cmd 
        eval $cmd   
    done 
}
pp--

