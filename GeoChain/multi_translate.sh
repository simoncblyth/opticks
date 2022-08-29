#!/bin/bash -l 

usage(){ cat << EOU
multi_translate.sh 
=====================

::

    gc
    vi multi_translate.sh    # change the list of GEOM to translate using GeoChain test machinery 
    ./multi_translate.sh 

EOU
}


geomlist(){ cat << EOL
nmskSolidMaskVirtual
nmskSolidMask
nmskMaskOut
nmskTopOut
nmskBottomOut
nmskMaskIn
nmskTopIn
nmskBottomIn
nmskSolidMaskTail
EOL
}

for geom in $(geomlist) ; do 
   echo $BASH_SOURCE $geom 
   GEOM=$geom ./translate.sh 
   [ $? -ne 0 ] && echo $BASH_SOURCE translate error for geom $geom && exit 1
done 

exit 0


