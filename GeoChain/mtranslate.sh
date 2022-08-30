#!/bin/bash -l 

usage(){ cat << EOU
mtranslate.sh 
=====================

::

    gc
    vi mtranslate.sh    # change the list of GEOM to translate using GeoChain test machinery 
    ./mtranslate.sh 

EOU
}



geomlist_nmskSolidMaskVirtual(){ cat << EOL
nmskSolidMaskVirtual
EOL
}

geomlist_nmskSolidMask(){ cat << EOL
nmskSolidMask
nmskMaskOut
nmskTopOut
nmskBottomOut
nmskMaskIn
nmskTopIn
nmskBottomIn
EOL
}

geomlist_nmskSolidMaskTail(){ cat << EOL
nmskSolidMaskTail

nmskTailOuter
nmskTailOuterIEllipsoid
nmskTailOuterITube
nmskTailOuterI
nmskTailOuterIITube

nmskTailInner
nmskTailInnerIEllipsoid
nmskTailInnerITube
nmskTailInnerI
nmskTailInnerIITube 

EOL
}

for geom in $(geomlist_nmskSolidMaskTail) ; do 
   echo $BASH_SOURCE $geom 
   GEOM=$geom ./translate.sh 
   [ $? -ne 0 ] && echo $BASH_SOURCE translate error for geom $geom && exit 1
done 

exit 0


