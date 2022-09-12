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

geomlist_nmskSolidMaskTail(){ cat << EOL | grep -v ^#


nmskSolidMaskTail

nmskTailOuter
#nmskTailOuterIEllipsoid
#nmskTailOuterITube
#nmskTailOuterI
#nmskTailOuterIITube

nmskTailInner
#nmskTailInnerIEllipsoid
#nmskTailInnerITube
#nmskTailInnerI
#nmskTailInnerIITube 

EOL
}

geomlist(){ cat << EOL | grep -v ^#
nmskSolidMask
nmskSolidMaskTail
nmskTailOuter
nmskTailInner

nmskTailInnerIEllipsoid
nmskTailInnerITube
nmskTailInnerI
nmskTailInnerIITube 

nmskTailOuterIEllipsoid
nmskTailOuterITube
nmskTailOuterI
nmskTailOuterIITube

EOL
}

geomlist1(){ cat << EOL | grep -v ^#
nmskTailInnerITube
nmskTailOuterITube
EOL
}


#opt=__U0
opt=__U1
for geom in $(geomlist) ; do 
   echo $BASH_SOURCE geom $geom opt $opt 
   GEOM=${geom}${opt} ./translate.sh 
   [ $? -ne 0 ] && echo $BASH_SOURCE translate error for geom $geom && exit 1
done 

exit 0


