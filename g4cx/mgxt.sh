#!/bin/bash -l 


geomlist(){ cat << EOL | grep -v ^#
nmskSolidMask
nmskSolidMaskTail
nmskTailOuter
nmskTailInner
EOL
}

#opt=__U0
opt=__U1
for geom in $(geomlist) ; do  
   echo $BASH_SOURCE geom $geom opt $opt 
   GEOM=${geom}${opt} ./gxt.sh $*
   [ $? -ne 0 ] && echo $BASH_SOURCE gxt error for geom $geom && exit 1
done 

exit 0


