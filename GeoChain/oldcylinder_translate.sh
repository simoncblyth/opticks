#!/bin/bash -l 
usage(){ cat << EOU
oldcylinder_translate.sh
===========================

NB after checking some geometry with CSG_OLDCYLINDER remember
to re-translate without the old cylinder by rerunning this 
with the X4Solid__convertTubs_cylinder_old_cylinder_implementation 
envvar commented. 


HMM somewhat confusingly had to override GEOM to get the oldcylinder, maybe from trim chopping the underscore?:: 

    X4Solid__convertTubs_cylinder_old_cylinder_implementation=1 GEOM=nmskSolidMaskTail__U1 ./translate.sh 
    X4Solid__convertTubs_cylinder_old_cylinder_implementation=1 GEOM=nmskSolidMaskTail__U1 

EOU
}

#export X4Solid__convertTubs_cylinder_old_cylinder_implementation=1

geomlist(){ cat << EOU | grep -v ^#
nmskSolidMaskTail__U1
nmskTailOuterITube__U1
EOU
}
for geom in $(geomlist) ; do 
   echo $BASH_SOURCE : $geom 
   GEOM=$geom ./translate.sh 
done


