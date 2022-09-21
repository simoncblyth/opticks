#!/bin/bash 
usage(){ cat << EOU
geomlist.sh
=============

To test::

    source ../bin/geomlist_test.sh 


Sourcing this script defines the geomlist bash function
when an argment is provided the corresponding function is run. 

Inputs:

geomlist_OPT:var 
    suffix that is appended to geom names following two underscores
geomlist_FOLD:var
    directory path format of the geometry fold 

Example Output envvars::

   SYMBOLS=STU
   S_GEOM
   S_FOLD
   T_GEOM
   T_FOLD
   U_GEOM
   U_FOLD

The function exports the list of geometry names from the *geomlist* bash 
function into a set of envvars that communicate the 
geometry folders, names and symbols in a form that is understood by 
the python ana/fold.py Fold.MultiLoad 

EOU
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

geomlist_short(){ cat << EOL | grep -v ^#
nmskSolidMaskVirtual
nmskTailInnerITube
nmskTailOuterITube
EOL
}

geomlist_old(){ cat << EOL | grep -v ^#
nmskSolidMask
nmskSolidMaskTail
nmskSolidMaskVirtual
EOL
}

geomlist_one(){ cat << EOL | grep -v ^#
nnvtPMTSolid
nnvtBodySolid
nnvtInner1Solid
nnvtInner2Solid
EOL
}






geomlist0(){ cat << EOL
nmskSolidMask
nmskSolidMaskTail
nmskSolidMaskVirtual
nnvtPMTSolid
nnvtBodySolid
nnvtInner1Solid
nnvtInner2Solid
EOL
}
geomlist0_LABEL="nmsk_nnvt_solids"


geomlist(){ cat << EOL | grep -v ^#
hmskSolidMaskVirtual
hmskSolidMask
hmskSolidMaskTail
hamaPMTSolid
hamaBodySolid
hamaInner1Solid
hamaInner2Solid
# PMTSolid is 0.001 mm larger than BodySolid : effectively degenerates  
# Inner1 and Inner2 have coindent plane across middle of PMT, they are 5mm smaller than inside BodySolid
EOL
}
geomlist_LABEL="hmsk_hama_solids"




geomlist_names()
{
    local gg
    read -d "" -a gg <<< $(geomlist)
    local len=${#gg[@]}
    for (( i=0; i<$len; i++ ));do 
       local s=${ss[i]}
       local g=${gg[i]}
       local h=${g}__${geomlist_OPT}
       echo $h
    done
}

geomlist_export()
{
    local gg
    read -d "" -a gg <<< $(geomlist)
    local len=${#gg[@]}
    local symbols=STUVWXYZW
    local ss
    declare -a ss
    local i
    for (( i=0; i<$len; i++ ));do ss[i]=${symbols:i:1} ; done

    for (( i=0; i<$len; i++ ));do 
       local s=${ss[i]}
       local g=${gg[i]}
       local h=${g}__${geomlist_OPT}
     
       local fk="${s}_FOLD"
       local fv=$(printf ${geomlist_FOLD} $h)
       local hk="${s}_LABEL"  # formerly _GEOM

       local chk=""
       if [ -f "$fv/sframe.npy" -a -f "$fv/simtrace.npy" ]; then 
           chk="OK"
       else
           chk="MISSING"
       fi 

       printf "i %2d s %1s g %20s fk %10s fv %70s hk %10s h %30s chk %s \n" $i $s $g $fk $fv $hk $h $chk


       export $fk=$fv
       export $hk=$h 
    done 
    export SYMBOLS=${symbols:0:$len}
}


geomlist_dump()
{
    vars="BASH_SOURCE geomlist_LABEL geomlist_FOLD geomlist_OPT SYMBOLS S_LABEL S_FOLD T_LABEL T_FOLD U_LABEL U_FOLD BASH_SOURCE"
    for var in $vars ; do printf "%15s %s \n" $var ${!var} ; done
}


for geomlist_arg in $* 
do
    case ${geomlist_arg} in
         names) geomlist_names ;;
        export) geomlist_export ;;
          dump) geomlist_dump ;;
   esac 
done


