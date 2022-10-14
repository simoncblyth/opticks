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

Geomlist functions include::

   geomlist_N_full
   geomlist_N_short
   geomlist_N
   geomlist_H



EOU
}


geomlist_N_full(){ cat << EOL | grep -v ^#
nmskSolidMask
nmskSolidMaskTail
nmskTailOuter
nmskTailInner

nmskSolidMask
nmskMaskOut
nmskTopOut
nmskBottomOut
nmskMaskIn
nmskTopIn
nmskBottomIn

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

geomlist_N_short(){ cat << EOL | grep -v ^#
nmskSolidMaskVirtual
nmskTailInnerITube
nmskTailOuterITube
EOL
}

geomlist_N(){ cat << EOL | grep -v ^#
nmskSolidMask
nmskSolidMaskTail
nmskSolidMaskVirtual
nnvtPMTSolid
nnvtBodySolid
nnvtInner1Solid
nnvtInner2Solid
EOL
}

geomlist_H(){ cat << EOL | grep -v ^#
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

geomlist_tag=H
#geomlist_tag=N


if [ "${geomlist_tag}" == "N" ]; then
   geomlist(){  geomlist_N ; }
   geomlist_LABEL="nmsk_nnvt_solids"
elif [  "${geomlist_tag}" == "H" ]; then
   geomlist(){  geomlist_H ; }
   geomlist_LABEL="hmsk_hama_solids"
fi 









geomlist_names()
{
    : bin/geomlist.sh 
    : reads geomlist into array variable then lists each name with opt appended
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
    : bin/geomlist.sh 
    : reads geomlist into array variable and exports into multiple envvars 
    : keyed by the SYMBOLS envvar 
    : the layout of envvars is understood by ana/fold.py  
    :
    : note current limitation of 9 GEOM could easily be extended
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
       local cfb=$(dirname $(dirname $fv))

       local chk="?"
       if [ ! -f "$cfb/CSGFoundry/solid.npy" ]; then
           chk="NO-CSGFoundry"
       elif [ -f "$fv/sframe.npy" -a -f "$fv/simtrace.npy" ]; then 
           chk="OK"
       else
           chk="NO-Intersect"
       fi 

       if [ -n "$VERBOSE" ]; then 
       printf "i %2d s %1s g %20s fk %10s fv %70s hk %10s h %30s chk %s \n" $i $s $g $fk $fv $hk $h $chk
       else
       printf "i %2d s %1s fv %70s chk %s \n" $i $s $fv "$chk"
       fi 


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


