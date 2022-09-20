#!/bin/bash 

usage(){ cat << EOU
geomlist.sh
============================

To test::

    source ../bin/geomlist_test.sh 


Sourcing this script defines and runs geomlist  bash function

Inputs:

geomlist:bash function
    providing list of geom names separated by newlines
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


geomlist(){ cat << EOL
nmskSolidMask
nmskSolidMaskTail
nmskSolidMaskVirtual
nnvtPMTSolid
nnvtBodySolid
nnvtInner1Solid
nnvtInner2Solid
EOL
}
geomlist_LABEL="nmsk_nnvt_solids"

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




geomlist_export


vars="BASH_SOURCE geomlist_LABEL geomlist_FOLD geomlist_OPT SYMBOLS S_LABEL S_FOLD T_LABEL T_FOLD U_LABEL U_FOLD BASH_SOURCE"
for var in $vars ; do printf "%15s %s \n" $var ${!var} ; done



