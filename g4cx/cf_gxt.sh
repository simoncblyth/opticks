#!/bin/bash -l 
usage(){ cat <<EOU
cf_gxt.sh : compare simtrace from three geometries 
=======================================================

::

   U_OFFSET=0,0,200 ./cf_gxt.sh 

   ./cf_gxt.sh mpcap
   ./cf_gxt.sh mppub

   OPT=U0 FOCUS=257,-39,7 ./cf_gxt.sh  
   OPT=U1 FOCUS=257,-39,7 ./cf_gxt.sh  

   OPT=U1 T_OFFSET=0,0,5 FOCUS=257,-39,7 ./cf_gxt.sh 
   OPT=U1 T_OFFSET=0,0,5 FOCUS=257,-39,7 ./cf_gxt.sh 

EOU
}

defarg="ana"
arg=${1:-$defarg}

#opt=U0
opt=U1
OPT=${OPT:-$opt}

export S_GEOM=nmskSolidMask__$OPT
export T_GEOM=nmskSolidMaskTail__$OPT
#export U_GEOM=nmskTailInner__$OPT
#export V_GEOM=nmskTailOuter__$OPT

focus=257,-39,7
export FOCUS=${FOCUS:-$focus}

geom=""
vars="S_GEOM T_GEOM U_GEOM V_GEOM"
for var in $vars ; do 
   if [ -n "${!var}" ]; then
      if [ -z "$geom" ]; then
         geom="${!var}" 
      else
         geom="$geom,${!var}" 
      fi
   fi 
done
MGEOM=$geom

export S_FOLD=/tmp/$USER/opticks/GeoChain/$S_GEOM/G4CXSimtraceTest/ALL
export T_FOLD=/tmp/$USER/opticks/GeoChain/$T_GEOM/G4CXSimtraceTest/ALL
export U_FOLD=/tmp/$USER/opticks/GeoChain/$U_GEOM/G4CXSimtraceTest/ALL
export V_FOLD=/tmp/$USER/opticks/GeoChain/$V_GEOM/G4CXSimtraceTest/ALL

# collective folder
export MFOLD=/tmp/$USER/opticks/GeoChain/$MGEOM/G4CXSimtraceTest/ALL

if [ "info" == "$arg" ]; then
    vars="BASH_SOURCE arg defarg S_GEOM T_GEOM U_GEOM V_GEOM S_FOLD T_FOLD U_FOLD V_FOLD MGEOM MFOLD OPT"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi 

if [ "ana" == "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/cf_G4CXSimtraceTest.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 1 
fi 

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$MFOLD/figs
    export CAP_REL=cf_gxt
    export CAP_STEM=${MGEOM}
    case $arg in
       mpcap) source mpcap.sh cap  ;;
       mppub) source mpcap.sh env  ;;
    esac

    if [ "$arg" == "mppub" ]; then
        source epub.sh
    fi
fi
exit 0 
