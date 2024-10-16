#!/bin/bash 
usage(){ cat << EOU
sreport_ab.sh : comparison between two report folders 
========================================================

::

   A_JOB=N7 B_JOB=A7 ~/opticks/sysrap/tests/sreport_ab.sh

   A_JOB=N7 B_JOB=A7 PLOT=Substamp_ALL_Etime_vs_Photon ~/o/sreport_ab.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

defarg="info_ana"
arg=${1:-$defarg}

name=sreport_ab
script=$name.py

source $HOME/.opticks/GEOM/GEOM.sh 

a=N7   
b=A7   

export A=${A:-$a}
export B=${B:-$b}

resolve(){
    case $1 in 
      N7) echo /data/blyth/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL1 ;; 
      A7) echo /data1/blyth/tmp/GEOM/$GEOM/CSGOptiXSMTest/ALL1    ;;
      S7) echo /data/simon/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL1 ;; 
    esac
}


export PLOT="AB_Substamp_ALL_Etime_vs_Photon"

export A_SREPORT_FOLD=$(resolve $A)_sreport  
export B_SREPORT_FOLD=$(resolve $B)_sreport  
export MODE=2                                 ## 2:matplotlib plotting 

vars="0 BASH_SOURCE arg defarg A B A_SREPORT_FOLD B_SREPORT_FOLD MODE PLOT"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    export COMMANDLINE="A=$A B=$B PLOT=$PLOT ~/o/sreport_ab.sh"
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$A_SREPORT_FOLD/figs
    export CAP_REL=sreport_ab
    export CAP_STEM=$PLOT
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
    esac
    if [ "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 

exit 0 

