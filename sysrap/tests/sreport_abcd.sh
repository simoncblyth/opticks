#!/bin/bash 
usage(){ cat << EOU
sreport_abcd.sh : comparison between upto four report folders A,B,C,D 
=======================================================================

::

   A=N7 B=A7 A_SLI=3:12 B_SLI=3:12 C=N8 D=A8 C_SLI=2:12 D_SLI=2:12 PLOT=ABCD_Substamp_ALL_Etime_vs_Photon ~/o/sreport_abcd.sh


Start with sreport_ab.sh for grabbing and two fold comparisons 
before using this for four way comparison. 


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

defarg="info_ana"
arg=${1:-$defarg}

name=sreport_abcd
script=$name.py

source $HOME/.opticks/GEOM/GEOM.sh 

a=N7   
b=A7   
c=N8   
d=A8   

export A=${A:-$a}
export B=${B:-$b}
export C=${C:-$c}
export D=${D:-$d}

resolve(){
    case $1 in 
      N7) echo /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;; 
      A7) echo /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1    ;;
      S7) echo /data/simon/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;; 

      N8) echo /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;; 
      A8) echo /data1/blyth/tmp/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;; 
    esac
}

plot=ABCD_Substamp_ALL_Etime_vs_Photon
export PLOT=${PLOT:-$plot}

export A_SREPORT_FOLD=$(resolve $A)_sreport  
export B_SREPORT_FOLD=$(resolve $B)_sreport  
export C_SREPORT_FOLD=$(resolve $C)_sreport  
export D_SREPORT_FOLD=$(resolve $D)_sreport  
export MODE=2                                 ## 2:matplotlib plotting 

vars="0 BASH_SOURCE arg defarg A B C D A_SREPORT_FOLD B_SREPORT_FOLD C_SREPORT_FOLD D_SREPORT_FOLD MODE PLOT script"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    export COMMANDLINE="A=$A B=$B C=$C D=$D PLOT=$PLOT ~/o/sreport_abcd.sh"
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$A_SREPORT_FOLD/figs
    export CAP_REL=sreport_abcd
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

