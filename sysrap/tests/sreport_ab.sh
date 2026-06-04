#!/bin/bash
usage(){ cat << EOU
sreport_ab.sh : comparison between two report folders
========================================================

::

   ~/o/sreport_ab.sh

   A=N7 B=A7 ~/o/sreport_ab.sh

   A=N7 B=A7 PLOT=Substamp_ALL_Etime_vs_Photon ~/o/sreport_ab.sh

   A=N7 B=A7 PLOT=AB_Substamp_ALL_Etime_vs_Photon ~/o/sreport_ab.sh
       ## commandline that reproduces chep 2024 v0 fig 5
       ## (XORWOW with 100M state loading)

   A=N8 B=A8 PLOT=AB_Substamp_ALL_Etime_vs_Photon ~/o/sreport_ab.sh
       ## commandline showing Philox scan
       ## (Philox with inline curand_init, no loading)


Usage from laptop example
--------------------------

This performance plotting has traditionally been done on laptop, after rsync-ing the
small metadata only "_sreport" folders from REMOTE workstations which
are ssh tunnel connected to laptop, via eg::

   laptop> ## check the P and A tunnels are running
   laptop> o ; git pull   ## update opticks source, for the scripts only

   laptop> TMP=/data/blyth/opticks TEST=medium_scan REMOTE=P ~/o/cxs_min.sh info   ## check remote LOGDIR is  correct
   laptop> TMP=/data/blyth/opticks TEST=medium_scan REMOTE=P ~/o/cxs_min.sh grep   ## rsync the report to laptop

   laptop> TMP=/data1/blyth/tmp TEST=medium_scan REMOTE=A ~/o/cxs_min.sh info   ## check remote LOGDIR is  correct
   laptop> TMP=/data1/blyth/tmp TEST=medium_scan REMOTE=A ~/o/cxs_min.sh grep   ## rsync the report to laptop


For reference the "grep" ~/o/cxs_min.sh grep subcommand::

    600 if [ "${arg/grep}" != "$arg" ]; then
    601     source $OPTICKS_HOME/bin/rsync.sh ${LOGDIR}_sreport
    602 fi

NB the LOGDIR (aka the run dir) is one level above the event folders
and the sreport folder is a sibling to that containing only metadata
of small size.



Grab the historical _sreport folders to new macOS laptop for this
-------------------------------------------------------------------

1. ~/home/osx/etc_synthetic/etc_synthetic.sh workaround to create top level /data and /data1 dirs on Sequoia which
   are symbolic links to /usr/local/data /usr/local/data1 - to mimic the Linux layout

   HMM the cxs_min.sh folder layouts have changed over time many times : so better to use the
   fixed historical resolve paths from this script as the basis for grabbing.
   Grab the historical _sreport folders from PD and AD to the new laptop with::


        G=N8 ~/o/sreport_ab.sh grep   ## from PD
        G=N7 ~/o/sreport_ab.sh grep   ## from PD

        G=A8 ~/o/sreport_ab.sh grep   ## from AD
        G=A7 ~/o/sreport_ab.sh grep   ## from AD

        G=S7 ~/o/sreport_ab.sh grep   ## FAILED NO SUCH DIRECTORY


2. then plot, getting very close to presented figs::

        A=N8 B=A8 ~/o/sreport_ab.sh ana
        A=N7 B=A7 ~/o/sreport_ab.sh ana



EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
RDIR=$(dirname $(dirname $PWD))


defarg="info_ana"
arg=${1:-$defarg}

name=sreport_ab
script=$name.py

source $HOME/.opticks/GEOM/GEOM.sh

a=N8
b=A8
g=N8

export A=${A:-$a}
export B=${B:-$b}
export G=${G:-$g}


anno(){
   case $1 in
      N7) echo  ;;
      A7) echo  ;;
      S7) echo FAILED - NO SUCH DIRECTORY  ;;
      N8) echo  ;;
      A8) echo  ;;
   esac
}


remote_node(){
   : determine ssh node tag depending on the path
   : /data from PD - RTX TITAN
   : /data1 from AD - RTX Ada 5000
   : /hpcfs/juno/junogpu from LD - RTX PRO 6000

   local fold=$1
   local rem=?

   if [[ "$fold" =~ ^/data1 ]]; then
        rem=AD
   elif [[ "$fold" =~ ^/data ]]; then
        rem=PD
   elif [[ "$fold" =~ ^/hpcfs/juno/junogpu ]]; then
        rem=LD
   fi
   echo "$rem"
}


resolve(){

    case $1 in
      N7) echo /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;;
      A7) echo    /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1    ;;

      S7) echo /data/simon/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;;

      N8) echo /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;;
      A8) echo    /data1/blyth/tmp/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;;
    esac
}

plot=AB_Substamp_ALL_Etime_vs_Photon
export PLOT=${PLOT:-$plot}

export A_SREPORT_FOLD=$(resolve $A)_sreport
export B_SREPORT_FOLD=$(resolve $B)_sreport

export G_SREPORT_FOLD=$(resolve $G)_sreport
export G_SREPORT_FOLD_NODE=$(remote_node $G_SREPORT_FOLD)


export MODE=2                                 ## 2:matplotlib plotting

vars="0 BASH_SOURCE PWD RDIR arg defarg A B G A_SREPORT_FOLD B_SREPORT_FOLD G_SREPORT_FOLD G_SREPORT_FOLD_NODE MODE PLOT script"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/ana}" != "$arg" ]; then
    export COMMANDLINE="A=$A B=$B PLOT=$PLOT ~/o/sreport_ab.sh"
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi

if [ "${arg/grep}" != "$arg" ]; then
    read -p "$BASH_SOURCE : rsync G[$G] [$G_SREPORT_FOLD] from [$G_SREPORT_FOLD_NODE] : enter YES to proceed : " ans

    if [ "$ans" == "YES" ]; then
        REMOTE=$G_SREPORT_FOLD_NODE source $RDIR/bin/rsync.sh $G_SREPORT_FOLD
    fi
fi






if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    ## HMM : IS THIS STILL USED ?
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

