#!/bin/bash -l 
usage(){ cat << EOU
cxs_solidXJfixture.sh
=======================

NB unlike many others this does not honour GEOM, **it sets GEOM** based on COMP 

Whilst iterating on cxs running on workstation and 
grabbing the intersect photons to local laptop with
the grab script is is convenient to use "lrun" mode.  

    ./cxs_grab.sh 
    ./cxs_solidXJfixture.sh lrun     

The grab writes CSGOptiXSimtraceTest_OUTPUT_DIR.sh script
to the LOGDIR directory which is then sourced by the cxs.sh
script when run in lrun mode.   

For custom geometry such as from GeoChain test geometry need to use::

    ./tmp_grab.sh png
    ./tmp_grab.sh png
    ./cxs_solidXJfixture.sh lrun     



Whilst working on the plot presentation of previously 
grabbed intersect photons the COMP control is used.
This is used for changing annotations and deciding 
on which to publish::

    COMP=PR_55 ./cxs_solidXJfixture.sh

Once decided to publish a figure png use the PUB control 
together with the usual pub.py arguments eg sp/s5/cp/dp/xdp::

    PUB=1 COMP=PR_55 ./cxs_solidXJfixture.sh sp
    PUB=1 COMP=PR_55 ./cxs_solidXJfixture.sh cp 
    PUB=1 COMP=PR_55 ./cxs_solidXJfixture.sh cp | sh 
    PUB=1 COMP=PR_55 ./cxs_solidXJfixture.sh cp | grep mpplt 
    PUB=1 COMP=PR_55 ./cxs_solidXJfixture.sh cp | grep mpplt | sh 


    COMP=XZ_10 ./cxs_solidXJfixture.sh 
    PUB=1 COMP=XZ_10 ./cxs_solidXJfixture.sh cp | sh 


    COMP=PR_0 ISEL=1 ./cxs_solidXJfixture.sh 

EOU

}


msg="=== $BASH_SOURCE :"

#comp=XZ_10         # done
#comp=YZ_10         # rerun

#comp=TP_10         # rerun
#comp=RT_10         # rerun

#comp=PR_55         # done 
#comp=TR_55         # done 

#comp=PR_0          # done 
#comp=TR_0          # done
#comp=TP_0          # done  
#comp=TP_0_Rshift   # done

comp=custom_XY
#comp=TR_2
#comp=TR_52
#comp=TR_41
#comp=PR_41
#comp=ClosePR_41
#comp=CloserPR_41



GBASE=XJfixtureConstruction
export COMP=${COMP:-$comp} 

case $COMP in 
  custom_XY)  geom=custom_${GBASE}XY ;;  
          *)  geom=${GBASE}${COMP}   ;; 
esac

export GEOM=$geom


isel=
cfbase=
ce_offset=0
ce_scale=0
gsplot=1

if [ "$GEOM" == "${GBASE}XZ_10" ]; then

    note="side cross section of sTarget sAcrylic sXJfixture sXJanchor  "
    moi="solidXJfixture:10"
    cegs=16:0:9:100           
    gridscale=0.20

    ce_offset=1    # pre-tangential-frame approach  
    ce_scale=1 

elif [ "$GEOM" == "${GBASE}YZ_10" ]; then

    note="view is difficult to interpret : could be a bug or just a slice at a funny angle, need tangential check"
    moi="solidXJfixture:10"
    cegs=0:16:9:100            
    gridscale=0.20

    ce_offset=1    # pre-tangential-frame approach  
    ce_scale=1 

elif [ "$GEOM" == "${GBASE}TP_10" ]; then

    note="nicely aligned rectangle in YZ=(TP), longer in T direction +- 1 extent unit, +-0.24 extent units in P  "
    moi="solidXJfixture:10:-3"
    #    R:T:P 
    cegs=0:16:9:100            
    gridscale=0.20

elif [ "$GEOM" == "${GBASE}RT_10" ]; then

    note="bang on tangential view from P(phi-tangent-direction) showing the radial coincidence issues clearly" 
    moi="solidXJfixture:10:-3"
    cegs=16:9:0:100            
    gridscale=0.20


elif [ "$GEOM" == "${GBASE}TR_2" ]; then

    note="depth check : looks like positioned to correspond to sTarget radius" 
    moi="solidXJfixture:2:-3"
    cegs=9:16:0:100            
    gridscale=0.80

elif [ "$GEOM" == "${GBASE}TR_52" ]; then

    note="picked as render suggests directly under foot : bumps into rods" 
    moi="solidXJfixture:52:-3"
    cegs=9:16:0:100            
    gridscale=0.80

elif [ "$GEOM" == "${GBASE}TR_41" ]; then

    note="picked as render looks on edge of foot : cross section shows poking out" 
    moi="solidXJfixture:41:-3"
    cegs=9:16:0:100            
    gridscale=0.80

elif [ "$GEOM" == "${GBASE}PR_41" ]; then

    note="very clear impinge" 
    moi="solidXJfixture:41:-3"
    cegs=9:0:16:100      
    gridscale=0.80

elif [ "$GEOM" == "${GBASE}ClosePR_41" ]; then

    note="very clear impinge" 
    moi="solidXJfixture:41:-3"
    cegs=9:0:16:100      
    gridscale=0.20

elif [ "$GEOM" == "${GBASE}CloserPR_41" ]; then

    note="very clear impinge" 
    moi="solidXJfixture:41:-3"
    cegs=9:0:16:100      
    gridscale=0.10

elif [ "$GEOM" == "${GBASE}PR_55" ]; then

    note="sXJfixture and sXJanchor are inside uni_acrylic1 and bump into uni1 "
    moi="solidXJfixture:55:-3"
    #    R:T:P        larger side of grid becomes horizontal : hence  PR  (not RP)
    cegs=9:0:16:100          
    gridscale=0.80

elif [ "$GEOM" == "${GBASE}TR_55" ]; then

    note="clearly solidXJfixture and solidXJanchor are inside the foot of the stick uni_acrylic1"
    note1="but as slice is to the side as can be seen from PR_55 do not see much of the stick "
    moi="solidXJfixture:55:-3"
    #    R:T:P        larger side of grid becomes horizontal : hence  TR  (not RT)
    cegs=9:16:0:100            
    gridscale=0.80

elif [ "$GEOM" == "${GBASE}PR_0" ]; then

    note="clear overlap : 3 rectangle : spurious intersects"
    moi="solidXJfixture:0:-3"
    #    R:T:P        larger side of grid becomes horizontal : hence  PR  (not RP)
    cegs=9:0:16:100          
    gridscale=0.10

elif [ "$GEOM" == "${GBASE}TR_0" ]; then

    note="mid-chimney with ce 0,0,17696.94  : solidSJReceiver and solidXJfixture clearly overlapped "
    moi="solidXJfixture:0:-3"
    #    R:T:P        larger side of grid becomes horizontal : hence  TR  (not RT)
    cegs=9:16:0:100            
    gridscale=0.10

elif [ "$GEOM" == "${GBASE}TP_0" ]; then

    note="crossed rectangles"
    moi="solidXJfixture:0:-3"
    #    R:T:P        larger side of grid becomes horizontal : hence  TP
    cegs=0:16:9:100            
    gridscale=0.10

elif [ "$GEOM" == "${GBASE}TP_0_Rshift" ]; then

    note="crossed celtic crosses"
    moi="solidXJfixture:0:-3"
    #    R:T:P        larger side of grid becomes horizontal : hence  TP
    cegs=0:16:9:-2:0:0:100            
    gridscale=0.10

elif [ "$GEOM" == "custom_${GBASE}XY" ]; then

    ## HMM : THIS IS KINDA MISPLACED BETTER IN A cxs_geochain.sh  
    note="all interior lines are spurious from coincidences"
    cfbase=$TMP/GeoChain/XJfixtureConstruction  
    moi=0
    #cegs=0:16:9:-2:0:0:100    # YZ with offset in X           
    cegs=0:16:9:0:0:0:100            
    gridscale=0.10

    ce_offset=1    # pre-tangential-frame approach  
    ce_scale=1 

    export ZZ=-33.5,6.5 
    #export YY=-65,-50,-35,0,35,50,65 
    #export YY=-45,-25,25,45 


else
    echo $msg ERROR GEOM $GEOM is not handled  
    echo $msg comp $comp COMP $COMP
    echo $msg geom $geom GEOM $GEOM
    exit 1 
fi


export TOPLINE="COMP=$COMP ./cxs_solidXJfixture.sh  "

if [ -n "$PUB" ]; then 
   source ./cxs_pub.sh $* 
else
   echo $msg COMP $COMP GEOM $GEOM note $note PUB $PUB
   source ./cxs.sh $*
fi


if [ -z "$cfbase" ]; then
    echo $msg cfbase $cfbase is not defined : so are handling standard geometry : hence use cxs_grab.sh to grab from workstation to laptop
else
    echo $msg cfbase $cfbase is defined : so are handling non-standard geometry : hence use tmp_grab.sh to grab from workstation to laptop
fi 






