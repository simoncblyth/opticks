#!/bin/bash -l 
usage(){ cat << EOU
U4SimtraceTest.sh
==========================

FewPMT.sh/tub3LogicalPMT::

    MODE=2 FOCUS=0,0,80 ~/opticks/u4/tests/U4SimtraceTest.sh ana 

    MODE=2 ~/opticks/u4/tests/U4SimtraceTest.sh

TODO
------

* add metadata checks that plotted U4SimulateTest photons are from a geometry matching the U4SimtraceTest geometry
 

Commands
-----------

run/dbg
    simtrace intersects against geometry 
ana
    presentation of simtrace intersects using python matplotlib OR pyvista
mpcap/pvcap
    screenshot current matplotlib/pyvista window with chrome cropped 
mppub/pvpub
    publication by copying matplotlib/pyvista screenshot png into presentation tree


Workflow to add plots to presentations
-----------------------------------------

1. check matplotlib window plot and annotations are presentable::

     APID=2563 AOPT=idx N=0 SUBTITLE="7:SD at vac/vac" ./U4SimtraceTest.sh ana

2. whilst some part of the matplotlib window is still visible, in a separate tab run::

       u4t ; ./U4SimtraceTest.sh mpcap   
     
   * If capture environment depends on envvars make sure that is consistent with above command
   * NB have to click on the matplotlib window (so it must be visible before running mpcap), that 
     turns it blue targetting the window screen capture

3. publish that, coping the .png into presentation tree::

     u4t ; PUB=2563_Unphysical_SD_in_vacuum ./U4SimtraceTest.sh mppub
     ## ensure PUB is a distinct identifier

4. reference the .png by copy/paste the "s5p_line" from the output of the above step 
   into ~/env/presentation/s5_background_image.txt (use presentation-e) 
   and reference that by adding a presentation page with matching title 
 
5. it is good to include the primary commandline from step 1 in the presentation, even if 
   not presented, in order to allow reproducing the plot 



Suggested Workflow To Find Photons to Compare and plot
--------------------------------------------------------

Use two U4SimulateTest.sh sessions for N=0 and N=1::

    u4t
    N=0 POM=1 ./U4SimulateTest.sh    ## "a" in Simtrace session 
    N=1 POM=1 ./U4SimulateTest.sh    ## "b" in Simtrace session 

And a third U4SimtraceTest.sh session::

    u4t
    N=1 ./U4SimtraceTest.sh ana 

Check a and b in simtrace session::

    In [1]: a.f.base, b.f.base
    Out[1]: 
    ('/tmp/blyth/opticks/GEOM/FewPMT/U4SimulateTest/ALL0',
     '/tmp/blyth/opticks/GEOM/FewPMT/U4SimulateTest/ALL1')

Pick some APID, BPID expected to be similar by comparing 
histories as visible in the first two sessions (eg np.c_[np.arange(40),q[:40]] ).
Use AOFF or BOFF to offset to make similar paths visible, eg AOFF=0,0,10
Keep starting and stopping the third session as change APID, BPID (-ve disables):: 

    APID=17 BPID=7  N=1 ./U4SimtraceTest.sh ana

Examples
-----------

::

    N=0 ./U4SimtraceTest.sh 
    N=1 ./U4SimtraceTest.sh 

    APID=173 PLAB=1 BGC=yellow ./U4SimtraceTest.sh ana

Z-changing big bouncers::

    N=0 APID=256 PLAB=1 BGC=white ./U4SimtraceTest.sh ana
    N=1 APID=261 PLAB=1 BGC=white ./U4SimtraceTest.sh ana

Grab the custom boundary status for each point::

    In [25]: t.aux[261,:32,1,3].copy().view(np.int8)[::4].copy().view("|S32")
    Out[25]: array([b'TTTZNZRZNZA'], dtype='|S32')

::

    N=0 APID=726 BPID=-1 AOPT=nrm FOCUS=0,10,185 ./U4SimtraceTest.sh ana
    N=1 APID=-1 BPID=726 BOPT=nrm FOCUS=0,10,185 ./U4SimtraceTest.sh ana


two_pmt::

    FOCUS=0,0,255 ./U4SimtraceTest.sh ana

    N=0 APID=813 AOPT=idx BGC=yellow FOCUS=0,0,255 ./U4SimtraceTest.sh ana
    N=0 APID=813 AOPT=ast BGC=yellow FOCUS=0,0,255 ./U4SimtraceTest.sh ana

    N=1 BPID=748 BOPT=ast,nrm BGC=yellow FOCUS=0,0,255 ./U4SimtraceTest.sh ana
    N=1 BPID=150 BOPT=nrm,ast BGC=yellow FOCUS=0,0,255 ./U4SimtraceTest.sh ana


MODE 3 pyvista 3D plotting::

    MODE=3 EYE=0,-1000,0 LOOK=0,0,0 UP=0,0,1 ./U4SimtraceTest.sh ana

Questions
-----------

Q: Where is the N envvar to control natural geometry acted upon ?
A: j/PMTSim/IGeomManager.h:IGeomManager::declProp interprets envvar to set values::

    epsilon:tests blyth$ grep UseNaturalGeometry *.*
    FewPMT.sh:export hama_UseNaturalGeometry=$version 
    FewPMT.sh:export nnvt_UseNaturalGeometry=$version 


Q: Where is the SEventConfig::IsRGModeSimtrace flipped ?  

  
EOU
}

DIR=$(dirname $BASH_SOURCE)
bin=U4SimtraceTest
apid=-1
bpid=-1
geom=FewPMT
evt=000
eye=0,1000,0   # +Y 1000mm

#cegs=16:0:9:1000   # default used from SFrameGenstep::MakeCenterExtentGensteps
cegs=16:0:9:5000    # increase photon count for more precise detail 

export VERSION=${N:-0}
export GEOM=${GEOM:-$geom}
export GEOMFOLD=/tmp/$USER/opticks/GEOM/$GEOM
export BASE=$GEOMFOLD/$bin
export FOLD=$BASE/$VERSION   ## controls where the executable writes geometry
export EVT=${EVT:-$evt}
export AFOLD=$GEOMFOLD/U4SimulateTest/ALL0/$EVT
export BFOLD=$GEOMFOLD/U4SimulateTest/ALL1/$EVT   # SEL1 another possibility 
export APID=${APID:-$apid}                        # APID for photons from ALL0
export BPID=${BPID:-$bpid}                        # BPID for photons from ALL1
export EYE=${EYE:-$eye}       # not extent scaled, just mm
export CEGS=${CEGS:-$cegs}
export CEHIGH_0=-1:1:0:0:7:8:1000:4
 

geomscript=$DIR/$GEOM.sh 
if [ -f "$geomscript" ]; then  
    source $geomscript 
else
    echo $BASH_SOURCE : no geomscript $geomscript
fi 

# GEOMList for the GEOM is set for example in the FewPMT.sh geomscript 
_GEOMList=${GEOM}_GEOMList
GEOMList=${!_GEOMList}

echo $BASH_SOURCE GEOMList : $GEOMList 


loglevels()
{
    export U4VolumeMaker=INFO
}
loglevels


log=${bin}.log
logN=${bin}_$VERSION.log

defarg="run_ana"
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
    [ -f "$log" ] && echo $BASH_SOURCE rename log $log to logN $logN && mv $log $logN    
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ "${arg/ana}" != "$arg"  ]; then
    [ "$arg" == "nana" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $DIR/$bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

if [ "$arg" == "pvcap" -o "$arg" == "pvpub" -o "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=U4SimtraceTest
    export CAP_STEM=$GEOM
    case $arg in  
       pvcap) source pvcap.sh cap  ;;  
       mpcap) source mpcap.sh cap  ;;  
       pvpub) source pvcap.sh env  ;;  
       mppub) source mpcap.sh env  ;;  
    esac
    if [ "$arg" == "pvpub" -o "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 

exit 0 
