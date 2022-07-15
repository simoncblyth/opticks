#!/bin/bash -l 
usage(){ cat << EOU
bin/OPTICKS_INPUT_PHOTON.sh 
==============================

Usage example::

   source ../bin/OPTICKS_INPUT_PHOTON.sh
   #source ../bin/OPTICKS_INPUT_PHOTON_.sh   # alternative without export

The internally sourced script OPTICKS_INPUT_PHOTON_.sh sets the variables and this exports them.
This separation allows bash control of access to the variables from python OR C++ level. 
For example this could be used to only export when a file path exists. 
The envvars when defined are used as SEventConfig defaults. 

OPTICKS_INPUT_PHOTON 
   name or path of input photon .npy file, default of SEventConfig::InputPhoton 

OPTICKS_INPUT_PHOTON_FRAME
   moi_or_iidx string eg "Hama:0:1000" OR "35000", default of SEventConfig::InputPhotonFrame

Full list of scripts relevant to OPTICKS_INPUT_PHOTON::

    epsilon:opticks blyth$ find . -name '*.sh' -exec grep -l OPTICKS_INPUT_PHOTON {} \;
    ./ana/input_photons_plt.sh
    ./CSGOptiX/cxs_raindrop.sh
    ./CSG/tests/CSGFoundry_getFrame_Test.sh
    ./bin/GEOM_.sh
    ./bin/OPTICKS_INPUT_PHOTON.sh
    ./sysrap/tests/SEvtTest.sh
    ./u4/u4s.sh
    ./g4cx/gxs.sh

EOU
}

ScriptDir=$(dirname $BASH_SOURCE)
source $ScriptDir/OPTICKS_INPUT_PHOTON_.sh    

[ -n "$OPTICKS_INPUT_PHOTON" ]       && export OPTICKS_INPUT_PHOTON
[ -n "$OPTICKS_INPUT_PHOTON_FRAME" ] && export OPTICKS_INPUT_PHOTON_FRAME

if [ -z "$QUIET" ]; then 
    vars="BASH_SOURCE ScriptDir OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME OPTICKS_INPUT_PHOTON_ABSPATH"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
    #for var in $vars ; do printf "== %s \n" "$(declare -p $var)" ; done   # -x in output shows exported
    echo 
fi


