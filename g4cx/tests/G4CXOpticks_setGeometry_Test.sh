#!/bin/bash
usage(){ cat << EOU
G4CXOpticks_setGeometry_Test.sh
===================================

::

    ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh
    LOG=1 ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh


Test of geometry conversion in isolation::
                  
   Geant4 --[U4]--> SSim/stree --[CSGImport]-> CSGFoundry 


HOW TO LOAD YOUR GDML FILE WITH THIS SCRIPT
---------------------------------------------

There are several ways to configure GDML loading currently, see "G4CXOpticks::setGeometry()".
However this script uses just one of them. The relevant lines 
for this config from this bash script are::

    source $HOME/.opticks/GEOM/GEOM.sh             # mini config script that only sets GEOM envvar 
    origin=$HOME/.opticks/GEOM/$GEOM/origin.gdml   # path to GDML
    export ${GEOM}_GDMLPathFromGEOM=$origin        # export path to GDML 

So to load your own GDML using this script, without changing this script:

1. decide on identifier string for your geometry, eg Z36 
2. create or edit the file $HOME/.opticks/GEOM/GEOM.sh for example containing
   (NB this file is not in the repository, it is in ~/.opticks)::

     #!/bin/bash 
     # THIS SCRIPT DOES ONE THING ONLY : IT EXPORTS GEOM
     geom=Z39
     export GEOM=$geom  

   * the "GEOM" bash function defaults to editing this GEOM.sh as 
     a shortcut for quickly changing between geometries

3. copy your gdml file to the path that this script expects::

     $HOME/.opticks/GEOM/Z36/origin.gdml 

4. now you can run the script with::

     ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh

5. examine translated geometry that is written to folders beneath::

     $HOME/.opticks/GEOM/Z36/CSGFoundry/

6. the Opticks CSGFoundry geometry is comprised of NumPy .npy and .txt files 
   which can all be examined from python, use the below command to 
   load the geometry into python::

     ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh ana


For example, the JUNO geometry with Opticks GEOM identifier "V1J011" 
is comprised of ~200 .npy and ~150 .txt files::

    epsilon:~ blyth$ find ~/.opticks/GEOM/V1J011/CSGFoundry -type f -name '*.npy' | wc -l 
         199
    epsilon:~ blyth$ find ~/.opticks/GEOM/V1J011/CSGFoundry -type f -name '*.txt' | wc -l 
         146
    epsilon:~ blyth$ find ~/.opticks/GEOM/V1J011/CSGFoundry -type f  | wc -l 
         345


Note that following the conventions of this script enables switching 
between geometries without changing this script, simply by setting the 
GEOM envvar that is exported from $HOME/.opticks/GEOM/GEOM.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh 

defarg=info_run_ana
arg=${1:-$defarg}

bin=G4CXOpticks_setGeometry_Test
script=$bin.py 

source $HOME/.opticks/GEOM/GEOM.sh   # mini config script that only sets GEOM envvar 
[ -z "$GEOM" ] && echo $BASH_SOURCE : FATAL GEOM $GEOM MUST BE SET && exit 1 


case $GEOM in 
   FewPMT) geomscript=../../u4/tests/FewPMT.sh ;;
esac

if [ -n "$geomscript" -a -f "$geomscript" ]; then 
    echo $BASH_SOURCE : GEOM $GEOM : sourcing geomscript $geomscript
    source $geomscript
else
    echo $BASH_SOURCE : GEOM $GEOM : no geomscript    
fi 




vars="BASH_SOURCE arg SDIR GEOM savedir FOLD bin geomscript script origin"


export GProperty_SIGINT=1
#export NTreeBalance__UnableToBalance_SIGINT=1
#export BFile__preparePath_SIGINT=1
#export GGeo__save_SIGINT=1

savedir=~/.opticks/GEOM/$GEOM

export FOLD=$savedir
export G4CXOpticks__setGeometry_saveGeometry=$savedir
export G4CXOpticks__saveGeometry_saveGGeo=1

#export NNodeNudger__DISABLE=1
#export X4Solid__convertPolycone_nudge_mode=0 # 0:DISABLE 
#export U4Polycone__DISABLE_NUDGE=1 
#export s_csg_level=2 
#export sn__level=2
#export U4Tree__DISABLE_OSUR_IMPLICIT=1
#export X4PhysicalVolume__ENABLE_OSUR_IMPLICIT=1


logging(){ 
   type $FUNCNAME
   export Dummy=INFO
   export G4CXOpticks=INFO
   export U4Tree=INFO
   #export CSGFoundry=INFO
   #export U4VolumeMaker=INFO
}
[ -n "$LOG" ] && logging 
env | grep =INFO


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    ## run does geometry setup like for dbg
    ./GXTestRunner.sh $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 

    origin=$HOME/.opticks/GEOM/$GEOM/origin.gdml    # path to GDML
    if [ -f "$origin" ]; then
        export ${GEOM}_GDMLPathFromGEOM=$origin      # export path to GDML 
    fi 

    gdb__  $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi 

if [ "${arg/pdb}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 4
fi 

exit 0

