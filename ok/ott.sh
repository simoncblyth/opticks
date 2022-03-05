#!/bin/bash -l 
usage(){ cat << EOU
ott.sh
========

To handle renders of large geometries with ancient 
laptop GPU disable the instanced geometry with::

   --enabledmergedmesh 0, 

This is done for arg of "old" or "new" but not for "last" 
as that is usually small test geometry. 

EOU
}


msg="=== $BASH_SOURCE :"
arg=${1:-last}
echo $msg arg $arg
source $OPTICKS_HOME/bin/geocache_hookup.sh $arg

opt=""
case $arg in 
  old) opt="--enabledmergedmesh 0," ;;
  new) opt="--enabledmergedmesh 0," ;;
esac

if [ -n "$opt" ]; then
   echo $msg arg $arg opt $opt  
fi 


export Frame=INFO
export GParts=INFO

which OTracerTest 

if [ -n "$DEBUG" ]; then
    lldb__ OTracerTest  $opt
else
    OTracerTest $opt
fi 

