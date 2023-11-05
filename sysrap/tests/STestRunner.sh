#!/usr/bin/env bash
usage(){ cat << EOU
STestRunner.sh
================

Following:

* https://enccs.github.io/cmake-workshop/
* https://enccs.github.io/cmake-workshop/hello-ctest/
* https://github.com/ENCCS/cmake-workshop/blob/main/content/code/day-1/06_bash-ctest/solution/CMakeLists.txt

Dev::

   om 
   om-cd
   ctest -N             # list tests

   ctest -R SEnvTest_PASS  --output-on-failure
   ctest -R SEnvTest_FAIL  --output-on-failure

EOU
}

EXECUTABLE="$1"
shift
ARGS="$@"

source $HOME/.opticks/GEOM/GEOM.sh 

vars="HOME PWD GEOM BASH_SOURCE EXECUTABLE ARGS"

for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 

env 

$EXECUTABLE $@

[ $? -ne 0 ] && echo $BASH_SOURCE : FAIL from $EXECUTABLE && exit 1 

exit 0

