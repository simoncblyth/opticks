#!/bin/bash -l 

cd $(dirname $BASH_SOURCE)

source $HOME/.opticks/GEOM/GEOM.sh 

name=OpticksPhotonSTANDALONETest 

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name


gcc $name.cc -I.. -std=c++11 -lstdc++ -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0 

