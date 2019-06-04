#!/bin/bash -l

arg=${1:-box}
shift

cd /tmp

echo ====== $0 $arg $* ====== PWD $PWD =================

tboolean-
cmd="tboolean-$arg --okg4 --compute $*"

echo $cmd
eval $cmd

echo ====== $0 $arg $* ====== PWD $PWD ============ RC $rc =======

exit $rc
