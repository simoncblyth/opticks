#!/bin/bash -l

arg=${1:-box}

cd /tmp

echo ====== $0 $* ====== PWD $PWD ========= arg $arg ========

tboolean-
cmd="tboolean-$arg --okg4 --load --args"
echo $cmd
eval $cmd
rc=$?

echo ====== $0 $* ====== PWD $PWD ========= arg $arg ======== RC $rc =======

exit $rc
