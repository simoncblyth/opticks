#!/bin/bash -l

cd /tmp

echo ====== $0 $* ====== PWD $PWD =================

geocache-
geocache-key-export

cmd="geocache-360 --fullscreen --args"
echo $cmd
eval $cmd
rc=$?

echo ====== $0 $* ====== PWD $PWD ================= RC $rc =======

exit $rc

