#!/bin/bash -l

arg=${1:-box}
echo ====== $0 $* ====== PWD $PWD ========= arg $arg ========

rc=42

echo ====== $0 $* ====== PWD $PWD ========= arg $arg ======== RC $rc =======

exit $rc
