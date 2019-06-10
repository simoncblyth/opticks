#!/bin/bash -l

tboolean-sh-notes(){ cat << EON
/**
tboolean.sh
==============

Direct Approach
-----------------

* setup executable with the new o.sh using --okg4 option
  to select OKG4Test 
* test geometry is based off the direct geocache
  that is pointed to by OPTICKS_KEY envvar 
* material/surface names need to match those in the 
  base geometry 

Legacy Approach
-----------------

* executables and environment are setup by op.sh 
  using --okg4 option to select OKG4Test 


**/
EON
}


arg=${1:-box}
shift

cd /tmp

DIRECT=1


if [ $DIRECT -eq 1 ]
then 
    unset IDPATH
    geocache-
    geocache-key-export
    echo IDPATH : $IDPATH
fi


echo ====== $0 $arg $* ====== PWD $PWD =================

tboolean-
cmd="tboolean-$arg --okg4 --compute $*"
#cmd="tboolean-$arg --okg4  $*"
echo $cmd
eval $cmd
rc=$?

echo ====== $0 $arg $* ====== PWD $PWD ============ RC $rc =======

exit $rc


