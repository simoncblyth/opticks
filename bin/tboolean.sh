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


Minimizing this
-----------------

Bash on macOS dislikes too many layers of bash scripts
and functions (its failing to pass in a TESTCONFIG),
so moved most of the argument setup into tboolean-
leaving just directory and environment setup here. 

**/
EON
}


cd /tmp
DIRECT=1

if [ $DIRECT -eq 1 ]; then 
    unset IDPATH
    geocache-
    geocache-key-export
    [ -n "$IDPATH" ] && echo $0 ERROR IDPATH should not be defined in direct running : $IDPATH && exit 101
fi

echo ====== $0 $arg $* ====== PWD $PWD =================

tboolean-
cmd="tboolean-lv $*"

echo $cmd
eval $cmd
rc=$?

echo ====== $0 $arg $* ====== PWD $PWD ============ RC $rc =======

exit $rc
