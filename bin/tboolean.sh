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


default="box"
[ -n "$PROXYLV" ] && default="proxy"


## only shift out the first argument when it doesnt start with a hyphen
## this avoids the need to provide the default arg when wish to set options

if [ ${#} -eq 0 ]; then
   arg=$default 
elif [ "${1:0:1}" == "-" ]; then  
   arg=$default 
else
   arg=$1
   shift
fi 


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
echo $cmd
eval $cmd
rc=$?

echo ====== $0 $arg $* ====== PWD $PWD ============ RC $rc =======

exit $rc


