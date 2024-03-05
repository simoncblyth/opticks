#!/bin/bash -l
usage(){ cat << EOU
examples/UseOpticksGLFWNoCMake/go.sh
=====================================

"oc" is no longer maintained : so this needed reworking 

::

    ~/o/examples/UseOpticksGLFWNoCMake/go.sh



EOU
}

opticks-
oe-

path=$(realpath $BASH_SOURCE)
sdir=$(dirname $path)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

pkg=OpticksGLFW
pc=glfw3    

echo gcc -c $sdir/Use$pkg.cc $(oc -cflags $pc)
     gcc -c $sdir/Use$pkg.cc $(oc -cflags $pc)
echo gcc Use$pkg.o -o Use$pkg $(oc -libs $pc --static) -lstdc++ 
     gcc Use$pkg.o -o Use$pkg $(oc -libs $pc --static) -lstdc++
echo DISPLAY=:0 ./Use$pkg
     DISPLAY=:0 ./Use$pkg

printf "\n\n%s\n"  "known to fail when headless"


