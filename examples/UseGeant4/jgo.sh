#!/bin/bash -l

opticks-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


#unset JUNOTOP

if [ -z "$JUNOTOP" ]; then 
   echo missing JUNOTOP
else
   source $JUNOTOP/bashrc.sh
   if [ ! -d "$JUNO_EXTLIB_Geant4_HOME" ]; then 
       echo missing JUNO_EXTLIB_Geant4_HOME
       exit 1 
   fi
   env | grep JUNO_EXTLIB_Geant4 

   #jg4cmakepath=$(find $JUNO_EXTLIB_Geant4_HOME -name UseGeant4.cmake) 
   #if [ ! -f "$jg4cmakepath" ]; then 
   #    echo missing jg4cmakepath
   #    exit 2
   #fi
   #jg4cmakedir=$(dirname $jg4cmakepath)
fi

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$(opticks-prefix)/externals

echo $CMAKE_PREFIX_PATH | tr ":" "\n"

cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
     -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
     -DOPTICKS_PREFIX=$(opticks-prefix)


cat << EON > /dev/null
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DGeant4_DIR=$jg4cmakedir    

Without the nudge -DGeant4_DIR the non-juno Geant4 is found 
unless avoid override of CMAKE_PREFIX_PATH envvar in the
cmake line.




EON


make
make install   


UseGeant4


