#!/bin/bash -l
##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

arg=${1:-build_run}
sdir=$(pwd)
name=$(basename $sdir)

if [ "${arg/build}" != "$arg" ] ; then 

    opticks-
    oe-
    om-
    bdir=/tmp/$USER/opticks/$name/build 
    rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 
    om-cmake $sdir
    make
    [ $? -ne 0 ] && exit 1

    make install   
fi 


if [ "${arg/run}" != "$arg" ] ; then 

    echo executing $name
    export SHADER_FOLD=$sdir/rec_flying_point
    #export ARRAY_FOLD=/tmp/$USER/opticks/GeoChain/BoxedSphere/CXRaindropTest
    export ARRAY_FOLD=/tmp/$USER/opticks/QSimTest/mock_propagate

    if [ -n "$DEBUG" ]; then 
        lldb__ $name 
    else
        $name
    fi 
fi


