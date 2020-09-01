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


opticks-
oe-
om-

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

echo bdir $bdir name $name

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


om-cmake $sdir


make
[ ! $? -eq 0 ] && echo build error && exit 1

make install   


runline(){ cat << EOL | grep -v \#
#lldb_ $1
#$1 $HOME/opticks_refs/Earth_Albedo_8192_4096.ppm
#$1 /tmp/SPPMTest.ppm
#$1 /tmp/SPPMTest2.ppm     
$1 /tmp/SPPMTest2.ppm --latlon 50.8919,-1.4483 --tanyfov 0.2
EOL
}

cmd=$(runline $name)
echo $cmd
eval $cmd

## see ImageNPYTest for creation of /tmp/SPPMTest2.ppm 
 
[ ! $? -eq 0 ] && echo runtime error && exit 1


npd-(){ cat << EOP
import os, numpy as np 
ipath = os.path.expandvars('$TMP/$1/inp.npy')
opath = os.path.expandvars('$TMP/$1/out.npy')
i = np.load(ipath)
o = np.load(opath)
print("npd inp %s %r " % (ipath,i.shape) )
print(i) 
print("npd out %s %r " % (opath,o.shape) )
print(o) 
EOP
} 
#npd- $name | python

cd $sdir

outpath=$TMP/$name/out.ppm
ls -l $outpath

if [ -n "$SSH_TTY" ]; then 
    echo local running outpath $outpath
    open $outpath
else
    echo remote running outpath $outpath
fi 

ipython -i dbg.py 


