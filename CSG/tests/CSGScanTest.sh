#!/bin/bash 
#source ../env.sh 
CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

#opts="-DDEBUG=1"
opts=""

name=CSGScanTest 
srcs="$name.cc 
      ../CSGFoundry.cc 
      ../CSGSolid.cc 
      ../CSGPrim.cc 
      ../CSGPrimSpec.cc 
      ../CSGNode.cc 
      ../CSGScan.cc 
      ../CSGName.cc 
      ../CSGTarget.cc 
      ../CSGMaker.cc 
      ../CU.cc 
      "

gcc \
    $srcs \
    -I.. \
    -std=c++11 \
    $opts \
    -I${CUDA_PREFIX}/include \
    -I${OPTICKS_PREFIX}/externals/glm/glm \
    -I${OPTICKS_PREFIX}/include/SysRap \
    -I${OPTICKS_PREFIX}/externals/plog/include \
    -L${CUDA_PREFIX}/lib -lcudart -lstdc++ \
    -L${OPTICKS_PREFIX}/lib \
    -lSysRap \
    -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1

base=/tmp/$USER/opticks/$name
#base=/tmp/CSGScanTest_scans

export CSGSCANTEST_BASE=$base
export CSGSCANTEST_SOLID=icyl


scans="axis rectangle circle"
for scan in $scans ; do 
    tmpdir=$base/${scan}_scan
    mkdir -p $tmpdir 
done 

case $(uname) in
  Darwin) var=DYLD_LIBRARY_PATH ;;
  Linux)  var=LD_LIBRARY_PATH ;;
esac
cmd="$var=${CUDA_PREFIX}/lib:${OPTICKS_PREFIX}/lib /tmp/$name $*"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2

scan-all()
{
    echo $FUNCNAME $*
    local scan
    for scan in $* ; do 
       tmpdir=$base/${scan}_scan
       echo $tmpdir
       ls -l $tmpdir
    done 
}
scan-recent(){
   echo $FUNCNAME 
   find $base -newer CSGScanTest.cc -exec ls -l {} \; 
}

#scan-all $scans
scan-recent 


${IPYTHON:-ipython} -i --pdb CSGScanTest.py 


exit 0 
