#!/bin/bash 
source ../env.sh 
CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

#opts="-DDEBUG=1"
opts=""

name=ScanTest 
srcs="$name.cc ../CSGFoundry.cc ../CSGSolid.cc ../CSGPrim.cc ../CSGNode.cc ../Scan.cc ../CU.cc ../Tran.cc ../Util.cc ../Geo.cc ../Grid.cc ../View.cc"
gcc \
          $srcs \
          -I.. \
          -std=c++11 \
          $opts \
          -I${CUDA_PREFIX}/include \
          -I$PREFIX/externals/glm/glm \
          -L${CUDA_PREFIX}/lib -lcudart -lstdc++ \
          -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1

base=/tmp/ScanTest_scans

scans="axis rectangle circle"
for scan in $scans ; do 
    tmpdir=$base/${scan}_scan
    mkdir -p $tmpdir 
done 

case $(uname) in
  Darwin) var=DYLD_LIBRARY_PATH ;;
  Linux)  var=LD_LIBRARY_PATH ;;
esac
cmd="$var=${CUDA_PREFIX}/lib /tmp/$name $*"
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
   find $base -newer ScanTest.cc -exec ls -l {} \; 
}

#scan-all $scans
scan-recent 


ipython -i --pdb ScanTest.py 


exit 0 
