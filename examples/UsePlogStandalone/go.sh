#!/bin/bash -l 
msg="=== $BASH_SOURCE :"
name=UsePlogStandalone
srcdir=$PWD
tmpdir=/tmp/$USER/opticks/$name

url=https://github.com/simoncblyth/plog.git
#url=https://github.com/SergiusTheBest/plog

URL=${URL:-$url}
echo $msg URL $URL 

mkdir -p $tmpdir
cd $tmpdir
pwd

if [ -d "plog" ]; then
    echo $msg plog has already been cloned into $tmpdir/plog 
    echo $msg from URL : $(grep url $tmpdir/plog/.git/config) 
    echo $msg delete that plog folder to clone again from different URL
else
    echo $msg cloning from URL $URL   
    git clone $URL
fi 


echo $msg compile with c++11
gcc $srcdir/$name.cc -std=c++11 -lstdc++ -I$tmpdir/plog/include -o $tmpdir/$name 
[ $? -ne 0 ] && echo $msg compilation error with c++11 && exit 1 

if [ "$(uname)" == "Linux" ]; then 
    echo $msg compile with c++17
    gcc $srcdir/$name.cc -std=c++17 -lstdc++ -I$tmpdir/plog/include -o $tmpdir/$name 
    [ $? -ne 0 ] && echo $msg compilation error with c++17 && exit 1 
fi


export PLOGPATH=$tmpdir/$name.log 
$tmpdir/$name
[ $? -ne 0 ] && echo $msg run error && exit 2

echo $msg cat $PLOGPATH
cat $PLOGPATH

exit 0 
