#!/bin/bash -l 

name=sdigest_test 

opt="-Wdeprecated-declarations"
case $(uname) in 
  Darwin) opt="" ;;
   Linux) opt="-lssl -lcrypto " ;;
esac

vccp=${VCCP:-11}
echo $BASH_SOURCE : VCCP $vccp
gcc $name.cc -std=c++$vccp -Wall -lstdc++ $opt -I.. -o /tmp/$name 
[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0 


