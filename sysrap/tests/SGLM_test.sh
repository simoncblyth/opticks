#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
name=SGLM_test 

gcc $name.cc -std=c++11 -lstdc++ -I.. -I$OPTICKS_PREFIX/externals/glm/glm -o /tmp/$name 
[ $? -ne 0 ] && echo $msg compile error && exit 1 


/tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 2 

exit 0 


