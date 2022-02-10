#!/bin/bash -l 
msg="=== $BASH_SOURCE :"

CSGQueryTest $* 
[ $? -ne 0 ] && echo $msg run error && exit 1

#${IPYTHON:-ipython} -i tests/CSGQueryTest.py 
#[ $? -ne 0 ] && echo $msg ana error && exit 2

exit 0 



