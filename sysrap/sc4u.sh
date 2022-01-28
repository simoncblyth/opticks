#!/bin/bash 

name=sc4u 
gcc $name.cc -std=c++11 -lstdc++ -I. -I/usr/local/cuda/include -o /tmp/$name 
[ $? -ne 0 ] && echo compile error && exit 1

/tmp/$name 
[ $? -ne 0 ] && echo run error && exit 2

ipython -i -c "import numpy as np ; a = np.load('/tmp/p.npy') ; print(a.view(np.int8)) " 
[ $? -ne 0 ] && echo ana error && exit 3

exit 0 

