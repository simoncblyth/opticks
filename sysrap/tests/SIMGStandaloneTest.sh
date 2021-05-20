#!/bin/bash -l 

name=SIMGStandaloneTest 

# attempt to suppress "was set but never used" warnings
# from complilation of stb_image.h using the below causing error 
# -Xcudafe "–-diag_suppress=set_but_not_used" 

nvcc $name.cu -lstdc++ -std=c++11  -I.. -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcudart -o /tmp/$name 
[ $? -ne 0 ] && echo compile FAIL && exit 1

/tmp/$name $*
[ $? -ne 0 ] && echo run FAIL && exit 2

exit 0 

