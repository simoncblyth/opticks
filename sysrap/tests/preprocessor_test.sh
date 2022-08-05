#!/bin/bash -l 

name=preprocessor_test 
vars="WITH_RED WITH_GREEN WITH_BLUE WITH_PUCE"
for var in $vars ; do 
    gcc $name.cc -std=c++11 -lstdc++ -D$var -o /tmp/${name}_${var}  && /tmp/${name}_${var}
done 

