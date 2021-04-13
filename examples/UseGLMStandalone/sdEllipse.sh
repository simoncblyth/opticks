#!/bin/bash -l 

name=$(basename $0)
stem=${name/.sh}

glm-
gcc $stem.cc -I$HOME/np -std=c++11 -lstdc++ -I$(glm-prefix) -o /tmp/$stem && /tmp/$stem  


ipython -i $stem.py 

