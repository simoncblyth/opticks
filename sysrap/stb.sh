#!/bin/bash

usage(){ cat << EOU

https://github.com/nothings/stb/

https://github.com/nothings/stb/blob/master/stb_image.h
https://github.com/nothings/stb/blob/master/stb_image_write.h
https://github.com/nothings/stb/blob/master/stb_truetype.h

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

hh="stb_image.h stb_image_write.h stb_truetype.h"

urlbase=https://raw.githubusercontent.com/nothings/stb/master

for h in $hh ; do
   curl -L -O $urlbase/$h 
done




