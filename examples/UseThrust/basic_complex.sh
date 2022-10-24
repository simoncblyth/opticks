#!/bin/bash -l 
usage(){ cat << EOU
basic_complex.sh 
==================

::

   ./basic_complex.sh dev     # on device with thrust::complex then host with thrust::complex
   ./basic_complex.sh host    # on host using thrust::complex
   ./basic_complex.sh std     # on host using std::complex

EOU
}

name=basic_complex 

defarg=dev
arg=${1:-$defarg}

if [ "$arg" == "ALL" ]; then

   ./basic_complex.sh dev 
   ./basic_complex.sh host
   ./basic_complex.sh std

fi 


if [ "${arg/dev}" != "$arg" ]; then 

    nvcc $name.cu \
          -std=c++11 -lstdc++ \
           -I. -I/usr/local/cuda/include \
            -DWITH_THRUST -o /tmp/$name 

    [ $? -ne 0 ] && echo $BASH_SOURCE nvcc build fail && exit 1 

    /tmp/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE run fail && exit 2 
fi 


if [ "${arg/host}" != "$arg" ]; then 

    hname=${name}_host

    gcc $hname.cc \
          -std=c++11 -lstdc++ \
           -I. -I/usr/local/cuda/include \
            -DWITH_THRUST -o /tmp/$hname

    [ $? -ne 0 ] && echo $BASH_SOURCE $hname build fail && exit 10 

    /tmp/$hname
    [ $? -ne 0 ] && echo $BASH_SOURCE $hname run fail && exit 20 

fi 


if [ "${arg/std}" != "$arg" ]; then 

    hname=${name}_host
    gcc $hname.cc \
          -std=c++11 -lstdc++ \
           -I. -I/usr/local/cuda/include \
            -o /tmp/${hname}_std

    [ $? -ne 0 ] && echo $BASH_SOURCE ${hname}_std build fail && exit 10 

    /tmp/${hname}_std
    [ $? -ne 0 ] && echo $BASH_SOURCE ${hname}_std run fail && exit 20 

fi 

exit 0


