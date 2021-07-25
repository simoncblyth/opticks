#!/bin/bash -l 

usage(){ cat << EOU

::

    TEST=Y ./tests/QCtxTest.sh 

EOU
}


export TEST=${TEST:-K}
echo $BASH_SOURCE TEST $TEST 

dir=/tmp/QCtxTest
mkdir -p $dir 

QCtxTest 
[ $? -ne 0 ] && echo runtime fail && exit 1  

ipython -i tests/QCtxTest.py  
[ $? -ne 0 ] && echo analysis fail && exit 2  

exit 0 


old()
{
    QCtxTest 
    ls -l $dir/wavelength_20.npy

    QCTX_DISABLE_HD=1 QCtxTest
    ls -l $dir/wavelength_0.npy

    QSCINT_DISABLE_INTERPOLATION=1 QCtxTest 
    ls -l $dir/wavelength_20_cudaFilterModePoint.npy

    QCTX_DISABLE_HD=1 QSCINT_DISABLE_INTERPOLATION=1 QCtxTest 
    ls -l $dir/wavelength_0_cudaFilterModePoint.npy

    ls -l $dir
}

