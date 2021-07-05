#!/bin/bash -l 

dir=/tmp/QCtxTest


QCtxTest 
ls -l $dir/wavelength_20.npy

QCTX_DISABLE_HD=1 QCtxTest
ls -l $dir/wavelength_0.npy

QSCINT_DISABLE_INTERPOLATION=1 QCtxTest 
ls -l $dir/wavelength_20_cudaFilterModePoint.npy

QCTX_DISABLE_HD=1 QSCINT_DISABLE_INTERPOLATION=1 QCtxTest 
ls -l $dir/wavelength_0_cudaFilterModePoint.npy

ls -l $dir


