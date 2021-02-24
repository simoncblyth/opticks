#!/bin/bash -l 

source ./env.sh 

[ -z "$OUTDIR" ] && echo OUTDIR not defined && return 1 

ppm=$OUTDIR/pixels.ppm
npy=$OUTDIR/posi.npy

ppm_()
{
   echo scp P:$ppm $ppm
   scp P:$ppm $ppm
   open $ppm 
}

npy_()
{
   echo scp P:$npy $npy
   scp P:$npy $npy
   ipython -i posi.py  
}


if [ "$1" == "ppm" ]; then
   ppm_
elif [ "$1" == "npy" ]; then
   npy_
else
   ppm_
   npy_
fi 




