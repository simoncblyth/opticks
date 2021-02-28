#!/bin/bash -l 

source ./env.sh 

[ -z "$OUTDIR" ] && echo OUTDIR not defined && return 1 

ppm=$OUTDIR/pixels.ppm
jpg=$OUTDIR/pixels.jpg
npy=$OUTDIR/posi.npy
mkdir -p $OUTDIR 


ppm_()
{
   local cmd="scp P:$ppm $ppm"
   echo $cmd
   eval $cmd
   open $ppm 
}
jpg_()
{
   local cmd="scp P:$jpg $jpg"
   echo $cmd
   eval $cmd
   open $jpg 
}

npy_()
{
   local cmd="scp P:$npy $npy"
   echo $cmd
   eval $cmd
   ipython -i posi.py  
}

all_()
{
   local cmd="rsync -rtz --del --progress P:$OUTDIR/ $OUTDIR/"
   echo $cmd
   eval $cmd

   echo $ppm
   ls -l $ppm
   open $ppm

}

if [ "$1" == "ppm" ]; then
   ppm_
elif [ "$1" == "jpg" ]; then
   jpg_
elif [ "$1" == "npy" ]; then
   npy_
elif [ "$1" == "all" ]; then
   all_
else
   all_
fi 




