#!/bin/bash -l 

spec=$1

source ./env.sh 

echo RM OUTDIR $OUTDIR
rm -rf $OUTDIR
mkdir -p $OUTDIR


gdb -ex r --args $BIN $spec
#$BIN $spec
[ $? -ne 0 ] && echo $0 : run  FAIL && exit 3

ppm=$OUTDIR/pixels.ppm
npy=$OUTDIR/posi.npy

echo BIN    : $BIN 
echo OUTDIR : $OUTDIR
echo spec : $spec
echo ppm  : $ppm
echo md5  : $(cat $ppm | md5sum)
echo npy  : $npy
ls -l $ppm $npy 


exit 0

