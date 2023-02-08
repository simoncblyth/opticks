#!/bin/bash -l 

SDIR=$(dirname $BASH_SOURCE)

last=10
export LAST=${LAST:-$last}

$SDIR/rst.sh 
$SDIR/txt.sh 
$SDIR/cc.sh 
$SDIR/h.sh 
$SDIR/hh.sh


 
