#!/bin/bash -l 

fold=$(dirname $BASH_SOURCE)
bin=QCerenkovTest 
tmpdir=/tmp/$USER/opticks/$bin
mkdir -p $tmpdir

which $bin
$bin

ipython -i $fold/$bin.py  

