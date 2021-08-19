#!/bin/bash -l 

nmm=${NMM:-9}   # geometry specific 
script=${SCRIPT:-cxr_overview}
export SCANNER="cxr_scan.sh"

usage(){ cat << EOU
::

    ./cxr_scan.sh 

EOU
}

scan-ee()
{
    echo "t0"
    for e in $(seq 0 $nmm) ; do echo  "$e," ; done
    for e in $(seq 0 $nmm) ; do echo "t$e," ; done
    for e in $(seq 0 $nmm) ; do echo "t8,$e" ; done
    echo "1,2,3,4"   # ONLY PMTs
}

for e in $(scan-ee) 
do 
    echo $e 
    EMM=$e ./$script.sh $*
done 


