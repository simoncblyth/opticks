#!/bin/bash -l 

nmm=${NMM:-9}   # geometry specific 

usage(){ cat << EOU
::

    snapscan.sh --cvd 1 --rtx 1 
    NMM=5 snapscan.sh --cvd 1 --rtx 1 

EOU
}

scan-ee()
{
    #echo "~0"
    #for e in $(seq 0 $nmm) ; do echo  "$e," ; done
    #for e in $(seq 0 $nmm) ; do echo "~$e," ; done
    for e in $(seq 0 $nmm) ; do echo "~8,$e" ; done
}

for e in $(scan-ee) 
do 
    echo $e 
    EMM="$e" snap.sh $*
done 


