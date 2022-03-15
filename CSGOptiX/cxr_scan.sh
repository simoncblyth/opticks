#!/bin/bash -l 
usage(){ cat << EOU
cxr_scan.sh
==============

Repeats a script such as cxr_overview.sh with EMM
envvar variation to change enabledmergedmesh. 

On GPU workstation::

    ./cxr_scan.sh 

On laptop::

    ./cf_grab.sh    # grab remote CSGFoundry dir for metadata
    ./cxr_table.sh  # grab remote .jpg renders and .json sidecars and make RST table 

+---+----------+------------------+------------------+-----------------------------------------------+
|idx|        -e|       time(s)    |      relative    |    enabled geometry description 41c046fe      |
+===+==========+==================+==================+===============================================+
|  0|        5,|        0.0004    |        0.0004    |    ONLY: 1:sStrutBallhead                     |
+---+----------+------------------+------------------+-----------------------------------------------+
|  1|        9,|        0.0006    |        0.0006    |    ONLY: 130:sPanel                           |
+---+----------+------------------+------------------+-----------------------------------------------+
|  2|        7,|        0.0006    |        0.0006    |    ONLY: 1:base_steel                         |
+---+----------+------------------+------------------+-----------------------------------------------+
|  3|        8,|        0.0007    |        0.0007    |    ONLY: 1:uni_acrylic1                       |
+---+----------+------------------+------------------+-----------------------------------------------+
|  4|        6,|        0.0008    |        0.0008    |    ONLY: 1:uni1                               |
+---+----------+------------------+------------------+-----------------------------------------------+
|  5|        1,|        0.0010    |        0.0010    |    ONLY: 5:PMT_3inch_pmt_solid                |
+---+----------+------------------+------------------+-----------------------------------------------+
|  6|        4,|        0.0020    |        0.0020    |    ONLY: 5:mask_PMT_20inch_vetosMask_virtual  |
+---+----------+------------------+------------------+-----------------------------------------------+
|  7|        3,|        0.0048    |        0.0048    |    ONLY: 6:HamamatsuR12860sMask_virtual       |
+---+----------+------------------+------------------+-----------------------------------------------+
|  8|        2,|        0.0050    |        0.0050    |    ONLY: 7:NNVTMCPPMTsMask_virtual            |
+---+----------+------------------+------------------+-----------------------------------------------+
|  9|        0,|        0.0068    |        0.0068    |    ONLY: 3089:sWorld                          |
+---+----------+------------------+------------------+-----------------------------------------------+
| 10|        t0|        0.0123    |        0.0123    |    ALL                                        |
+---+----------+------------------+------------------+-----------------------------------------------+

EOU
}

nmm=${NMM:-9}   # geometry specific 
nlv=${NLV:-141}
## hmm could generate a metadata bash script to provide this kinda thing in the geocache


script=${SCRIPT:-cxr_overview}
export SCANNER="cxr_scan.sh"

scan-emm-()
{
    echo "t0"        # ALL 
    for e in $(seq 0 $nmm) ; do echo  "$e," ; done    # enabling each solid one-by-one
    for e in $(seq 0 $nmm) ; do echo "t$e," ; done    # disabling each solid one-by-one
    for e in $(seq 0 $nmm) ; do echo "t8,$e" ; done   # disabling 8 and each solid one by-by-one
    echo "1,2,3,4"   # ONLY PMTs
}

scan-elv-()
{
    for e in $(seq 0 $nlv) ; do echo "t$e" ; done    # disabling each midx one-by-one
    for e in $(seq 0 $nlv) ; do echo "$e" ; done     # enabling each midx one-by-one
}

scan-emm()
{
    local e 
    for e in $(scan-emm-) 
    do 
        echo $e 
        EMM=$e ./$script.sh $*
    done 
}

scan-elv()
{
    local e 
    for e in $(scan-elv-) 
    do 
        echo $e 
        ELV=$e ./$script.sh $*
    done 
}

scan-elv


