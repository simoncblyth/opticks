#!/bin/bash 
usage(){ cat << EOU
cxr_cfbase.sh
================

Test CFBASE running, create::

    epsilon:~ blyth$ ini
    epsilon:~ blyth$ echo $EG_CFBASE
    /usr/local/opticks/geocache/OKX4Test_lWorld0x5780b30_PV_g4live/g4ok_gltf/5303cd587554cb16682990189831ae83/1/CSG_GGeo
    epsilon:~ blyth$ 
    epsilon:~ blyth$ 
    epsilon:~ blyth$ mkdir /tmp/cfbase
    epsilon:~ blyth$ cp -r $EG_CFBASE/CSGFoundry /tmp/cfbase/
    epsilon:~ blyth$ l /tmp/cfbase/
    total 0
    0 drwxr-xr-x  15 blyth  wheel   480 Mar  8 10:45 CSGFoundry
    0 drwxr-xr-x   3 blyth  wheel    96 Mar  8 10:45 .
    0 drwxrwxrwt  48 root   wheel  1536 Mar  8 10:45 ..
    epsilon:~ blyth$ 

EOU
}

msg="=== $BASH_SOURCE :"

export CFBASE=/tmp/cfbase
source ./cxr.sh



