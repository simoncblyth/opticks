#!/bin/bash 
usage(){ cat << EOU
cxr_cfbase.sh
================

Test CFBASE running, create::

    N[blyth@localhost CSG_GGeo]$ mkdir /tmp/cfbase
    N[blyth@localhost CSG_GGeo]$ cp -r /home/blyth/.opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry /tmp/cfbase/

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

    N[blyth@localhost CSGOptiX]$ l /tmp/cfbase/CSGFoundry/
    total 236
      0 drwxr-xr-x. 2 blyth blyth    209 Mar  8 19:06 .
      4 -rw-rw-r--. 1 blyth blyth     16 Mar  8 19:06 bnd_meta.txt
    192 -rw-rw-r--. 1 blyth blyth 194960 Mar  8 19:06 bnd.npy
      4 -rw-rw-r--. 1 blyth blyth    128 Mar  8 19:06 icdf.npy
      4 -rw-rw-r--. 1 blyth blyth    192 Mar  8 19:06 inst.npy
      4 -rw-rw-r--. 1 blyth blyth   2752 Mar  8 19:06 itra.npy
      4 -rw-rw-r--. 1 blyth blyth   3584 Mar  8 19:06 node.npy
      4 -rw-rw-r--. 1 blyth blyth   1920 Mar  8 19:06 prim.npy
      4 -rw-rw-r--. 1 blyth blyth   2752 Mar  8 19:06 tran.npy
      0 drwxrwxr-x. 3 blyth blyth     24 Mar  8 19:06 ..
      4 -rw-rw-r--. 1 blyth blyth     32 Mar  8 19:06 meshname.txt
      4 -rw-rw-r--. 1 blyth blyth     10 Mar  8 19:06 meta.txt
      4 -rw-rw-r--. 1 blyth blyth     15 Mar  8 19:06 mmlabel.txt
      4 -rw-rw-r--. 1 blyth blyth    176 Mar  8 19:06 solid.npy
    N[blyth@localhost CSGOptiX]$ 


transporting 

   cd /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry


EOU
}

msg="=== $BASH_SOURCE :"

export CFBASE=/tmp/cfbase
source ./cxr.sh


