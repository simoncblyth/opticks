#!/bin/bash
usage(){ cat << EOU
scontext_test.sh
=================

This script must currently use the CMake built scontext_test
executable not the executable that this script was formerly
able to build.

TODO : get this standalone script to compile again by better
       dependency control with SEventConfig.cc (and dependencies)


::

    ~/o/sysrap/tests/scontext_test.sh run


* note that the nvidia-smi ordering does not necessarily match the CUDA ordering
* also I think that nvidia-smi ordering may change after reboots
* also the CUDA ordering can be changed with eg CUDA_VISIBLE_DEVICES=1,0

::

    N[blyth@localhost tests]$ ~/o/sysrap/tests/scontext_test.sh run
    0:TITAN_V 1:TITAN_RTX
    N[blyth@localhost tests]$ CUDA_VISIBLE_DEVICES=1 ~/o/sysrap/tests/scontext_test.sh
    1:TITAN_RTX
    N[blyth@localhost tests]$ CUDA_VISIBLE_DEVICES=0 ~/o/sysrap/tests/scontext_test.sh
    0:TITAN_V
    N[blyth@localhost tests]$ CUDA_VISIBLE_DEVICES=1 ~/o/sysrap/tests/scontext_test.sh
    1:TITAN_RTX
    N[blyth@localhost tests]$ CUDA_VISIBLE_DEVICES=0,1 ~/o/sysrap/tests/scontext_test.sh
    0:TITAN_V 1:TITAN_RTX
    N[blyth@localhost tests]$ CUDA_VISIBLE_DEVICES=1,0 ~/o/sysrap/tests/scontext_test.sh
    1:TITAN_RTX 0:TITAN_V
    N[blyth@localhost tests]$



    N[blyth@localhost tests]$ VERBOSE=1 ~/o/sysrap/tests/scontext_test.sh
    scontext::desc
    all_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    visible_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX


    N[blyth@localhost tests]$ VERBOSE=1 CUDA_VISIBLE_DEVICES=0 ~/o/sysrap/tests/scontext_test.sh
    scontext::desc
    all_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    visible_devices
    [0:TITAN_V]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V

    N[blyth@localhost tests]$ VERBOSE=1 CUDA_VISIBLE_DEVICES=1 ~/o/sysrap/tests/scontext_test.sh
    scontext::desc
    all_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    visible_devices
    [1:TITAN_RTX]
    idx/ord/mpc/cc:0/1/72/75  23.652 GB  TITAN RTX

    N[blyth@localhost tests]$ VERBOSE=1 CUDA_VISIBLE_DEVICES=1,0 ~/o/sysrap/tests/scontext_test.sh
    scontext::desc
    all_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    visible_devices
    [1:TITAN_RTX 0:TITAN_V]
    idx/ord/mpc/cc:0/1/72/75  23.652 GB  TITAN RTX
    idx/ord/mpc/cc:1/0/80/70  11.784 GB  TITAN V

    N[blyth@localhost tests]$ VERBOSE=1 CUDA_VISIBLE_DEVICES=0,1 ~/o/sysrap/tests/scontext_test.sh
    scontext::desc
    all_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    visible_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX



    N[blyth@localhost tests]$ nvidia-smi
    Mon Jun  5 19:51:47 2023
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN RTX           Off  | 00000000:73:00.0 Off |                  N/A |
    | 39%   55C    P0    71W / 280W |      0MiB / 24219MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  TITAN V             Off  | 00000000:A6:00.0 Off |                  N/A |
    | 46%   54C    P8    N/A /  N/A |      0MiB / 12066MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+




Try deleting ~/.opticks/scontext and see how to recreate
-----------------------------------------------------------

* THESE ARE OLD NOTES PRIOR TO MOVE TO USE ~/.opticks/sdevice/sdevice.bin


Initially running works::

    A[blyth@localhost ~]$ opticks/sysrap/tests/scontext_test.sh run
    0:NVIDIA_RTX_5000_Ada_GenerationA[blyth@localhost ~]$ l ~/.opticks/scontext/
    total 4
    4 -rw-r--r--. 1 blyth blyth 304 Aug 30 16:07 sdevice.bin
    0 drwxr-xr-x. 7 blyth blyth  86 Aug 30 09:51 ..
    0 drwxr-xr-x. 2 blyth blyth  25 Aug 29 22:06 .

Remove the scontext directory causes the expected error::

    A[blyth@localhost ~]$ rm -rf  ~/.opticks/scontext
    A[blyth@localhost ~]$ opticks/sysrap/tests/scontext_test.sh run
    sdevice::Load failed read from  dirpath_ /home/blyth/.opticks/scontext dirpath /home/blyth/.opticks/scontext path /home/blyth/.opticks/scontext/sdevice.bin
    sdevice::Load failed read from  dirpath_ /home/blyth/.opticks/scontext dirpath /home/blyth/.opticks/scontext path /home/blyth/.opticks/scontext/sdevice.bin
    -1:NVIDIA_RTX_5000_Ada_GenerationA[blyth@localhost ~]$

And the running did not create the directory and sdevice.bin file::

    A[blyth@localhost ~]$ l ~/.opticks/scontext
    total 0
    0 drwxr-xr-x. 2 blyth blyth  6 Feb 14 15:26 .
    0 drwxr-xr-x. 7 blyth blyth 86 Feb 14 15:26 ..

Observe that CUDA_VISIBLE_DEVICES is defined::

    A[blyth@localhost ~]$ echo $CUDA_VISIBLE_DEVICES
    0

Only with CUDA_VISIBLE_DEVICES unset does the file get persisted::

    A[blyth@localhost ~]$ unset CUDA_VISIBLE_DEVICES
    A[blyth@localhost ~]$ opticks/sysrap/tests/scontext_test.sh run
    0:NVIDIA_RTX_5000_Ada_GenerationA[blyth@localhost ~]$
    A[blyth@localhost ~]$
    A[blyth@localhost ~]$ l ~/.opticks/scontext/
    total 4
    0 drwxr-xr-x. 2 blyth blyth  25 Feb 14 15:30 .
    4 -rw-r--r--. 1 blyth blyth 304 Feb 14 15:30 sdevice.bin
    0 drwxr-xr-x. 7 blyth blyth  86 Feb 14 15:26 ..
    A[blyth@localhost ~]$ opticks/sysrap/tests/scontext_test.sh run
    0:NVIDIA_RTX_5000_Ada_GenerationA[blyth@localhost ~]$
    A[blyth@localhost ~]$

How about setting it blank rather than unset ?::

    A[blyth@localhost ~]$ rm -rf  ~/.opticks/scontext
    A[blyth@localhost ~]$ CUDA_VISIBLE_DEVICES="" opticks/sysrap/tests/scontext_test.sh run
    sdevice::Load failed read from  dirpath_ /home/blyth/.opticks/scontext dirpath /home/blyth/.opticks/scontext path /home/blyth/.opticks/scontext/sdevice.bin
    sdevice::Load failed read from  dirpath_ /home/blyth/.opticks/scontext dirpath /home/blyth/.opticks/scontext path /home/blyth/.opticks/scontext/sdevice.bin
    scontext::initConfig : ZERO VISIBLE DEVICES - CHECK CUDA_VISIBLE_DEVICES envvar
    A[blyth@localhost ~]$

    A[blyth@localhost ~]$ rm -rf  ~/.opticks/scontext
    A[blyth@localhost ~]$ CUDA_VISIBLE_DEVICES= opticks/sysrap/tests/scontext_test.sh run
    sdevice::Load failed read from  dirpath_ /home/blyth/.opticks/scontext dirpath /home/blyth/.opticks/scontext path /home/blyth/.opticks/scontext/sdevice.bin
    sdevice::Load failed read from  dirpath_ /home/blyth/.opticks/scontext dirpath /home/blyth/.opticks/scontext path /home/blyth/.opticks/scontext/sdevice.bin
    scontext::initConfig : ZERO VISIBLE DEVICES - CHECK CUDA_VISIBLE_DEVICES envvar
    A[blyth@localhost ~]$

Nope it does need to be unset::

    A[blyth@localhost ~]$ rm -rf  ~/.opticks/scontext
    A[blyth@localhost ~]$ unset CUDA_VISIBLE_DEVICES ; opticks/sysrap/tests/scontext_test.sh run
    0:NVIDIA_RTX_5000_Ada_GenerationA[blyth@localhost ~]$
    A[blyth@localhost ~]$
    A[blyth@localhost ~]$ l ~/.opticks/scontext/
    total 4
    0 drwxr-xr-x. 2 blyth blyth  25 Feb 14 15:42 .
    4 -rw-r--r--. 1 blyth blyth 304 Feb 14 15:42 sdevice.bin
    0 drwxr-xr-x. 7 blyth blyth  86 Feb 14 15:42 ..
    A[blyth@localhost ~]$




EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

name=scontext_test
bin=$name

defarg="info_run"
arg=${1:-$defarg}

vars="BASH_SOURCE 0 PWD name TMP bin defarg arg CUDA_VISIBLE_DEVICES"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi



exit 0


