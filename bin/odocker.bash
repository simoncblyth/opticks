odocker-source(){ echo $BASH_SOURCE ; }
odocker-vi(){ vi $(odocker-source)  ; }
odocker-env(){  olocal- ; opticks- ; }
odocker-usage(){ cat << EOU

Docker 
==========


Install
---------

* https://docs.docker.com/install/


Instructions to try to follow
---------------------------------

* https://devblogs.nvidia.com/gpu-containers-runtime/



* https://developer.ibm.com/linuxonpower/2018/09/19/using-nvidia-docker-2-0-rhel-7/

  Why dont need to install nvidia-docker-2 to get the OCI hook benefits with RHEL/CentOS.


LXC
------

* https://discuss.linuxcontainers.org/t/lxd-3-0-0-has-been-released/1491

A newly introduced nvidia.runtime container configuration key, combined with a
copy of the nvidia-container-cli tool and liblxc 3.0 now makes it possible to
automatically detect all the right bits on the host system and pass them into
the container at boot time.

This lets you save a lot of space and greatly simplifies maintenance.

Copy pasted from "video" https://asciinema.org/a/174076::

    ubuntu@canonical-lxd:~$ lxc launch ubuntu:16.04 c1 
    Creating c1
    Starting c1
    ubuntu@canonical-lxd:~$ lxc exec c1 bash
    root@c1:~# nvidia-smi
    nvidia-smi: command not found
    root@c1:~# exit
    ubuntu@canonical-lxd:~$ lxc config device add c1 gt730 gpu id=1
    Device gt730 added to c1 
    ubuntu@canonical-lxd:~$ lxc config set c1 nvidia.runtime true
    ubuntu@canonical-lxd:~$ lxc restart c1 
    ubuntu@canonical-lxd:~$ lxc exec c1 bash
    root@c1:~# nvidia-smi
    Tue Apr  3 02:47:37 2018
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 390.30                 Driver Version: 390.30                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GT 730      Off  | 00000000:02:06.0 N/A |                  N/A |
    | 30%   33C    P0    N/A /  N/A |      0MiB /  2002MiB |     N/A      Default |
    +-------------------------------+----------------------+----------------------+
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0                    Not Supported                                       |
    +-----------------------------------------------------------------------------+
    root@c1:~# exit
    ubuntu@canonical-lxd:~$                          





NGC : NVIDIA GPU Cloud
------------------------

* :google:`docker container Titan V`

* https://docs.nvidia.com/ngc/index.html
* https://docs.nvidia.com/ngc/ngc-titan-setup-guide/index.html

  Instructions for Ubuntu only.

* https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)

  * instructions for CentOS


* https://docs.nvidia.com/ngc/ngc-user-guide/singularity.html


CentOS 7 too old ?
~~~~~~~~~~~~~~~~~~~~

nvidia-docker 2.0 requires GNU/Linux x86_64 with kernel version > 3.10



* https://en.wikipedia.org/wiki/CentOS

::

    cat /etc/redhat-release 
    CentOS Linux release 7.5.1804 (Core) 

    Linux localhost.localdomain 3.10.0-862.6.3.el7.x86_64 #1 SMP Tue Jun 26 16:32:21 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux


 
Question : would docker containers allow me to fully test Opticks with different Linux distros? 
------------------------------------------------------------------------------------------------

vbx- for checking Ubuntu was real handly but virtualbox with access to GPU
is too difficult : maybe can do similar with docker or LXC ?




EOU
}
