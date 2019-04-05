olxd-source(){ echo $BASH_SOURCE ; }
olxd-vi(){ vi $(olxd-source)  ; }
olxd-env(){  olocal- ; opticks- ; }
olxd-usage(){ cat << EOU

LXD/LXC
==========

linuxcontainers.org is the umbrella project behind LXC, LXD and LXCFS.

* https://linuxcontainers.org/
* https://linuxcontainers.org/lxd/introduction/
* https://linuxcontainers.org/lxd/try-it/

  A Good intro, by trying it 


Snap Package
-------------

LXD works on any recent Linux distribution. LXD upstream directly maintains the
Ubuntu packages and also publishes a snap package which can be used with most
of the popular Linux distributions.
 

Community Guide
------------------

* https://www.cyberciti.biz/faq/how-to-set-up-and-use-lxd-on-centos-linux-7-x-server/



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


EOU
}
