olxd-source(){ echo $BASH_SOURCE ; }
olxd-vi(){ vi $(olxd-source)  ; }
olxd-env(){  olocal- ; opticks- ; }
olxd-usage(){ cat << EOU

LXD : pronounced Lex-Dee
=========================

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

* https://docs.snapcraft.io/installing-snapd/6735
* https://docs.snapcraft.io/installing-snap-on-centos/10020

::

    sudo yum install snapd

    [blyth@localhost opticks]$ sudo systemctl enable --now snapd.socket
    Created symlink from /etc/systemd/system/sockets.target.wants/snapd.socket to /usr/lib/systemd/system/snapd.socket.


::

    [blyth@localhost opticks]$ snap install lxd
    error: too early for operation, device not yet seeded or device model not acknowledged

    [blyth@localhost opticks]$ sudo  snap install lxd           ## appeared to work but quite a few SELinux denials, so might be borked
    2019-04-05T21:36:43+08:00 INFO Waiting for restart...
    lxd 3.11 from Canonical✓ installed


    [blyth@localhost opticks]$ snap run lvd
    error: cannot find current revision for snap lvd: readlink /var/lib/snapd/snap/lvd/current: no such file or directory


Initially could not find lxd.  After "sudo yum update" get further, but still lots of SELinux denials.

::

    [blyth@localhost opticks]$ lxd
    2019/04/05 22:19:09.397308 cmd_run.go:367: restoring default SELinux context of /home/blyth/snap
    Error: This must be run as root
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ sudo lxd
    sudo: lxd: command not found
    [blyth@localhost opticks]$ sudo $(which lxd)
    EROR[04-05|22:19:32] Failed to start the daemon: LXD is already running 
    Error: LXD is already running
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ which lxd
    /var/lib/snapd/snap/bin/lxd
    [blyth@localhost opticks]$ 


::

    [blyth@localhost opticks]$ lxc list 
    If this is your first time running LXD on this machine, you should also run: lxd init
    To start your first container, try: lxc launch ubuntu:18.04

    Error: Get http://unix.socket/1.0: dial unix /var/snap/lxd/common/lxd/unix.socket: connect: permission denied
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ sudo lxd init
    [sudo] password for blyth: 
    sudo: lxd: command not found
    [blyth@localhost opticks]$ sudo $(which lxd) init


    blyth@localhost opticks]$ sudo $(which lxd) init
    Would you like to use LXD clustering? (yes/no) [default=no]: 
    Do you want to configure a new storage pool? (yes/no) [default=yes]: 
    Name of the new storage pool [default=default]: 
    Name of the storage backend to use (btrfs, ceph, dir, lvm) [default=btrfs]: 
    Create a new BTRFS pool? (yes/no) [default=yes]: 
    Would you like to use an existing block device? (yes/no) [default=no]: 
    Size in GB of the new loop device (1GB minimum) [default=15GB]: 
    Would you like to connect to a MAAS server? (yes/no) [default=no]: 
    Would you like to create a new local network bridge? (yes/no) [default=yes]: 
    What should the new bridge be called? [default=lxdbr0]: 
    What IPv4 address should be used? (CIDR subnet notation, “auto” or “none”) [default=auto]: 
    What IPv6 address should be used? (CIDR subnet notation, “auto” or “none”) [default=auto]: 
    Would you like LXD to be available over the network? (yes/no) [default=no]: 
    Would you like stale cached images to be updated automatically? (yes/no) [default=yes] 
    Would you like a YAML "lxd init" preseed to be printed? (yes/no) [default=no]: 
    [blyth@localhost opticks]$ 


    blyth@localhost opticks]$ sudo $(which lxc) launch ubuntu:18.04
    To start your first container, try: lxc launch ubuntu:18.04

    Creating the container
    Container name is: sound-shad               
    Starting sound-shad
    Error: Failed to run: /snap/lxd/current/bin/lxd forkstart sound-shad /var/snap/lxd/common/lxd/containers /var/snap/lxd/common/lxd/logs/sound-shad/lxc.conf: 
    Try `lxc info --show-log local:sound-shad` for more info
    [blyth@localhost opticks]$ 

    blyth@localhost opticks]$ sudo $(which lxc) info --show-log local:sound-shad
    Name: sound-shad
    Location: none
    Remote: unix://
    Architecture: x86_64
    Created: 2019/04/05 14:26 UTC
    Status: Stopped
    Type: persistent
    Profiles: default

    Log:

    lxc sound-shad 20190405142644.603 ERROR    start - start.c:lxc_spawn:1703 - Invalid argument - Failed to clone a new set of namespaces
    lxc sound-shad 20190405142644.661 WARN     network - network.c:lxc_delete_network_priv:2613 - Invalid argument - Failed to remove interface "vethS0MP5P" from "lxdbr0"
    lxc sound-shad 20190405142644.661 ERROR    start - start.c:__lxc_start:1975 - Failed to spawn container "sound-shad"
    lxc sound-shad 20190405142644.661 ERROR    lxccontainer - lxccontainer.c:wait_on_daemonized_start:864 - Received container state "ABORTING" instead of "RUNNING"
    lxc sound-shad 20190405142644.662 ERROR    conf - conf.c:userns_exec_1:4397 - Failed to clone process in new user namespace
    lxc sound-shad 20190405142644.662 WARN     cgfsng - cgroups/cgfsng.c:cgfsng_payload_destroy:1122 - Failed to destroy cgroups
    lxc 20190405142644.693 WARN     commands - commands.c:lxc_cmd_rsp_recv:132 - Connection reset by peer - Failed to receive response for command "get_state"




Suspect all the issue are from SELinux enforcing.


To switch off without changing /etc/selinux/config amd rebooting::

    [blyth@localhost opticks]$ getenforce
    Enforcing
    [blyth@localhost opticks]$ sudo setenforce 0
    [sudo] password for blyth: 
    [blyth@localhost opticks]$ getenforce
    Permissive
    [blyth@localhost opticks]$ sudo setenforce 1
    [blyth@localhost opticks]$ getenforce
    Enforcing




* https://github.com/lxc/lxd/issues


 

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


LXD NVIDIA GPU OpenGL ?
---------------------------

* https://discuss.linuxcontainers.org/t/nvidia-opengl-in-the-container/4410




EOU
}
