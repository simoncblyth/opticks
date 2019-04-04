vbx-source(){ echo $BASH_SOURCE ; }
vbx-vi(){ vi $(vbx-source)  ; }
vbx-env(){  olocal- ; opticks- ; }
vbx-usage(){ cat << EOU

vbx : Virtual Box for testing Opticks on other Linux distros ?
================================================================================================

https://www.virtualbox.org/wiki/Linux_Downloads

Users of Oracle Linux / RHEL can add  the Oracle Linux repo file to /etc/yum.repos.d/. 


Objective
-----------

User Elias is reporting runtime nnode related SEGV issues 
with Opticks on Ubuntu 16.0.4 
with gcc 5.4.0 (?) at least CMake says 
"The CXX compiler identification is GNU 5.4.0" 

It would be good for me to reproduce the problem in such a system

* https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/
* https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/16.04.6/ubuntu-16.04.6-desktop-amd64.iso

But lets started with the latest Ubuntu 18.04.2 


April 4, 2019
--------------

Succeeded to install VirtualBox-6.0 onto precision workstation with::

   vbx-get-repo
   vbx-install

Installation message::

   creating group 'vboxusers'. VM users must be member of that group!

After this "Applications > System Tools > Oracle VM VirtualBox" is available


Download Ubuntu ISO from https://www.ubuntu.com/desktop 
 ~/Downloads/ubuntu-18.04.2-desktop-amd64.iso


Virtualbox tips
~~~~~~~~~~~~~~~~~~

* There is no "host" button but the right control button "rctrl" takes its place
* host+C switches between scaled and windowed mode
* host+home to access virtualbox menus, which otherwise have somehow disappeared 



Getting Ubuntu 18.04 kitted out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.tecmint.com/install-virtualbox-guest-additions-in-ubuntu/

::

   sudo apt install gcc curl mercurial git vim cmake


.bashrc
~~~~~~~~

Append to tail of .bashrc:: 

    export LC_ALL=en_US.UTF-8

    vip(){ vim $HOME/.bashrc ; }
    ini(){ source $HOME/.bashrc ; }

    export OPTICKS_HOME=$HOME/opticks
    export LOCAL_BASE=/usr/local
    opticks-(){ . $OPTICKS_HOME/opticks.bash && opticks-env $* && opticks-export ; }

    export PYTHONPATH=$HOME
    export PATH=$LOCAL_BASE/opticks/lib:$OPTICKS_HOME/bin:$OPTICKS_HOME/ana:$PATH


    o(){ opticks- ; cd $(opticks-home) ; hg st ; }
    t(){ type $* ; }

    opticks-




Guest Additions not showing up : so no copy/paste : so try to ssh in 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Did the below in the hope of getting copy/paste to work 
from host to virtualbox.  Didnt work, but inevitably many 
of the installed packages are being used.

::

   sudo apt update
   sudo apt upgrade
   sudo apt install build-essential dkms linux-headers-$(uname -r)


::

    sudo apt install net-tools       # want to ssh into the virtualbox ubuntu, need ifconfig for address
    sudo apt install openssh-server  # maybe not needed


Virtualbox : Settings > Network > Advanced > Port Forwarding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forward port 2222 of the host to port 22 in the virtualbox 
which has IP 10.0.2.15 (obtained by running ifconfig in the Ubuntu vbx)

::

    Name           Protocol   Host IP       Host Port    Guest IP      Guest Port 
    Rule 1         TCP        127.0.0.1     2222         10.0.2.15     22


Now can ssh in from the host with::

   ssh 127.0.0.1 -p 2222


Maybe that easier by adding to ~/.ssh/config::

    host V
        user blyth
        hostname 127.0.0.1
        port 2222

Avoid passwords using by copying host public key into the vbx:: 

    ssh--putkey V

Now can get in with "ssh V"::

    [blyth@localhost ~]$ ssh V
    Welcome to Ubuntu 18.04.2 LTS (GNU/Linux 4.18.0-17-generic x86_64)
    ...
    Your Hardware Enablement Stack (HWE) is supported until April 2023.
    Last login: Thu Apr  4 14:26:20 2019 from 10.0.2.2
    blyth@blyth-VirtualBox:~$ 

    # gcc version 7.3.0 (Ubuntu 7.3.0-27ubuntu1~18.04)


Opticks preliminaries
~~~~~~~~~~~~~~~~~~~~~~~~

::

    sudo mkdir /usr/local/opticks
    sudo chown blyth /usr/local/opticks

::

    apt-cache search libboost
    apt show libboost-all-dev

    sudo apt install libboost-all-dev


Partial install of Opticks into virtualbox Ubuntu on DELL Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No hope of a full install inside the virtualbox (as CUDA 
access to GPU is beyond virtualization?) but even so 
a partial Ubuntu install may help with understanding 
user SEGV in npy- for example.


Externals::

   bcm-; bcm--
   plog-; plog--

Subs::

   cd okconf   # removed the REQIRED in FindOpticksCUDA.cmake
   om-install

   cd ../sysrap  
   om-install    

   ## fails for lack of openssl/md5.h
   sudo apt install libssl-dev
   
   om-install  ## completes after installing libssl-dev   
   om-test     ## one failing test, SOKConfTest : expected FAIL as lack CUDA, OptiX, Geant4

   cd ../boostrap
   om-install         ## finds Boost 1.65.1
   om-test            ## all tests pass
  
   cd ../npy       

   om-install        ## fails, missing GLM
   glm-;glm--
   om-install        ## fails, missing YoctoGL
   oyoctogl-;oyoctogl--         ## lots of sign compare warnings 

   om-install        ## fails, missing OpenMesh
   openmesh-;openmesh--
   
   ## compilation fails, gcc 7.3 requires <sys/time.h> header in OpenMesh-6.3/src/OpenMesh/Tools/Utils/conio.cc see openmesh-vi
   ## manually changed the external

   cd ~/opticks/npy
   om-install

   om-test   ## 1/117 fails   NLoadTest from lack of opticksdata gensteps

   om-subs   ## check the order

   cd ../yoctoglrap
   om-install 
   om-test   ## all pass

   cd ../optickscore
   om-install
   
   ## fixed some OpticksCfg string truncation warnings

   ## got the usual OpticksBufferSpec error, when no OptiX 
   ## added a DUMMY to allow to compile without OptiX a few more subs

   cd ../ggeo
   om-install
   om-test    ## notes/issues/ggeo-fails-virtualbox-ubuntu_18_04_2.rst   22/50 resource related fails

   assimp-;assimp--


Want to commit changes to bitbucket
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ssh-keygen 
2. add the public key to my bitbucket account webinterface
3. "hg commit" failed for lack of username, so copy over the config from host::

   scp .hgrc V:






EOU
}


vbx-get-repo()
{
    local msg="=== $FUNCNAME :"
    cd /etc/yum.repos.d/
    local url=https://download.virtualbox.org/virtualbox/rpm/el/virtualbox.repo
    local repo=$(basename $url)

    [ -f "$repo" ] && echo $msg repo $repo exists already && return  

    local cmd="sudo curl -L -O $url" 
    echo $msg $cmd
    eval $cmd
}

vbx-install()
{
    local cmd="sudo yum install VirtualBox-6.0"
    echo $msg $cmd
    eval $cmd
}


