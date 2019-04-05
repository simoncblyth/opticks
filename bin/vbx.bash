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


Containers Refs
------------------

* https://blog.risingstack.com/operating-system-containers-vs-application-containers/

* https://hub.docker.com/_/ubuntu



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
* during Ubuntu install, if some buttons are off screen simply drag the window to make them visible 


Virtualbox CUDA ? Maybe possible with new Linux kernels compiled with IOMMU support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://superuser.com/questions/1388815/use-host-cuda-from-virtualbox

* :google:`virtualbox pcipassthrough`
* https://www.virtualbox.org/manual/ch09.html#pcipassthrough

  * looks to be very experimental currently 

* :google:`linux NVIDIA GPU virtualbox pcipassthrough`



Basic Tools
~~~~~~~~~~~~~~~

::

   sudo apt install openssh-server gcc curl mercurial git vim cmake


.bashrc
~~~~~~~~

Append to tail of .bashrc:: 


Ubuntu 18.04 : Guest Additions not showing up : so no copy/paste : so try to ssh in 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.tecmint.com/install-virtualbox-guest-additions-in-ubuntu/

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

Make that easier by adding to ~/.ssh/config::

    host V
        user blyth
        hostname 127.0.0.1
        port 2222

Avoid passwords by copying host public key into the vbx (from env-):: 

    ssh-;ssh--putkey V

Now can get in with "ssh V"::

    [blyth@localhost ~]$ ssh V
    Welcome to Ubuntu 18.04.2 LTS (GNU/Linux 4.18.0-17-generic x86_64)
    ...
    Your Hardware Enablement Stack (HWE) is supported until April 2023.
    Last login: Thu Apr  4 14:26:20 2019 from 10.0.2.2
    blyth@blyth-VirtualBox:~$ 

    # gcc version 7.3.0 (Ubuntu 7.3.0-27ubuntu1~18.04)


Ubuntu 16 : cannot ssh in ? FIXED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initially with the same network port forwarding as above are unable to ssh in,
got "connection reset by peer". Succeed to ssh in after install openssh-server::

    sudo apt update
    sudo apt upgrade
    sudo apt install openssh-server
    ## remove old know_hosts line

::

    blyth@localhost Downloads]$ ssh V
    Welcome to Ubuntu 16.04.6 LTS (GNU/Linux 4.15.0-45-generic x86_64)
    ...
    New release '18.04.2 LTS' available.
    blyth@blyth-VirtualBox:~$ 




Opticks preliminaries
~~~~~~~~~~~~~~~~~~~~~~~~

::

    sudo mkdir /usr/local/opticks
    sudo chown blyth /usr/local/opticks
    mkdir /usr/local/opticks/build

::

    apt-cache search libboost
    apt show libboost-all-dev
    sudo apt install libboost-all-dev


Get Opticks
~~~~~~~~~~~~~

Want to be able to push, so using ssh.

1. ssh-keygen
2. copy/paste public key into bitbucket web interface

::

   hg clone ssh://hg@bitbucket.org/simoncblyth/opticks 


Hookup Opticks to .bashrc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   vbx-bashrc- >> .bashrc

From host 
~~~~~~~~~~

::

    scp .vimrc V:
    scp .hgrc  V:



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
   om-install         ## finds Boost 1.65.1 (Ubu18)   1.58.0 (on Ubu16) 
   om-test            ## all tests pass
  
   cd ../npy       

   ## install the externals mandatory for npy
   glm-;glm--
   oyoctogl-;oyoctogl--         ## lots of sign compare warnings 
   openmesh-;openmesh--

   om-install        
   
   ## With Ubuntu 18 
   ##    1. OpenMesh compilation fails, 
   ##       gcc 7.3 requires <sys/time.h> header in OpenMesh-6.3/src/OpenMesh/Tools/Utils/conio.cc see openmesh-vi
   ##       manually changed the external
   ##
   ## With Ubuntu 16 
   ##    1. fails to configure npy as stock CMake 3.5 is not new enough, 3.8+ is needed
   ##       gets further after installing newer CMake with ocmake-;ocmake--
   ##    2. the failure of oyoctogl-- to complete prevents npy configure, had to manually 
   ##       change stb_image.h for the gcc, see oyoctogl-
   ##

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
   cd ~/opticks/assimprap
   om-install
   om-test    ## all fail with resource problems, same as the GGeo fails

   cd ../openmeshrap/
   om-install 
   om-test    ## passes


   cd ../opticksgeo
   om-install
   om-test     ## fails similar to GGeo

   om-subs   

   ## next one is cudarap, so this as far as it goes inside the virtualbox
   ##  without major efforts to get OpenGL going without CUDA+OptiX 



Want to commit changes to bitbucket
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"hg commit" failed for lack of username, so copy over the config from host::

   scp .hgrc V:



Ubuntu 16 : cmake 3.5.1 fails to configure npy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    - Configuring NPY
    CMake Error at /usr/share/cmake-3.5/Modules/CMakeFindDependencyMacro.cmake:45 (message):
      Invalid arguments to find_dependency
    Call Stack (most recent call first):
      /usr/local/opticks/lib/cmake/boostrap/boostrap-config.cmake:11 (find_dependency)
      CMakeLists.txt:14 (find_package)

/usr/local/opticks/lib/cmake/boostrap/boostrap-config.cmake::

    # Library: Boost::system
    find_dependency(Boost REQUIRED COMPONENTS system;program_options;filesystem;regex)

Problem is that COMPONENTS is not supported until 3.8.0 according to 

* https://github.com/pabloariasal/modern-cmake-sample/issues/5



Ubuntu 16 : getting a newer CMake
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* added externals/ocmake.bash script for getting a newer cmake  



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




vbx-bashrc-(){ 

cat << EOH
###
### Start $FUNCNAME $(date)
###
EOH

cat << 'EOB'

export LC_ALL=en_US.UTF-8

vip(){ vim $HOME/.bashrc ; }
ini(){ source $HOME/.bashrc ; }

export OPTICKS_HOME=$HOME/opticks
export LOCAL_BASE=/usr/local
opticks-(){ . $OPTICKS_HOME/opticks.bash && opticks-env $* && opticks-export ; }

export PYTHONPATH=$HOME
export PATH=$LOCAL_BASE/opticks/lib:$OPTICKS_HOME/bin:$OPTICKS_HOME/ana:$PATH

o(){ opticks- ; cd $(opticks-home) ; hg st ; }
on(){ cd $OPTICKS_HOME/notes/issues ; }
t(){ type $* ; }

opticks-

EOB

cat << EOT
###
### End $FUNCNAME $(date)
###
EOT

}



