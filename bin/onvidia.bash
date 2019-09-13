##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

onvidia-source(){ echo $BASH_SOURCE ; }
onvidia-vi(){ vi $(onvidia-source)  ; }
onvidia-env(){  olocal- ; opticks- ; }
onvidia-driver-version(){  nvidia-smi --query-gpu=driver_version --format=csv,noheader | uniq ; }
onvidia-export(){ export OPTICKS_NVIDIA_DRIVER_VERSION=$(onvidia-driver-version) ; }

onvidia-usage(){ cat << EOU

NVIDIA Linux : notes on drivers setup etc..
=============================================

Overview
----------

Note that the CUDA runfile and the NVIDIA driver runfile 
both contain an NVIDIA display driver, typically installed by the 
CUDA runfile is older, currently 418.39 with CUDA 10.1 
vs 418.56 on its own.

::

    [blyth@localhost ~]$ l ~/Downloads/   ## new drivers
    total 6046500
    -rw-r--r--. 1 blyth blyth 2423314285 Apr  8 14:24 cuda_10.1.105_418.39_linux.run
    -rw-rw-r--. 1 blyth blyth  107195640 Apr  8 14:15 NVIDIA-Linux-x86_64-418.56.run

    [root@localhost ~]# ls -l /root/cuda*   ## old drivers
    -rw-r--r--. 1 root root   72871665 Jul  5  2018 /root/cuda_9.2.88.1_linux
    -rw-r--r--. 1 root root 1758421686 Jul  5  2018 /root/cuda_9.2.88_396.26_linux    


OptiX 6.0.0 release notes : needs 418.30 or later on Linux
------------------------------------------------------------

::

    Graphics Driver: 
    Windows: driver version 418.81 or later is required.  
    Linux:   driver version 418.30 or later is required.  
    Windows 7/8.1/10 64-bit; 
    Linux RHEL 4.8+ or Ubuntu 10.10+ 64-bit 


NVIDIA Linux CUDA documentatation
------------------------------------

* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html


Try to downgrade to the CUDA 10.1 driver
-----------------------------------------

From laptop ssh into the GPU machine, for copy/paste-ability::

    epsilon:opticks blyth$ ssh I
    Last login: Tue Apr  9 22:04:40 2019
    [root@localhost ~]# 
    [root@localhost ~]# systemctl isolate multi-user.target    
    ## immediately Desktop GUI processes get killed, and is dropped to console

Verify that the GPU is not used::

    [root@localhost ~]# ps aux | grep X
    root      13024  0.0  0.0 225724  4824 ?        Ss   Apr11   0:00 /usr/bin/abrt-watch-log -F Backtrace /var/log/Xorg.0.log -- /usr/bin/abrt-dump-xorg -xD
    root      89798  0.0  0.0 112712   964 pts/0    S+   14:44   0:00 grep --color=auto X
    [root@localhost ~]# nvidia-smi 
    Sun Apr 14 14:45:07 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN RTX           Off  | 00000000:73:00.0 Off |                  N/A |
    | 21%   35C    P0     1W / 280W |      0MiB / 24189MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    [root@localhost ~]# 


::

    [root@localhost ~]# cd /home/blyth/Downloads/
    [root@localhost Downloads]# ls -l 
    total 6046500
    -rw-r--r--. 1 blyth blyth 2423314285 Apr  8 14:24 cuda_10.1.105_418.39_linux.run
    -rw-rw-r--. 1 blyth blyth  107195640 Apr  8 14:15 NVIDIA-Linux-x86_64-418.56.run

    [root@localhost Downloads]# sh cuda_10.1.105_418.39_linux.run   ## in the curses interface, select only the driver
    ===========
    = Summary =
    ===========

    Driver:   Installed
    Toolkit:  Not Selected
    Samples:  Not Selected

    To uninstall the NVIDIA Driver, run nvidia-uninstall
    Logfile is /var/log/cuda-installer.log
    [root@localhost Downloads]# 

    From that log::

        [INFO]: Install NVIDIA's 32-bit compatibility libraries? (Answer: Yes)
        [INFO]: Will install GLVND GLX client libraries.
        [INFO]: Will install GLVND EGL client libraries.
        [INFO]: Skipping GLX non-GLVND file: "libGL.so.418.39"
        [INFO]: Skipping GLX non-GLVND file: "libGL.so.1"
        [INFO]: Skipping GLX non-GLVND file: "libGL.so"
        [INFO]: Skipping EGL non-GLVND file: "libEGL.so.418.39"
        [INFO]: Skipping EGL non-GLVND file: "libEGL.so"
        [INFO]: Skipping EGL non-GLVND file: "libEGL.so.1"
        [INFO]: Skipping GLX non-GLVND file: "./32/libGL.so.418.39"
        [INFO]: Skipping GLX non-GLVND file: "libGL.so.1"
        [INFO]: Skipping GLX non-GLVND file: "libGL.so"
        [INFO]: Skipping EGL non-GLVND file: "./32/libEGL.so.418.39"
        [INFO]: Skipping EGL non-GLVND file: "libEGL.so"
        [INFO]: Skipping EGL non-GLVND file: "libEGL.so.1"
        [INFO]: Uninstalling the previous installation with /usr/bin/nvidia-uninstall.


    [root@localhost Downloads]# nvidia-smi
    Sun Apr 14 14:54:51 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.39       Driver Version: 418.39       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN RTX           Off  | 00000000:73:00.0 Off |                  N/A |
    | 22%   41C    P0    65W / 280W |      0MiB / 24189MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    [root@localhost Downloads]# 

    [root@localhost Downloads]# systemctl isolate runlevel5   ## back to GUI : throws up the login menu
    
    ## same problem 


    Try old driver

    accept/decline/quit: accept

    Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 396.26?
    (y)es/(n)o/(q)uit: y

    Do you want to install the OpenGL libraries?
    (y)es/(n)o/(q)uit [ default is yes ]: y

    Do you want to run nvidia-xconfig?
    This will update the system X configuration file so that the NVIDIA X driver
    is used. The pre-existing X configuration file will be backed up.
    This option should not be used on systems that require a custom
    X configuration, such as systems with multiple GPU vendors.
    (y)es/(n)o/(q)uit [ default is no ]: 

    Installing the NVIDIA display driver...
    The driver installation is unable to locate the kernel source. Please make sure that the kernel source packages are installed and set up correctly.
    If you know that the kernel source packages are installed and set up correctly, you may pass the location of the kernel source with the '--kernel-source-path' flag.

    ===========
    = Summary =
    ===========

    Driver:   Installation Failed
    Toolkit:  Not Selected
    Samples:  Not Selected


    Logfile is /tmp/cuda_install_100422.log


::

    [254264.910720] nvidia-modeset: Unloading
    [254264.919620] nvidia-uvm: Unloaded the UVM driver in 8 mode
    [254264.927519] nvidia-nvlink: Unregistered the Nvlink Core, major device number 237
    -> Install NVIDIA's 32-bit compatibility libraries? (Answer: Yes)
    -> Will install GLVND GLX client libraries.
    -> Will install GLVND EGL client libraries.
    -> Skipping GLX non-GLVND file: "libGL.so.418.56"
    -> Skipping GLX non-GLVND file: "libGL.so.1"
    -> Skipping GLX non-GLVND file: "libGL.so"
    -> Skipping EGL non-GLVND file: "libEGL.so.418.56"
    -> Skipping EGL non-GLVND file: "libEGL.so"
    -> Skipping EGL non-GLVND file: "libEGL.so.1"
    -> Skipping GLX non-GLVND file: "./32/libGL.so.418.56"
    -> Skipping GLX non-GLVND file: "libGL.so.1"
    -> Skipping GLX non-GLVND file: "libGL.so"
    -> Skipping EGL non-GLVND file: "./32/libEGL.so.418.56"
    -> Skipping EGL non-GLVND file: "libEGL.so"
    -> Skipping EGL non-GLVND file: "libEGL.so.1"
    Looking for install checker script at ./libglvnd_install_checker/check-libglvnd-install.sh
       executing: '/bin/sh ./libglvnd_install_checker/check-libglvnd-install.sh'...
       Checking for libglvnd installation.
       Checking libGLdispatch...
       Checking libGLdispatch dispatch table
       Checking call through libGLdispatch
       All OK
       libGLdispatch is OK
       Checking for libGLX
       libGLX is OK
       Checking for libEGL
       libEGL is OK
       Checking entrypoint library libOpenGL.so.0
       Checking call through libGLdispatch
       Checking call through library libOpenGL.so.0
       All OK
       Entrypoint library libOpenGL.so.0 is OK
       Checking entrypoint library libGL.so.1
       Checking call through libGLdispatch
       Checking call through library libGL.so.1
       All OK
       Entrypoint library libGL.so.1 is OK

       Found libglvnd libraries: libGL.so.1 libOpenGL.so.0 libEGL.so.1 libGLX.so.0 libGLdispatch.so.0
       Missing libglvnd libraries:

       libglvnd appears to be installed.
    Will not install libglvnd libraries.
    -> Skipping GLVND file: "libOpenGL.so.0"
    -> Skipping GLVND file: "libOpenGL.so"
    -> Skipping GLVND file: "libGLESv1_CM.so.1.2.0"
    -> Skipping GLVND file: "libGLESv1_CM.so.1"



    /var/log/nvidia-installer.log


nvidia-persistenced
---------------------

Minimum times of GPU using tests:

oxrap
   around 0.5s
thrap
   around 0.3s
cudarap
   around 0.3s

Tried starting nvidia-persistenced to see if the minimum time would go down, 
but dont see any difference with Titan V, CUDA 10.1, driver 418.56 and OptiX 5.1.0



NVIDIA Linux : User written guides for various distros
--------------------------------------------------------

* https://www.dedoimedo.com/computers/centos-7-nvidia.html
* https://ingowald.blog/installing-the-latest-nvidia-driver-cuda-and-optix-on-linux-ubuntu-18-04/
* https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07
* https://wiki.gentoo.org/wiki/NVIDIA/nvidia-drivers


NVIDIA forum
----------------

* https://devtalk.nvidia.com/default/board/98/linux/
* https://devtalk.nvidia.com/default/topic/522835/linux/if-you-have-a-problem-please-read-this-first/

  advises using /bin/nvidia-bug-report.sh to prepare report 


OptiX Forum
--------------

* https://devtalk.nvidia.com/default/topic/1047061/optix/optix-6-0-release/

OptiX 6.0 is now available.

* Download: https://developer.nvidia.com/designworks/optix/download
* Release Notes: https://developer.nvidia.com/designworks/optix/downloads/6.0.0/releasenotes
* Documentation: http://raytracing-docs.nvidia.com/optix/index.html

Note this release of OptiX requires a display driver version 418 or higher. For
linux, you may need to look in Beta drivers.

Drivers: https://www.nvidia.com/Download/index.aspx



* https://devtalk.nvidia.com/default/topic/1049543/optix/-resolved-ubuntu-a-supported-nvidia-gpu-could-not-be-found/

The 418.30 driver added OptiX 6.0.0 support under Linux, but motion blur was broken in that specific build.
There is a 418.56 driver listed as newest Linux drivers. Maybe try that instead.


Linux Driver Version History
-----------------------------

* https://www.nvidia.com/object/linux-amd64-display-archive.html

::

    Linux x64 (AMD64/EM64T) Display Driver

    Version: 418.56
    Operating System: Linux 64-bit
    Release Date: March 20, 2019
    Linux x64 (AMD64/EM64T) Display Driver

    Version: 418.43
    Operating System: Linux 64-bit
    Release Date: February 22, 2019
    Linux x64 (AMD64/EM64T) Display Driver

    Version: 410.104
    Operating System: Linux 64-bit
    Release Date: February 22, 2019
    Linux x64 (AMD64/EM64T) Display Driver

    Version: 390.116
    Operating System: Linux 64-bit
    Release Date: February 22, 2019
    Linux x64 (AMD64/EM64T) Display Driver

    Version: 418.30
    Operating System: Linux 64-bit
    Release Date: January 30, 2019


ELrepo ?
---------

* http://elrepo.org/tiki/Packages#N

Nice idea in principal (centralised problem solving), 
but how fast are they at updating for the latest drivers.


Brief History of Driver/Kernel Update issue with Xserver startup SEGV 
--------------------------------------------------------------------------

Fri 5 April 2019
    while attempting to install LXD linux containers using snap (canonical) 
    onto CentOS did a "yum update", in the hope of resolving an issue of 
    SELinux breaking the installation of snap and/or LXD.  
    This (to my later surprise) updated the kernel to 3.10.0-957.10.1.el7.x86_64
    but no reboot was done so the old kernel continued to run normally.

Sun 8 April
    happened to noticed that the /etc/centos-release was inconsistent with uname -r,
    suggesting that the kernel has been updated against my wishes (I wrongly thought
    that "yum upgrade" would so that but not "yum update" : in fact thet will both 
    do that : when upstream updates the kernel package). 

    Performed a reboot and found the GUI to be low resolution. /var/log/Xorg.0.log
    includes "Display was unrecognized" and has errors::

    nvidia-smi fails to communicate with driver and opticks-t has many fails::

        [blyth@localhost issues]$ nvidia-smi
        NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. 
        Make sure that the latest NVIDIA driver is installed and running.

    For details see notes/issues/nvidia-driver-fails-following-yum-update.rst

    Rebooting again using the GRUB menu to pick the 
    2nd entry kernel, 3.10.0-862.6.3.el7.x86_64, with this the display returned
    to the normal full resolution.  nvidia-smi worked ok but opticks-t shows 3/275
    FAILS for the OpenGL using :  OKTest, OKG4Test, OTracerTest all with 
    "GLX: Failed to create context: GLXBadFBConfig" 

    Keep copies of the two logs::

       [blyth@localhost ~]$ mkdir ~/Xorg
       [blyth@localhost ~]$ cp /var/log/Xorg.0.log ~/Xorg/Xorg.0.log-$(uname -r)
       [blyth@localhost log]$ cp Xorg.0.log.old ~/Xorg/   ## prior log, new kernel+old driver 

    Examining Xorg.0.log.old (new kernel 3.10.0-957.10.1.el7.x86_64 with the old driver)
    shows that it loaded nouveau and got loads of errors (nouveau doensnt support TITAN V)
    resulting in the black screen.

    Examining Xorg.0.log-3.10.0-862.6.3.el7.x86_64 shows that GLX errors::

        [blyth@localhost ~]$ grep \(EE\) ~/Xorg/Xorg.0.log-$(uname -r)
            (WW) warning, (EE) error, (NI) not implemented, (??) unknown.
        [    47.484] (EE) open /dev/fb0: No such file or directory
        [    47.485] (EE) NVIDIA(0): Failed to initialize the GLX module; please check in your X
        [    47.485] (EE) NVIDIA(0):     log file that the GLX module has been loaded in your X
        [    47.485] (EE) NVIDIA(0):     server, and that the module is the NVIDIA GLX module.  If
        [    47.485] (EE) NVIDIA(0):     you continue to encounter problems, Please try
        [    47.485] (EE) NVIDIA(0):     reinstalling the NVIDIA driver.

 

Mon 9 April 
    Attempt to get the new kernel to work by installing newer nvidia 
    drivers, but find both 418.39 from cuda-10.1 runfile and 418.56 from 
    NVIDIA_Linux runfile fail with SEGV crashes of the Xserver : the boot never
    gets thru to the GUI.  Have to recover by re-installing an old driver.

    Notice that the nouveau driver is mentioned in /var/log/Xorg.0.log
    so take several steps to absolutely avoid loading of nouveau module
    by changing grub kernel command line (see below), and recreating initramfs img with "dracut --force",
    for details see below onvidia-disabling-nouveau.

    After avoiding nouveau the Xserver still SEGV but this time a stack trace
    is provided in /var/log/Xorg.0.log that points to /usr/lib64/xorg/modules/extensions/libglx.so::

        [    49.488] (EE) Backtrace:
        [    49.489] (EE) 0: /usr/bin/X (xorg_backtrace+0x55) [0x555bf8b8c0f5]
        [    49.489] (EE) 1: /usr/bin/X (0x555bf89db000+0x1b4d79) [0x555bf8b8fd79]
        [    49.489] (EE) 2: /lib64/libpthread.so.0 (0x7f23d8ddd000+0xf5d0) [0x7f23d8dec5d0]
        [    49.489] (EE) 3: /usr/lib64/xorg/modules/extensions/libglx.so (0x7f23d60e1000+0x29341) [0x7f23d610a341]
        [    49.489] (EE) 4: /usr/lib64/xorg/modules/extensions/libglx.so (0x7f23d60e1000+0x28772) [0x7f23d6109772]
        [    49.489] (EE) 5: /usr/lib64/xorg/modules/extensions/libglx.so (0x7f23d60e1000+0x27b5a) [0x7f23d6108b5a]
        [    49.489] (EE) 6: /usr/bin/X (InitExtensions+0x5d) [0x555bf8aa369d]
        [    49.489] (EE) 7: /usr/bin/X (0x555bf89db000+0x601d6) [0x555bf8a3b1d6]
        [    49.489] (EE) 8: /lib64/libc.so.6 (__libc_start_main+0xf5) [0x7f23d8a323d5]
        [    49.489] (EE) 9: /usr/bin/X (0x555bf89db000+0x4a4ce) [0x555bf8a254ce]
        [    49.489] (EE)
        [    49.489] (EE) Segmentation fault at address 0x7f23d442b948
        [    49.489] (EE)
        Fatal server error:
     
    Using that clue enables google searches to find others with the same issue

    * :google:`nvidia centos SEGV /usr/lib64/xorg/modules/extensions/libglx.so`


First try : installing new display and CUDA drivers using Console, as GUI cannot be running
----------------------------------------------------------------------------------------------

::

    ctrl+alt+f2 # from login menu or elsewhere, gets to Console

    init 3  ##  runlevel3 : non graphical console, with networking working
            ##  old way with init still works, probably is translated to "systemctl ..."
            ##  yep see "man telinit" "man init"

    ran nvidia driver.run first 

    then cuda....run   (418.39)  included

    init 5    ## back to GUI runlevel5

    ## problem in x server : Xorg server crashed with SEGV

    ## in the Xorg logfile the nouvea module was being loaded as well as nvidia


this did not fully get rid of nouveau
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the failing case nouveau did not get unloaded::

    [blyth@localhost Xorg]$ grep nouveau *
    Xorg.0.log-3.10.0-862.6.3.el7.x86_64:[    47.452] (==) Matched nouveau as autoconfigured driver 1
    Xorg.0.log-3.10.0-862.6.3.el7.x86_64:[    47.462] (II) LoadModule: "nouveau"
    Xorg.0.log-3.10.0-862.6.3.el7.x86_64:[    47.462] (II) Loading /usr/lib64/xorg/modules/drivers/nouveau_drv.so
    Xorg.0.log-3.10.0-862.6.3.el7.x86_64:[    47.463] (II) Module nouveau: vendor="X.Org Foundation"
    Xorg.0.log-3.10.0-862.6.3.el7.x86_64:[    49.292] (II) UnloadModule: "nouveau"
    Xorg.0.log-3.10.0-862.6.3.el7.x86_64:[    49.292] (II) Unloading nouveau
    Xorg.0.log.old:[    47.145] (==) Matched nouveau as autoconfigured driver 0
    Xorg.0.log.old:[    47.146] (II) LoadModule: "nouveau"
    Xorg.0.log.old:[    47.146] (II) Loading /usr/lib64/xorg/modules/drivers/nouveau_drv.so
    Xorg.0.log.old:[    47.148] (II) Module nouveau: vendor="X.Org Foundation"
    [blyth@localhost Xorg]$ 


unknown display, low resolution  : due to incompatible driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.linuxquestions.org/questions/centos-111/cannot-change-screen-resolution-on-unknown-display-in-centos7-4175583712/



After absolutely eradicating nouveau : X still cannot start : BUT get a backtrace in log
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


After absolutely eradicating nouveau still cannot start X, but this 
time get a stack trace in the log pointing at /usr/lib64/xorg/modules/extensions/libglx.so

/var/log/Xorg.0.log::

    [    47.171]
    X.Org X Server 1.20.1
    X Protocol Version 11, Revision 0
    [    47.171] Build Operating System:  3.10.0-957.1.3.el7.x86_64
    [    47.171] Current Operating System: Linux localhost.localdomain 3.10.0-957.10.1.el7.x86_64 #1 SMP Mon Mar 18 15:06:45 UTC 2019 x86_64
    [    47.171] Kernel command line: BOOT_IMAGE=/vmlinuz-3.10.0-957.10.1.el7.x86_64 root=/dev/mapper/cl-root ro rd.lvm.lv=cl/root rd.lvm.lv=cl/swap rd.driver.blacklist=nouveau modprobe.blacklist=nouveau nouveau.modeset=0
    [    47.171] Build Date: 14 March 2019  10:37:36AM
    [    47.171] Build ID: xorg-x11-server 1.20.1-5.3.el7_6
    [    47.171] Current version of pixman: 0.34.0
    [    47.171]    Before reporting problems, check http://wiki.x.org
        to make sure that you have the latest version.

    ...

    [    47.225] (II) LoadModule: "glx"
    [    47.225] (II) Loading /usr/lib64/xorg/modules/extensions/libglx.so
    [    47.230] (II) Module glx: vendor="X.Org Foundation"
    [    47.230]    compiled for 1.19.5, module version = 1.0.0
                          ******************* INCONSISTENT VERSION : SHOULD BE 1.20.1 ************* 
    [    47.230]    ABI class: X.Org Server Extension, version 10.0
    [    47.230] (II) LoadModule: "nvidia"
    [    47.230] (II) Loading /usr/lib64/xorg/modules/drivers/nvidia_drv.so
    [    47.237] (II) Module nvidia: vendor="NVIDIA Corporation"
    [    47.237]    compiled for 4.0.2, module version = 1.0.0
    [    47.237]    Module class: X.Org Video Driver
    [    47.238] (II) NVIDIA dlloader X Driver  418.56  Fri Mar 15 12:37:35 CDT 2019
    [    47.238] (II) NVIDIA Unified Driver for all Supported NVIDIA GPUs
    [    47.238] (++) using VT number 1

    [    47.248] (II) Loading sub module "fb"
    [    47.248] (II) LoadModule: "fb"
    [    47.249] (II) Loading /usr/lib64/xorg/modules/libfb.so
    [    47.250] (II) Module fb: vendor="X.Org Foundation"
    [    47.250]    compiled for 1.20.1, module version = 1.0.0
                   **********************   
    [    47.250]    ABI class: X.Org ANSI C Emulation, version 0.4
    [    47.250] (II) Loading sub module "wfb"
    [    47.250] (II) LoadModule: "wfb"
    [    47.250] (II) Loading /usr/lib64/xorg/modules/libwfb.so
    [    47.250] (II) Module wfb: vendor="X.Org Foundation"
    [    47.250]    compiled for 1.20.1, module version = 1.0.0
                   **********************   
    [    47.250]    ABI class: X.Org ANSI C Emulation, version 0.4

    ...

    [    49.343] (II) Initializing extension DPMS
    [    49.343] (II) Initializing extension Present
    [    49.344] (II) Initializing extension DRI3
    [    49.344] (II) Initializing extension X-Resource
    [    49.344] (II) Initializing extension XVideo
    [    49.344] (II) Initializing extension XVideo-MotionCompensation
    [    49.344] (II) Initializing extension SELinux
    [    49.345] (II) SELinux: Disabled by boolean
    [    49.345] (II) Initializing extension GLX
    [    49.346] (II) Initializing extension GLX
    [    49.346] (II) Initializing extension XFree86-VidModeExtension
    [    49.346] (II) Initializing extension XFree86-DGA
    [    49.346] (II) Initializing extension XFree86-DRI
    [    49.346] (II) Initializing extension DRI2
    [    49.346] (II) Initializing extension GLX
    [    49.346] (II) AIGLX: Screen 0 is not DRI2 capable
    [    49.347] (EE) AIGLX: reverting to software rendering
    [    49.487] (II) IGLX: enabled GLX_MESA_copy_sub_buffer
    [    49.488] (EE)
    [    49.488] (EE) Backtrace:
    [    49.489] (EE) 0: /usr/bin/X (xorg_backtrace+0x55) [0x555bf8b8c0f5]
    [    49.489] (EE) 1: /usr/bin/X (0x555bf89db000+0x1b4d79) [0x555bf8b8fd79]
    [    49.489] (EE) 2: /lib64/libpthread.so.0 (0x7f23d8ddd000+0xf5d0) [0x7f23d8dec5d0]
    [    49.489] (EE) 3: /usr/lib64/xorg/modules/extensions/libglx.so (0x7f23d60e1000+0x29341) [0x7f23d610a341]
    [    49.489] (EE) 4: /usr/lib64/xorg/modules/extensions/libglx.so (0x7f23d60e1000+0x28772) [0x7f23d6109772]
    [    49.489] (EE) 5: /usr/lib64/xorg/modules/extensions/libglx.so (0x7f23d60e1000+0x27b5a) [0x7f23d6108b5a]
    [    49.489] (EE) 6: /usr/bin/X (InitExtensions+0x5d) [0x555bf8aa369d]
    [    49.489] (EE) 7: /usr/bin/X (0x555bf89db000+0x601d6) [0x555bf8a3b1d6]
    [    49.489] (EE) 8: /lib64/libc.so.6 (__libc_start_main+0xf5) [0x7f23d8a323d5]
    [    49.489] (EE) 9: /usr/bin/X (0x555bf89db000+0x4a4ce) [0x555bf8a254ce]
    [    49.489] (EE)
    [    49.489] (EE) Segmentation fault at address 0x7f23d442b948
    [    49.489] (EE)
    Fatal server error:
    [    49.489] (EE) Caught signal 11 (Segmentation fault). Server aborting
    [    49.490] (EE)
    [    49.490] (EE)
    Please consult the The X.Org Foundation support
             at http://wiki.x.org
     for help.
    [    49.490] (EE) Please also check the log file at "/var/log/Xorg.0.log" for additional information.
    [    49.490] (EE)
    [    49.729] (EE) Server terminated with error (1). Closing log file.



backtrace points finger at culprit /usr/lib64/xorg/modules/extensions/libglx.so
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Which is coming from xorg-x11-server::

    [blyth@localhost log]$ rpm -qf /usr/lib64/xorg/modules/extensions/libglx.so
    xorg-x11-server-Xorg-1.20.1-5.3.el7_6.x86_64

    blyth@localhost log]$ yum list installed | grep xorg-x11-server-Xorg
    xorg-x11-server-Xorg.x86_64            1.20.1-5.3.el7_6                @updates 


find others with the same issue : libglx.so compiled against wrong X server version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :google:`nvidia centos SEGV /usr/lib64/xorg/modules/extensions/libglx.so`


* https://devtalk.nvidia.com/default/topic/1044851/linux/fyi-nvidia-410-78-driver-fails-with-segmentation-fault-on-fedora-fc29-workstation-with-nvs-510-card/

::

    Update: nvidia support advised kindly (and very quickly, thank you very much!)
    that the following lines in the Xorg.0.log file suggest that the root of the
    problem seems to be that the "glx" module in the fc29 libglx.so library is
    compiled against X Server 1.19.3 instead of 1.20.3, and that this mismatch is
    incompatible with their driver:

    X.Org X Server 1.20.3
    [...]
    [ 9.736] Build ID: xorg-x11-server 1.20.3-1.fc29
    [...]
    [ 9.767] (II) LoadModule: "glx"
    [ 9.770] (II) Loading /usr/lib64/xorg/modules/extensions/libglx.so
    [ 9.780] (II) Module glx: vendor="X.Org Foundation"
    [ 9.780] compiled for 1.19.3, module version = 1.0.0
    [ 9.780] ABI class: X.Org Server Extension, version 10.0

    I filed a bug report for fc29 at https://bugzilla.redhat.com/show_bug.cgi?id=1655801.

    Posted 12/03/2018 11:54 PM   

* https://bugzilla.redhat.com/show_bug.cgi?id=1655801


following this fix works : remove old driver, reinstall xorg|mesa pkgs, install new driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/1046090/linux/rhel-7-6-x-server-will-not-start-using-latest-drivers-410-73-410-93-415-25-on-quadro-k2200-k620-k600/

::

    As documented in the "centos-7-geforce-gt740-unable-to-start-x-server" post:

    The xorg-server version 1.20 with glx module version 1.19 was the issue was
    the problem. To fix the issue I did the following:

    # systemctl isolate runlevel3
    # ./NVIDIA-Linux-x86_64-390.87.run --uninstall
    # yum -y reinstall `rpm -qa | egrep -i "xorg|mesa"`
    # ./NVIDIA-Linux-x86_64-410.93.run
    # systemctl isolate runlevel5 


* https://devtalk.nvidia.com/default/topic/1045859/linux/centos-7-geforce-gt740-unable-to-start-x-server/

::

    yum reinstall `rpm -qa | egrep -i "xorg|mesa"`  
       ## about 40 pkgs, the reinstall took less than a minute
   


EOU
}





onvidia-linux-console(){ cat << EOC

Linux Console
=======================

* :google:`linux switch to console RHEL`   # RHEL more hits than CentOS
* https://superuser.com/questions/144666/what-is-the-difference-between-shell-console-and-terminal

The console is a special sort of terminal. Historically, the console was a
single keyboard and monitor plugged into a dedicated serial console port on a
computer used for direct communication at a low level with the operating
system. Modern linux systems provide virtual consoles. These are accessed
through key combinations (e.g. Alt+F1 or Ctrl+Alt+F1; the function key numbers
different consoles) which are handled at low levels of the linux operating
system -- this means that there is no special service which needs to be
installed and configured to run. Interacting with the console is also done
using a shell program.


CentOS : Observe can switch GUI/Console with ctrl-alt-f1/f2
-------------------------------------------------------------

* ctrl-alt-f2 : to console
* ctrl-alt-f1 : to GUI 


chvt - change foreground virtual terminal
--------------------------------------------

The  command "chvt N" makes /dev/ttyN the foreground terminal.  
(The corresponding screen is created if it did not exist yet.  
To get rid of unused VTs, use deallocvt(1).)  
The key combination (Ctrl-)LeftAlt-FN (with N in the range  1-12) 
usually has a similar effect.


EOC
}





onnvidia-xorg-conf(){ cat << EOC

/etc/X11/xorg.conf the Device section specified driver "nvidia"
------------------------------------------------------------------

refs

* https://www.x.org/wiki/



::

    [root@localhost blyth]# grep Using\ config /var/log/Xorg.0.log
    [    46.389] (==) Using config file: "/etc/X11/xorg.conf"
    [    46.389] (==) Using config directory: "/etc/X11/xorg.conf.d"


    [root@localhost ~]# ls -l /etc/X11/xorg.conf.d
    total 4
    -rw-r--r--. 1 root root 232 Jul  5  2018 00-keyboard.conf


* http://download.nvidia.com/XFree86/Linux-x86/340.104/README/editxconfig.html

::

    [root@localhost blyth]# cat /etc/X11/xorg.conf
    # nvidia-xconfig: X configuration file generated by nvidia-xconfig
    # nvidia-xconfig:  version 396.26  (buildmeister@swio-display-x64-rhel04-19)  Mon Apr 30 18:40:19 PDT 2018


    Section "ServerLayout"
        Identifier     "Layout0"
        Screen      0  "Screen0" 0 0
        InputDevice    "Keyboard0" "CoreKeyboard"
        InputDevice    "Mouse0" "CorePointer"
    EndSection

    ...

    Section "Monitor"
        Identifier     "Monitor0"
        VendorName     "Unknown"
        ModelName      "Unknown"
        HorizSync       28.0 - 33.0
        VertRefresh     43.0 - 72.0
        Option         "DPMS"
    EndSection

    Section "Device"
        Identifier     "Device0"
        Driver         "nvidia"
        VendorName     "NVIDIA Corporation"
    EndSection

    Section "Screen"
        Identifier     "Screen0"
        Device         "Device0"
        Monitor        "Monitor0"
        DefaultDepth    24
        SubSection     "Display"
            Depth       24
        EndSubSection
    EndSection

EOC
}




onvidia-systemctl(){ cat << EOS

systemcrl replacement for init
==================================

See::

   man init
   man telinit

* https://www.tecmint.com/change-runlevels-targets-in-systemd/

::

    Run level 0 is matched by poweroff.target (and runlevel0.target is a symbolic link to poweroff.target).
    Run level 1 is matched by rescue.target (and runlevel1.target is a symbolic link to rescue.target).
    Run level 3 is emulated by multi-user.target (and runlevel3.target is a symbolic link to multi-user.target).
    Run level 5 is emulated by graphical.target (and runlevel5.target is a symbolic link to graphical.target).
    Run level 6 is emulated by reboot.target (and runlevel6.target is a symbolic link to reboot.target).
    Emergency is matched by emergency.target.

::

    [root@localhost log]# systemctl get-default
    graphical.target

    [root@localhost log]# systemctl set-default multi-user.target    ## old way was changing /etc/inittab
    Removed symlink /etc/systemd/system/default.target.
    Created symlink from /etc/systemd/system/default.target to /usr/lib/systemd/system/multi-user.target.

    [root@localhost log]# systemctl get-default
    multi-user.target


While the system is running, you can switch the target (run level), meaning
only services as well as units defined under that target will now run on the
system.

To switch to runlevel 3, run the following command.

    systemctl isolate multi-user.target      ## same as "init 3" (?)

To change the system to runlevel 5, type the command below.
 
    systemctl isolate graphical.target       ## same as "init 5" (?)


EOS
}


onvidia-disabling-nouveau(){ cat << EON

Nouveau driver module must not be loaded into kernel
------------------------------------------------------

Disabling nouveau is a 3 stage thing 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://linuxconfig.org/how-to-disable-nouveau-nvidia-driver-on-ubuntu-18-04-bionic-beaver-linux


::

    $ cat /etc/modprobe.d/blacklist-nvidia-nouveau.conf
    blacklist nouveau
    options nouveau modeset=0


1. write the /etc/modprobe.d/blacklist-nvidia-nouveau.conf
2. update kernel initramfs
3. reboot 


nvidia and nouveau : can theu both be loaded ? NO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://askubuntu.com/questions/271613/am-i-using-the-nouveau-driver-or-the-proprietary-nvidia-driver

If nvidia restricted module was loaded , then the nouveau module Cannot be loaded too (conflict each other).

nouveau hints from /var/log/nvidia-installer.log
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-> There appears to already be a driver installed on your system (version:
418.56).  As part of installing this driver (version: 396.26), the existing
driver will be uninstalled.  Are you sure you want to continue? (Answer:
Continue installation)

WARNING: One or more modprobe configuration files to disable Nouveau are already present at: 
    /usr/lib/modprobe.d/nvidia-installer-disable-nouveau.conf, 
    /etc/modprobe.d/nvidia-installer-disable-nouveau.conf.  
Please be sure you have rebooted your system since these files were written.  
If you have rebooted, then Nouveau may be enabled for other reasons, such as being included 
in the system initial ramdisk or in your X configuration file.  
Please consult the NVIDIA driver README and your Linux distribution's documentation 
for details on how to correctly disable the Nouveau kernel driver.
-> For some distributions, Nouveau can be disabled by adding a file in the modprobe configuration directory.  
Would you like nvidia-installer to attempt to create this modprobe file for you? (Answer: Yes)
-> One or more modprobe configuration files to disable Nouveau have been written.  
For some distributions, this may be sufficient to disable Nouveau; 
other distributions may require modification of the initial ramdisk.  
Please reboot your system and attempt NVIDIA driver installation again.  
Note if you later wish to reenable Nouveau, you will need to delete these files: 
   /usr/lib/modprobe.d/nvidia-installer-disable-nouveau.conf
   /etc/modprobe.d/nvidia-installer-disable-nouveau.conf


LOG : modify grub kernel commandline to absolutely definitely exclude nouveau
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* change /etc/default/grub removed "rhgb quiet"
* added : "rd.driver.blacklist=nouveau modprobe.blacklist=nouveau nouveau.modeset=0"

::

    [root@localhost blyth]# diff /etc/default/grub.orig /etc/default/grub
    6c6
    < GRUB_CMDLINE_LINUX="rd.lvm.lv=cl/root rd.lvm.lv=cl/swap rhgb quiet"
    ---
    > GRUB_CMDLINE_LINUX="rd.lvm.lv=cl/root rd.lvm.lv=cl/swap rd.driver.blacklist=nouveau modprobe.blacklist=nouveau nouveau.modeset=0"
    [root@localhost blyth]# 

Recreate /boot/grub2/grub.cfg::

    grub2-mkconfig -o /boot/grub2/grub.cfg

    [root@localhost default]# diff /boot/grub2/grub.cfg.orig /boot/grub2/grub.cfg   ## expected changes and a LANG too 


meanings of grub arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.redhat.com/archives/rhl-list/2004-May/msg07775.html

rhgb
    redhat graphical boot - This is a GUI mode booting screen with most of
    the information hidden while the user sees a rotating activity icon spining and
    brief information as to what the computer is doing.

quiet
    hides the majority of boot messages before rhgb starts. These are
    supposed to make the common user more comfortable. They get alarmed about
    seeing the kernel and initializing messages, so they hide them for their comfort.


* https://www.centos.org/forums/viewtopic.php?t=69421
* https://wiki.centos.org/HowTos/Grub2
* https://www.centos.org/forums/viewtopic.php?t=69421

    
LOG : CentOS : recreating initramfs with dracut
-------------------------------------------------

::

    [root@localhost blyth]# ls -l /boot/
    total 167432
    -rw-r--r--. 1 root root   137696 Nov 23  2016 config-3.10.0-514.el7.x86_64
    -rw-r--r--. 1 root root   147837 Jun 27  2018 config-3.10.0-862.6.3.el7.x86_64
    -rw-r--r--. 1 root root   151923 Mar 18 23:10 config-3.10.0-957.10.1.el7.x86_64
    drwx------. 3 root root       17 Nov  9 08:46 efi
    drwxr-xr-x. 2 root root       27 Jul  5  2018 grub
    drwx------. 5 root root      188 Apr  8 21:55 grub2
    -rw-------. 1 root root 48297217 Jul  5  2018 initramfs-0-rescue-f42ac84eae3c4cecb3d493c899463c30.img
    -rw-------. 1 root root 22175491 Jul  5  2018 initramfs-3.10.0-514.el7.x86_64.img
    -rw-------. 1 root root 32372862 Apr  5 22:13 initramfs-3.10.0-862.6.3.el7.x86_64.img
    -rw-------. 1 root root 32885195 Apr  5 22:11 initramfs-3.10.0-957.10.1.el7.x86_64.img
    -rw-r--r--. 1 root root   613759 Jul  5  2018 initrd-plymouth.img
    -rw-r--r--. 1 root root   277953 Nov 23  2016 symvers-3.10.0-514.el7.x86_64.gz
    -rw-r--r--. 1 root root   305113 Jun 27  2018 symvers-3.10.0-862.6.3.el7.x86_64.gz
    -rw-r--r--. 1 root root   314087 Mar 18 23:10 symvers-3.10.0-957.10.1.el7.x86_64.gz
    -rw-------. 1 root root  3113253 Nov 23  2016 System.map-3.10.0-514.el7.x86_64
    -rw-------. 1 root root  3412056 Jun 27  2018 System.map-3.10.0-862.6.3.el7.x86_64
    -rw-------. 1 root root  3544363 Mar 18 23:10 System.map-3.10.0-957.10.1.el7.x86_64
    -rwxr-xr-x. 1 root root  5392080 Jul  5  2018 vmlinuz-0-rescue-f42ac84eae3c4cecb3d493c899463c30
    -rwxr-xr-x. 1 root root  5392080 Nov 23  2016 vmlinuz-3.10.0-514.el7.x86_64
    -rwxr-xr-x. 1 root root  6233824 Jun 27  2018 vmlinuz-3.10.0-862.6.3.el7.x86_64
    -rwxr-xr-x. 1 root root  6643904 Mar 18 23:10 vmlinuz-3.10.0-957.10.1.el7.x86_64
    [root@localhost blyth]# uname -r
    3.10.0-957.10.1.el7.x86_64

::

    [root@localhost blyth]# dracut
    Will not override existing initramfs (/boot/initramfs-3.10.0-957.10.1.el7.x86_64.img) without --force

    Broadcast message from systemd-journald@localhost.localdomain (Mon 2019-04-08 22:07:24 CST):

    dracut[17119]: Will not override existing initramfs (/boot/initramfs-3.10.0-957.10.1.el7.x86_64.img) without --force


    Message from syslogd@localhost at Apr  8 22:07:24 ...
     dracut:Will not override existing initramfs (/boot/initramfs-3.10.0-957.10.1.el7.x86_64.img) without --force
    [root@localhost blyth]# 

    [root@localhost blyth]# dracut --force
    [root@localhost blyth]# 


    [root@localhost blyth]# ls -l /boot/initramfs-3.10.0-957.10.1.el7.x86_64.img
    -rw-------. 1 root root 32486967 Apr  8 22:09 /boot/initramfs-3.10.0-957.10.1.el7.x86_64.img


    [root@localhost blyth]# echo $(( 32885195 - 32486967))     ## significantly smaller
    398228


Fedora
--------

* https://ask.fedoraproject.org/en/question/117408/how-to-diable-nouveau-in-fedora27/

Add nouveau.modeset=0 rd.driver.blacklist=nouveau at the end of the line beginning with linux. 


Ubuntu Grub
------------

* https://help.ubuntu.com/community/Grub2/Setup

::

   update-grub


EON
}


onvidia-x11-docs(){ cat << EOC


NVIDIA X11
------------

* http://download.nvidia.com/XFree86/Linux-x86/340.104/README/introduction.html
* http://download.nvidia.com/XFree86/Linux-x86/340.104/README/commonproblems.html
* https://us.download.nvidia.cn/XFree86/Linux-x86_64/384.69/README/installdriver.html

interaction with the Nouveau driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://us.download.nvidia.cn/XFree86/Linux-x86_64/384.69/README/commonproblems.html#nouveau

...Nouveau must be disabled before installing the NVIDIA driver

Nouveau performs modesets in the kernel. This can make disabling Nouveau
difficult, as the kernel modeset is used to display a framebuffer console,
which means that Nouveau will be in use even if X is not running. As long as
Nouveau is in use, its kernel module cannot be unloaded, which will prevent the
NVIDIA kernel module from loading. It is therefore important to make sure that
Nouveau's kernel modesetting is disabled before installing the NVIDIA driver.

EOC
}


onvidia-make-sure-nouveau-is-not-loaded(){ cat << EON

Ensuring nouveau not loaded into kernel
-----------------------------------------

Not so easy. Success is when its no longer mentioned in Xorg.0.log.
For details see above onvidia-x11-docs 


::

    [root@localhost xorg-2019-04-08-14:33:37-22120-1]# grep Loading Xorg.0.log 
    [   385.687] (II) Loading /usr/lib64/xorg/modules/extensions/libglx.so
    [   385.938] (II) Loading /usr/lib64/xorg/modules/drivers/nvidia_drv.so
    [   385.940] (II) Loading /usr/lib64/xorg/modules/drivers/nouveau_drv.so
    [   385.941] (II) Loading /usr/lib64/xorg/modules/drivers/modesetting_drv.so
    [   385.941] (II) Loading /usr/lib64/xorg/modules/drivers/fbdev_drv.so
    [   385.942] (II) Loading /usr/lib64/xorg/modules/drivers/vesa_drv.so
    [   385.954] (II) Loading sub module "fb"
    [   385.954] (II) Loading /usr/lib64/xorg/modules/libfb.so
    [   385.955] (II) Loading sub module "wfb"
    [   385.955] (II) Loading /usr/lib64/xorg/modules/libwfb.so
    [   385.955] (II) Loading sub module "ramdac"
    [   385.957] (II) Loading sub module "fbdevhw"
    [   385.958] (II) Loading /usr/lib64/xorg/modules/libfbdevhw.so
    [   385.959] (II) Loading sub module "glxserver_nvidia"
    [   385.959] (II) Loading /usr/lib64/xorg/modules/extensions/libglxserver_nvidia.so
    [   387.962] (II) Loading sub module "dri2"
    [root@localhost xorg-2019-04-08-14:33:37-22120-1]# 



    ## after reverting to the old driver : note that noveau is not loaded ???

    [root@localhost cuda-10.1]# grep Loading /var/log/Xorg.0.log
    [   639.338] (II) Loading /usr/lib64/xorg/modules/extensions/libglx.so
    [   639.346] (II) Loading /usr/lib64/xorg/modules/drivers/nvidia_drv.so
    [   639.356] (II) Loading sub module "fb"
    [   639.356] (II) Loading /usr/lib64/xorg/modules/libfb.so
    [   639.357] (II) Loading sub module "wfb"
    [   639.357] (II) Loading /usr/lib64/xorg/modules/libwfb.so
    [   639.358] (II) Loading sub module "ramdac"
    [   641.256] (II) Loading sub module "dri2"
    [   641.308] (II) Loading /usr/lib64/xorg/modules/input/evdev_drv.so
    [root@localhost cuda-10.1]# 


EON
}



onvidia-background(){ cat << EOI

Background Info on Linux booting
===================================

Ubuntu : what is initramfs
-----------------------------

* https://ubuntuforums.org/showthread.php?t=2356604

::

    initramfs is a tiny version of the OS that gets loaded by the bootloader, right
    after the kernel. It lives in RAM, and it provides *just* enough tools and
    instructions to tell the kernel how to set up the real filesystem, mount the
    HDD read/write, and start loading all the system services. It includes the stub
    of init, PID #1. If your initramfs is broken, your boot fails.

    update-initramfs is a script that updates initramfs to work with a new kernel.
    In the Debian universe, you shouldn't need to run this command manually except
    under very unusual circumstances - a post-install script automatically handles
    it for you when you install a new kernel package. 

::

    [blyth@localhost Downloads]$ ll /boot/initramfs*
    -rw-------. 1 root root 22175491 Jul  5  2018 /boot/initramfs-3.10.0-514.el7.x86_64.img
    -rw-------. 1 root root 48297217 Jul  5  2018 /boot/initramfs-0-rescue-f42ac84eae3c4cecb3d493c899463c30.img
    -rw-------. 1 root root 32885195 Apr  5 22:11 /boot/initramfs-3.10.0-957.10.1.el7.x86_64.img
    -rw-------. 1 root root 32372862 Apr  5 22:13 /boot/initramfs-3.10.0-862.6.3.el7.x86_64.img

    [blyth@localhost Downloads]$ du -h /boot/initramfs*
    47M	/boot/initramfs-0-rescue-f42ac84eae3c4cecb3d493c899463c30.img
    22M	/boot/initramfs-3.10.0-514.el7.x86_64.img
    31M	/boot/initramfs-3.10.0-862.6.3.el7.x86_64.img
    32M	/boot/initramfs-3.10.0-957.10.1.el7.x86_64.img


* https://wiki.gentoo.org/wiki/Initramfs/Guide

To create an initramfs, it is important to know what additional drivers,
scripts and tools will be needed to boot the system. For instance, if LVM is
used, then LVM tools will be needed in the initramfs. Likewise, if software
RAID is used, mdadm utilities will be needed, etc. 

::

   so it seems that its necessary to regenerate the initramfs when you want 
   subsequent boots to have different drivers : eg when you want to remove nouveau


CentOS : dracut - low-level tool for generating an initramfs image 
----------------------------------------------------------------------

dracut creates an initial image used by the kernel for preloading the block
device modules (such as IDE, SCSI or RAID) which are needed to access the root
filesystem, mounting the root filesystem and booting into the real system.

If the initramfs image already exists, dracut will display an error message,
and to overwrite the existing image, you have to use the --force option.
  
::

   dracut --force



EOI
}



onvidia-tools(){ cat << EOT

Tools for debugging driver issues
=====================================




nvidia-smi reports Driver Version :  396.26 
----------------------------------------------

::

    [root@localhost ~]# nvidia-smi
    Mon Apr  8 20:42:34 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 396.26                 Driver Version: 396.26                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN V             Off  | 00000000:73:00.0  On |                  N/A |
    | 28%   40C    P8    27W / 250W |    213MiB / 12066MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0     13731      G   /usr/bin/X                                   117MiB |
    |    0     15881      G   /usr/bin/gnome-shell                          94MiB |


driver version
-----------------

::

    [root@localhost blyth]# cat /proc/driver/nvidia/version
    NVRM version: NVIDIA UNIX x86_64 Kernel Module  396.26  Mon Apr 30 18:01:39 PDT 2018
    GCC version:  gcc version 4.8.5 20150623 (Red Hat 4.8.5-36) (GCC) 
    [root@localhost blyth]# 




dmesg is used to examine or control the kernel ring buffer.
-------------------------------------------------------------

::

   dmesg --color  



lspci : listing devices on the PCI bus
----------------------------------------

::

    [blyth@localhost ~]$ lspci | grep -i nv
    73:00.0 VGA compatible controller: NVIDIA Corporation GV100 [TITAN V] (rev a1)
    73:00.1 Audio device: NVIDIA Corporation Device 10f2 (rev a1)

    [root@localhost blyth]# lspci -nnk | grep -iA10 vga    ## grep case insensitive with context after the match
    73:00.0 VGA compatible controller [0300]: NVIDIA Corporation GV100 [TITAN V] [10de:1d81] (rev a1)
        Subsystem: NVIDIA Corporation Device [10de:1218]
        Kernel driver in use: nvidia
        Kernel modules: nouveau, nvidia_drm, nvidia
    73:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:10f2] (rev a1)
        Subsystem: NVIDIA Corporation Device [10de:1218]
        Kernel driver in use: snd_hda_intel
        Kernel modules: snd_hda_intel
    a0:04.0 System peripheral [0880]: Intel Corporation Sky Lake-E CBDMA Registers [8086:2021] (rev 04)
        Subsystem: Intel Corporation Device [8086:0000]
    a0:04.1 System peripheral [0880]: Intel Corporation Sky Lake-E CBDMA Registers [8086:2021] (rev 04)
    [root@localhost blyth]# 

::

    [blyth@localhost ~]$ sudo lspci -s 73:00.0 -v
    73:00.0 VGA compatible controller: NVIDIA Corporation GV100 [TITAN V] (rev a1) (prog-if 00 [VGA controller])
        Subsystem: NVIDIA Corporation Device 1218
        Flags: bus master, fast devsel, latency 0, IRQ 47, NUMA node 0
        Memory at c4000000 (32-bit, non-prefetchable) [size=16M]
        Memory at b0000000 (64-bit, prefetchable) [size=256M]
        Memory at c0000000 (64-bit, prefetchable) [size=32M]
        I/O ports at 9000 [size=128]
        [virtual] Expansion ROM at c5000000 [disabled] [size=512K]
        Capabilities: [60] Power Management version 3
        Capabilities: [68] MSI: Enable+ Count=1/1 Maskable- 64bit+
        Capabilities: [78] Express Legacy Endpoint, MSI 00
        Capabilities: [100] Virtual Channel
        Capabilities: [250] Latency Tolerance Reporting
        Capabilities: [258] L1 PM Substates
        Capabilities: [128] Power Budgeting <?>
        Capabilities: [420] Advanced Error Reporting
        Capabilities: [600] Vendor Specific Information: ID=0001 Rev=1 Len=024 <?>
        Capabilities: [900] #19
        Capabilities: [ac0] #23
        Kernel driver in use: nvidia
        Kernel modules: nouveau, nvidia_drm, nvidia




lshw
~~~~~~

::

    [root@localhost blyth]# lshw -class video
      *-display                 
           description: VGA compatible controller
           product: GV100 [TITAN V]
           vendor: NVIDIA Corporation
           physical id: 0
           bus info: pci@0000:73:00.0
           version: a1
           width: 64 bits
           clock: 33MHz
           capabilities: pm msi pciexpress vga_controller bus_master cap_list rom
           configuration: driver=nvidia latency=0
           resources: irq:47 memory:c4000000-c4ffffff memory:b0000000-bfffffff memory:c0000000-c1ffffff ioport:9000(size=128) memory:c5000000-c507ffff
    [root@localhost blyth]# 



modinfo : Show information about a Linux Kernel module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [root@localhost blyth]# modinfo nvidia
    filename:       /lib/modules/3.10.0-957.10.1.el7.x86_64/kernel/drivers/video/nvidia.ko
    alias:          char-major-195-*
    version:        396.26
    supported:      external
    license:        NVIDIA
    retpoline:      Y
    rhelversion:    7.6
    srcversion:     AE579930EF8F20A66867263
    alias:          pci:v000010DEd00000E00sv*sd*bc04sc80i00*
    alias:          pci:v000010DEd*sv*sd*bc03sc02i00*
    alias:          pci:v000010DEd*sv*sd*bc03sc00i00*
    depends:        ipmi_msghandler
    vermagic:       3.10.0-957.10.1.el7.x86_64 SMP mod_unload modversions 
    parm:           NVreg_Mobile:int
    parm:           NVreg_ResmanDebugLevel:int
    parm:           NVreg_RmLogonRC:int
    parm:           NVreg_ModifyDeviceFiles:int
    parm:           NVreg_DeviceFileUID:int
    parm:           NVreg_DeviceFileGID:int
    parm:           NVreg_DeviceFileMode:int
    parm:           NVreg_UpdateMemoryTypes:int
    parm:           NVreg_InitializeSystemMemoryAllocations:int
    parm:           NVreg_UsePageAttributeTable:int
    parm:           NVreg_MapRegistersEarly:int
    parm:           NVreg_RegisterForACPIEvents:int
    parm:           NVreg_CheckPCIConfigSpace:int
    parm:           NVreg_EnablePCIeGen3:int
    parm:           NVreg_EnableMSI:int
    parm:           NVreg_TCEBypassMode:int
    parm:           NVreg_UseThreadedInterrupts:int
    parm:           NVreg_EnableStreamMemOPs:int
    parm:           NVreg_EnableBacklightHandler:int
    parm:           NVreg_EnableUserNUMAManagement:int
    parm:           NVreg_MemoryPoolSize:int
    parm:           NVreg_IgnoreMMIOCheck:int
    parm:           NVreg_RegistryDwords:charp
    parm:           NVreg_RegistryDwordsPerDevice:charp
    parm:           NVreg_RmMsg:charp
    parm:           NVreg_AssignGpus:charp
    [root@localhost blyth]# 



lsmod - Show the status of modules in the Linux Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [root@localhost blyth]# lsmod | grep nvidia
    nvidia_drm             43785  3 
    nvidia_modeset       1081940  6 nvidia_drm
    nvidia              14019834  266 nvidia_modeset
    drm_kms_helper        179394  1 nvidia_drm
    drm                   429744  6 drm_kms_helper,nvidia_drm
    ipmi_msghandler        56032  2 ipmi_devintf,nvidia
    [root@localhost blyth]# 
    [root@localhost blyth]# 
    [root@localhost blyth]# lsmod | grep nouveau



modprobe - Add and remove modules from the Linux Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [root@localhost blyth]# modprobe --showconfig | grep nouveau
    blacklist nouveau
    blacklist nouveau
    options nouveau modeset=0
    options nouveau modeset=0
    alias pci:v000010DEd*sv*sd*bc03sc*i* nouveau
    alias pci:v000012D2d*sv*sd*bc03sc*i* nouveau
    [root@localhost blyth]# 

    [root@localhost blyth]# modprobe --showconfig | grep nvidia
    blacklist nvidiafb
    alias char_major_195_* nvidia
    alias mbp_nvidia_bl apple_bl
    alias pci:v000010DEd*sv*sd*bc03sc00i00* nvidia_drm
    alias pci:v000010DEd*sv*sd*bc03sc00i00* nvidia
    alias pci:v000010DEd*sv*sd*bc03sc02i00* nvidia_drm
    alias pci:v000010DEd*sv*sd*bc03sc02i00* nvidia
    alias pci:v000010DEd00000E00sv*sd*bc04sc80i00* nvidia_drm
    alias pci:v000010DEd00000E00sv*sd*bc04sc80i00* nvidia
    alias symbol:nvKmsKapiGetFunctionsTable nvidia_modeset
    alias symbol:nvUvmInterfaceAddressSpaceCreate nvidia
    alias symbol:nvUvmInterfaceAddressSpaceDestroy nvidia
    alias symbol:nvUvmInterfaceBindChannelResources nvidia
    alias symbol:nvUvmInterfaceChannelAllocate nvidia
    alias symbol:nvUvmInterfaceChannelDestroy nvidia
    ...



Update Silver Precision Driver for Quadro RTX 8000
-------------------------------------------------------

* logout 
* from ssh and nvidia-smi see that X and gnome are still using GPU, so 
   

From root over ssh switch to non-graphics::

    [root@gilda03 ~]# systemctl isolate multi-user.target 

    
nvidia-smi now shows nothing on GPU.

* run the runscript

::

    [root@gilda03 ~]# cd /home/blyth
    [root@gilda03 ~]# bash NVIDIA-Linux-x86_64-430.50.run

A console text GUI comes up::

    There appears to already be a driver installed on your system (version:
    418.56).  As part of installing this driver (version: 430.50), the existing
    driver will be uninstalled.  Are you sure you want to continue?

* said Y to 32-bit compatibility libs and xconfig setup
* after that nvidia-smi shows the updated driver


Back to X GUI::

    [root@gilda03 blyth]# systemctl isolate runlevel5      ## same as init 5 ?



Fixed problems with nvidia driver downloads page by adding few more sites to gfw- proxy list
------------------------------------------------------------------------------------------------

* https://www.nvidia.com/Download/Find.aspx?lang=en-us

* driver list from "All" search for Quadro RTX 8000 Linux 64-bit drivers

::

    430.50   September 11, 2019
    435.21   August 29, 2019
    418.88   July 29, 2019
    430.40   July 29, 2019
    430.34   July 9, 2019
    430.26   June 10, 2019
    430.14   May 14, 2019
    418.74   May 7, 2019
    430.09   April 23, 2019          BETA
    418.56   March 20, 2019          (using this on DELL Precision.Gold with OptiX 6, TITAN RTX and V)
    418.43   February 22, 2019
    410.104  February 22, 2019 
    418.30   January 30, 2019        BETA
    410.93   January 3, 2019         (added support for Quadro RTX 8000) 

Looks to be three series::

    430 long-lived branch 
    435 short-lived branch and official release
    418 

    

OptiX 6: minimum R418
OptiX 7: 435.12 or newer on Linux

* https://www.nvidia.com/download/driverResults.aspx/149785/en-us
  435.17 BETA (does not list Quadro RTX 8000)

* https://www.nvidia.com/download/driverResults.aspx/150803/en-us
  435.21 2019.8.29 Quadro RTX 8000 is listed 



Making sense of driver versions

* https://devtalk.nvidia.com/default/topic/1048306/linux/what-are-the-different-driver-versions-long-lived-vs-short-lived-vs-geforce-com/post/5321905/#5321905

Any given release branch is either long-lived or short-lived. The difference is
in how long the branch is maintained and how many releases made from each
branch. A short-lived branch typically has only one or two (non-beta) releases,
while long-lived branches will have several.

So both 410 and 418 are long-lived branches, and the releases for 410.104 and
418.43 just reflect that -- some customers are still using the 410 series and
don't want to move to 418, so we made a release from that branch even though a
newer long-lived branch is available.

When we make changes to the driver, we evaluate the oldest branch the change
needs to go into. New features go into whatever the latest branch is, while bug
fixes go into the older branches and are integrated through the newer branches.
So using a *short-lived branch doesn't mean that you miss out on fixes, it just
means that you also get the latest features*.

The sticky post lists the latest releases from the current short- and
long-lived branches.

* https://devtalk.nvidia.com/default/topic/533434/linux/current-graphics-driver-releases/

  Current releases (as of Sep 12, 2019)
  Current long-lived branch release: 430.50 (x86_64)
  Current official release: 435.21 (x86_64)

  New legacy release 390.129 is now available.  Posted 07/29/2019 10:14 PM   
  New short-lived branch release 435.21 is now available.  Posted 08/29/2019 04:59 PM   
  New long-lived branch release 430.50 is now available.  Posted 19 hours ago (Sept 12 2019)


* https://www.nvidia.com/en-us/drivers/unix/linux-amd64-display-archive/



* https://devtalk.nvidia.com/default/topic/1030867/linux/what-does-the-short-long-lived-branch-mean-/

I don't think they are more conservative with long lived branches. I think they
just support long lived branches for six months or so and short lived branches
for around three months only. So if 390 is a long lived branch, 393 is a short
lived branch, 396 is a long lived branch, and so on.  In other words, it is one
support release for three months, two for the next three months, then one again
for another three months. 

* https://wiki.ubuntu.com/NVidiaUpdates


nvidia-smi monitoring
------------------------

::

    watch -d -n 0.5 nvidia-smi

EOT
}


