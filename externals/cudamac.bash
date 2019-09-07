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

cudamac-source(){   echo $BASH_SOURCE; }
cudamac-vi(){       vi $(cudamac-source) ; }
cudamac-env(){      olocal- ; }
cudamac-usage(){ cat << \EOU

CUDA on Mac
==============

Guide
--------

* http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html

eGPU 
-----

* https://github.com/marnovo/macOS-eGPU-CUDA-guide
* https://support.apple.com/en-us/HT208544#ntf

  None of the supported GPUs are from NVIDIA


Apple and NVIDIA : Why dont they like to play together ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://appleinsider.com/articles/19/01/18/apples-management-doesnt-want-nvidia-support-in-macos-and-thats-a-bad-sign-for-the-mac-pro



CUDA Driver 387.178
----------------------

In System Prefs observe the prefpane says there us a newer 
CUDA Driver 387.178 and that are currently on 387.128

* huh : I thought thats the one I installed (maybe because not yet rebooted?)
* even after reboot stays the same

Current macOS builds installed
--------------------------------

::

   10.9.4      13E28    /Volumes/Delta
   10.9.4      13E28    /Volumes/DeltaClone
   10.13.3     17D102   /Volumes/TestHighSierra
   10.13.4     17E199   /Volumes/Epsilon  

* in "About This Mac" option click on the 
  version eg "10.13.4" to see the build code eg "17E199"

nvidia display driver (aka GPU driver)
----------------------------------------

The GPU driver is normally provided by the vendor (apple) 
as typically you do not update GPUs in Macs.  BUT the CUDA
driver requires a newer GPU driver than the old one
provided by apple.

GPU drivers are kernel extensions (forcing caution), and requiring a 
precise match between the GPU driver and macOS version+build number.   


Dec 2018 : Review Versions
----------------------------

Before Update::

    macOS        : 10.13.4 17E199        (About This Mac + option click on version)
    GPU Driver   : 387.10.10.10.30.106   (Menu bar NVIDIA panel OR from System Prefs > NVIDIA Driver Manager)
    CUDA Driver  : 387.128               (from System Prefs > CUDA)
 
    (*) states that CUDA 410.130 Driver is available

After macOS Update + GPU Driver Update::

    macOS         :  10.13.6 (17G65)
    GPU Driver    :  387.10.10.10.40.105   
    CUDA Driver   :  387.128 


Look for problems with searches on version numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :google:`macos 10.13.6 nvidia web driver`

* https://www.nvidia.com/download/driverResults.aspx/136062/en-us

  QUADRO & GEFORCE MACOS DRIVER RELEASE 387.10.10.10.40.105

  New in Release 387.10.10.10.40.105:
  Graphics driver updated for macOS High Sierra 10.13.6 (17G65)


Thoughts on when to update macOS + GPU driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you have a good reason to do so, eg want to use a newer Xcode 10.1 for its support for new devices.
Also require:

* macOS updates are at least 2-3 months old 
* you are not in a critical period, eg just before a presentation


Aligning three release sequences :  macOS point release, NVIDIA GPU drivers, NVIDIA CUDA drivers
-----------------------------------------------------------------------------------------------------

Sources:

* (third party list of GPU drivers for each macOS build) http://www.macvidcards.com/drivers.html
* (third party list of macOS point releases) https://robservatory.com/a-useless-analysis-of-os-x-release-dates/
* NVIDIA list of CUDA releases,  https://www.nvidia.com/object/mac-driver-archive.html 

Other sources:

* (news of new web release display drivers) https://www.insanelymac.com/forum/forums/topic/324195-nvidia-web-driver-updates-for-macos-high-sierra-update-03302018/
* https://forums.geforce.com/default/topic/1044393/macos-10-13-4-and-nvidia-web-driver-/?offset=7

Observations:

* macOS point releases eg 10.13.4 approx every month, and then builds 
  are released circa every week thereafter

* NVIDIA display drivers released soon after every macOS build release 

* NVIDIA CUDA drivers released only for the first point release build, 
  within a couple of days from the point release

Extrapolation:

* highly likely that a new CUDA driver appropriate for 10.13.4(17E199) or a subsequent 
  build will arrive within a few days (early April) : so just hold out for that

  * YEP: CUDA Driver 387.178, 04/02/2018

Strategy:

* following macOS point releases do not allow Mac App Store to update, hold back
  until the CUDA Driver comes out : which will inform you of which triptych 
  to run with for the next month or so 

* so this goes on every month or so, **dont update when Mac App Store suggests it**
  wait until CUDA driver comes out 

* when being cautious (eg when have working system) could update the EpsilonClone
  on external SSD and do the Triptych update first on that, steps:

  * update EpsilonClone with clone- or Superduper/CCC
  * boot into the EpsilonClone and allow Mac App Store to update to the 
    point release and build that has been annointed by NVIDIA
  * update GPU driver
  * update CUDA driver
  * test CUDA/OptiX/Opticks on the external 

   
::


    10.13.2 [2017 Dec 6, 10.13.2]

        (17C88) 378.10.10.10.25.102    (## CUDA Driver 387.99, 12/08/2017 ##)
        (17C89) 378.10.10.10.25.103 
       (17C205) 378.10.10.10.25.104 
      (17C2120) 378.10.10.10.25.105 
      (17C2205) 378.10.10.10.25.106

    10.13.3  [2018 Jan 23, 10.13.3]

       (17D47)   387.10.10.10.25.156   (## CUDA Driver 387.128, 01/25/2018 ##) 
       (17D2047) 387.10.10.10.25.157 
        (17D102) 387.10.10.10.25.158 
       (17D2102) 387.10.10.10.25.159

    10.13.4 [2018 Mar 29, 10.13.4]

       (17E199) 387.10.10.10.30.103   (## CUDA Driver 387.178, 04/02/2018  ##)


CUDA Mac Driver : Archive
----------------------------

* https://www.nvidia.com/object/mac-driver-archive.html

::

    CUDA 410.130 driver for MAC   Release Date: 09/19/2018

    CUDA 396.148 driver for MAC  Release Date: 07/09/2018
    CUDA 396.64 driver for MAC   Release Date: 05/17/2018
    CUDA 387.178 driver for MAC  Release Date: 04/02/2018
    CUDA 387.128 driver for MAC  Release Date: 01/25/2018













* High Sierra 10.13.6 now listed in App store, updates


CUDA Driver Syspanel
----------------------

Says::

    CUDA Driver Version : 387.128    (CUDA 396.148 Driver update is available)




Spate of kernel panics
------------------------

* https://forums.geforce.com/default/topic/930758/geforce-drivers/-problem-spontaneous-kernel-panics-with-nvidia-web-driver-on-mac/


::

    865 Aug 20 10:23:04 epsilon GoogleSoftwareUpdateAgent[33997]: 2018-08-20 10:23:04.616 GoogleSoftwareUpdateAgent[33997/0x7fff98781380] [lvl=2] -[KSAgentApp(PrivateMethods) applicationDidFinishLaunching:]      Agent running failed with error: <KSError:0x1004195b0
     866         domain="com.Google.Keystone.AgentErrorDomain"
     867         code=103
     868         userInfo={
     869             NSLocalizedDescription = "Agent is running as user id 504, which is not the console user.";
     870             line = 409;
     871             filename = "KSAgentApp.m";
     872             function = "-[KSAgentApp(PrivateMethods) checkForUpdatesUsingArguments:invocation:error:]";
     873             date = 2018-08-20 02:23:04 +0000;
     874         }
     875     >
     876 Aug 20 10:23:04 epsilon com.apple.xpc.launchd[1] (com.google.keystone.user.agent[33997]): Service exited with abnormal code: 1
     877 Aug 20 10:27:08 localhost bootlog[0]: BOOT_TIME 1534732028 0
     878 Aug 20 10:27:21 localhost syslogd[56]: Configuration Notice:




::

    Anonymous UUID:       32BCAB7F-2AEA-A951-3785-013ECFB913EA

    Mon Aug 20 10:27:21 2018

    *** Panic Report ***
    panic(cpu 0 caller 0xffffff800e489754): "thread_invoke: preemption_level 1, possible cause: blocking while holding a spinlock, or within interrupt context"@/BuildRoot/Library/Caches/com.apple.xbs/Sources/xnu/xnu-4570.51.1/osfmk/kern/sched_prim.c:2231
    Backtrace (CPU 0), Frame : Return Address
    0xffffff91ff2f3610 : 0xffffff800e46e166 
    0xffffff91ff2f3660 : 0xffffff800e596714 
    0xffffff91ff2f36a0 : 0xffffff800e588a00 
    0xffffff91ff2f3720 : 0xffffff800e420180 
    0xffffff91ff2f3740 : 0xffffff800e46dbdc 
    0xffffff91ff2f3870 : 0xffffff800e46d99c 
    0xffffff91ff2f38d0 : 0xffffff800e489754 
    0xffffff91ff2f3950 : 0xffffff800e4885df 
    0xffffff91ff2f39a0 : 0xffffff800e581fd6 
    0xffffff91ff2f3a00 : 0xffffff800e41eaad 
    0xffffff91ff2f3a20 : 0xffffff800e4f3f91 
    0xffffff91ff2f3b00 : 0xffffff800e4ba723 
    0xffffff91ff2f3c30 : 0xffffff800e44fc16 
    0xffffff91ff2f3c60 : 0xffffff800e450803 
    0xffffff91ff2f3cb0 : 0xffffff800e473df2 
    0xffffff91ff2f3cf0 : 0xffffff7f8ed979fd 
    0xffffff91ff2f3d30 : 0xffffff7f8edef029 
    0xffffff91ff2f3d50 : 0xffffff7f8eea759e 
    0xffffff91ff2f3da0 : 0xffffff7f8eeeca8c 
    0xffffff91ff2f3dc0 : 0xffffff7f8faaa242 
    0xffffff91ff2f3e10 : 0xffffff7f8edf621c 
    0xffffff91ff2f3ed0 : 0xffffff800ea9a005 
    0xffffff91ff2f3f30 : 0xffffff800ea98772 
    0xffffff91ff2f3f70 : 0xffffff800ea97dac 
    0xffffff91ff2f3fa0 : 0xffffff800e41f4f7 
          Kernel Extensions in backtrace:
             com.nvidia.web.NVDAResmanWeb(10.3.1)[8E2AB3E3-4EE5-3F90-B6D8-54CEB8595A5F]@0xffffff7f8ed90000->0xffffff7f8f408fff
                dependency: com.apple.iokit.IOPCIFamily(2.9)[1850E7DA-E707-3027-A3AA-637C80B57219]@0xffffff7f8ec94000
                dependency: com.apple.iokit.IONDRVSupport(519.15)[B419F958-11B8-3F7D-A31B-A72166B6E234]@0xffffff7f8ed75000
                dependency: com.apple.iokit.IOGraphicsFamily(519.15)[D5F2A20D-CAB0-33B2-91B9-E8755DFC34CB]@0xffffff7f8ed1f000
                dependency: com.apple.AppleGraphicsDeviceControl(3.18.48)[89491182-0B41-3BC3-B16F-D5043425D66F]@0xffffff7f8ed85000
             com.nvidia.web.NVDAGK100HalWeb(10.3.1)[BC0C27F0-12AF-36CA-AC52-ACD84F718B30]@0xffffff7f8f99b000->0xffffff7f8faf8fff
                dependency: com.nvidia.web.NVDAResmanWeb(10.3.1)[8E2AB3E3-4EE5-3F90-B6D8-54CEB8595A5F]@0xffffff7f8ed90000
                dependency: com.apple.iokit.IOPCIFamily(2.9)[1850E7DA-E707-3027-A3AA-637C80B57219]@0xffffff7f8ec94000

    BSD process name corresponding to current thread: kernel_task

    Mac OS version:
    17E199

    Kernel version:
    Darwin Kernel Version 17.5.0: Mon Mar  5 22:24:32 PST 2018; root:xnu-4570.51.1~1/RELEASE_X86_64
    Kernel UUID: 1B55340B-0B14-3026-8A47-1E139DB63DA3




::

    Sun Aug 19 13:01:16 2018

       *** Panic Report ***
        panic(cpu 0 caller 0xffffff800da89754): "thread_invoke: preemption_level 1, possible cause: blocking while holding a spinlock, or within interrupt context"@/BuildRoot/Library/Caches/com.apple.xbs/Sources/xnu/xnu-4570.51.1/osfmk/kern/sched_prim.c:2231
        Backtrace (CPU 0), Frame : Return Address
        0xffffff81f8ec3610 : 0xffffff800da6e166 
        0xffffff81f8ec3660 : 0xffffff800db96714 
        0xffffff81f8ec36a0 : 0xffffff800db88a00 
        ...

         Kernel Extensions in backtrace:
             com.nvidia.web.NVDAResmanWeb(10.3.1)[8E2AB3E3-4EE5-3F90-B6D8-54CEB8595A5F]@0xffffff7f8e390000->0xffffff7f8ea08fff
                dependency: com.apple.iokit.IOPCIFamily(2.9)[1850E7DA-E707-3027-A3AA-637C80B57219]@0xffffff7f8e294000
                dependency: com.apple.iokit.IONDRVSupport(519.15)[B419F958-11B8-3F7D-A31B-A72166B6E234]@0xffffff7f8e375000
                dependency: com.apple.iokit.IOGraphicsFamily(519.15)[D5F2A20D-CAB0-33B2-91B9-E8755DFC34CB]@0xffffff7f8e31f000
                dependency: com.apple.AppleGraphicsDeviceControl(3.18.48)[89491182-0B41-3BC3-B16F-D5043425D66F]@0xffffff7f8e385000
             com.nvidia.web.NVDAGK100HalWeb(10.3.1)[BC0C27F0-12AF-36CA-AC52-ACD84F718B30]@0xffffff7f8ef9b000->0xffffff7f8f0f8fff
                dependency: com.nvidia.web.NVDAResmanWeb(10.3.1)[8E2AB3E3-4EE5-3F90-B6D8-54CEB8595A5F]@0xffffff7f8e390000
                dependency: com.apple.iokit.IOPCIFamily(2.9)[1850E7DA-E707-3027-A3AA-637C80B57219]@0xffffff7f8e294000



Check /var/log/system.log looks like the time on the panic is after the reboot, so nothing logged prior to panic?:: 


    1020 Aug 19 12:51:02 epsilon login[5175]: USER_PROCESS: 5175 ttys006
    1021 Aug 19 12:56:15 epsilon syslogd[56]: ASL Sender Statistics
    1022 Aug 19 12:59:58 epsilon GoogleSoftwareUpdateAgent[5443]: 2018-08-19 12:59:58.858 GoogleSoftwareUpdateAgent[5443/0x7fff8d0f2380] [lvl=2] -[KSAgentApp(PrivateMethods) setUpLoggerOutputF     orVerboseMode:] Agent default/global settings: <KSAgentSettings:0x7fc96c436120 bundleID=com.google.Keystone.Agent lastCheck=2018-08-19 02:59:12 +0000 lastServerCheck=2018-08-19 02:59:     11 +0000 lastCheckStart=2018-08-19 02:59:09 +0000 checkInterval=18000.000000 uiDisplayInterval=604800.000000 sleepInterval=1800.000000 jitterInterval=900 maxRunInterval=0.000000 isCon     soleUser=1 ticketStorePath=/Users/blyth/Library/Google/GoogleSoftwareUpdate/TicketStore/Keystone.ticketstore runMode=3 daemonUpdateEngineBrokerServiceName=com.google.Keystone.Daemon.U     pdateEngine daemonAdministrationServiceName=com.google.Keystone.Daemon.Administration alwaysPromptForUpdates=0 lastUIDisplayed=(null) alwaysShowStatusItem=0 updateCheckTag=(null) prin     tResults=NO userInitiated=NO>
    1023 Aug 19 13:01:04 localhost bootlog[0]: BOOT_TIME 1534654864 0
    1024 Aug 19 13:01:16 localhost syslogd[56]: Configuration Notice:
    1025     ASL Module "com.apple.cdscheduler" claims selected messages.
    1026     Those messages may not appear in standard system log files or in the ASL database.
    1027 Aug 19 13:01:16 localhost syslogd[56]: Configuration Notice:
    1028     ASL Module "com.apple.install" claims selected messages.
    1029     Those messages may not appear in standard system log files or in the ASL database.


/var/log/wifi.log::

    Sun Aug 19 10:42:45.695 <airportd[162]> _processIPv4Changes: ARP/NDP offloads disabled, not programming the offload
    Sun Aug 19 10:42:45.707 <airportd[162]> _processIPv4Changes: ARP/NDP offloads disabled, not programming the offload
    Sun Aug 19 10:42:51.659 <kernel> Setting BTCoex Config: enable_2G:1, profile_2g:0, enable_5G:1, profile_5G:0
    Sun Aug 19 13:01:16.823 ***Starting Up***
    Sun Aug 19 13:01:16.893 <kernel> IO80211Controller::addSubscriptionForThisReporterFetchedOnTimer() Failed to addSubscription for group Interface p2p0 subgroup Data Packets driver 0x6252c64761f0ccff - data underrun
    Sun Aug 19 13:01:16.893 <kernel> IO80211InterfaceMonitor::configureSubscriptions() failed to add subscription
    Sun Aug 19 13:01:17.064 <kernel>  Creating all peerManager reporters




* :google:`macOS kernel panic com.nvidia.web.NVDAResmanWeb(10.3.1)` 

* https://github.com/lvs1974/NvidiaGraphicsFixup/releases
* https://github.com/acidanthera/WhateverGreen
* https://github.com/acidanthera/Lilu
* https://github.com/acidanthera/WhateverGreen/blob/master/WhateverGreen/kern_ngfx.cpp


NVIDIA GPU driver
-------------------

Aug 18, 2018 : updated 103 to 106 using the NVIDIA panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Did the update in reponse to ~daily kernel panics, that occur randomly. 
Updated::
 
     from 387.10.10.30.103 
     to   387.10.10.30.106 

But following the update got another panic the day after.



387.10.10.10.30.103 2018.4.2 10.13.4(17E199)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://images.nvidia.com/mac/pkg/387/WebDriver-387.10.10.10.30.103.pkg

* http://us.download.nvidia.com/Mac/Quadro_Certified/387.10.10.10.30.103/WebDriver-387.10.10.10.30.103.pkg


There is no working index on NVIDIAs site and third parties tend to 
just give pkg links, so to find information pages 
on mac nvidia display drivers:

* :google:`387.10.10.10.30.103 site:www.nvidia.com`
   
* http://www.nvidia.com/download/driverResults.aspx/133514/en-us

::

    Version:	387.10.10.10.30.103
    Release Date:	2018.4.2
    Operating System:	macOS High Sierra 10.13.4
    CUDA Toolkit:	9.1
    Language:	English (US)
    File Size:	60.95 MB


    New in Release 387.10.10.10.30.103:
    Graphics driver updated for macOS High Sierra 10.13.4 (17E199)
    Contains performance improvements and bug fixes for a wide range of applications.
    Includes NVIDIA Driver Manager preference pane.
    Includes BETA support for iMac and MacBook Pro systems with NVIDIA graphics

    BETA support is for 
    iMac 14,2 / 14,3 (2013), 
    iMac 13,1 / 13,2 (2012) and 
    MacBook Pro 11,3 (2013), 
    MacBook Pro 10,1 (2012), and 
    MacBook Pro 9,1 (2012) users.




List of CUDA Mac drivers
---------------------------

Mal-named link 

* https://www.nvidia.com/object/mac-driver-archive.html

::

    http://www.nvidia.com/object/macosx-cuda-387.178-driver.html
    CUDA Mac Driver 387.178 
    CUDA driver update to support CUDA Toolkit 9.1, macOS 10.13.4 and NVIDIA display driver 378.10.10.10.30.103


    01/25/2018 387.128
    CUDA driver update to support CUDA Toolkit 9.1, macOS 10.13.3 and NVIDIA display driver 378.10.10.10.25.156
 
    Supports all NVIDIA products available on Mac HW.
    Note: Quadro FX for Mac or GeForce for Mac must be installed prior to CUDA 387.128 installation

    12/08/2017 387.99
    CUDA driver update to support CUDA Toolkit 9.0, macOS 10.13.2 and NVIDIA display driver 378.10.10.10.25.102 





NVIDIA pages QUADRO & GEFORCE MACOS DRIVER RELEASE 
---------------------------------------------------

Unfortunately NVIDIA driver list pages dont work for mac, 
so have to resort to web trawling to find the page.

* :google:`QUADRO & GEFORCE MAC site:www.nvidia.com`



378.10.10.10.15.121  2017.10.27
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.nvidia.com/download/driverResults.aspx/126157/en-us

New in Release 378.10.10.10.15.121:
Graphics driver updated for macOS High Sierra 10.13 (17A405)
Contains performance improvements and bug fixes for a wide range of applications.
Includes NVIDIA Driver Manager preference pane.
Includes BETA support for iMac and MacBook Pro systems with NVIDIA graphics


cuda mac 10.13
----------------


* https://devtalk.nvidia.com/default/topic/1025945/cuda-setup-and-installation/mac-cuda-9-0-driver-fully-compatible-with-macos-high-sierra-10-13-error-quot-update-required-quot-solved-/

* https://www.tonymacx86.com/threads/nvidia-releases-alternate-graphics-drivers-for-macos-high-sierra-10-13-3-387-10-10-10-25.243857/



Installing the GPU Driver
-----------------------------

QUADRO & GEFORCE MACOS DRIVER RELEASE 387.10.10.10.25.156 for macOS v10.13.3 (17D47)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MINIMUM SYSTEM REQUIREMENTS for Driver Release 387.10.10.10.25.156

Model identifier should be Mac Pro 5,1 (2010), Mac Pro 4,1 (2009) or Mac Pro 3,1 (2008)
macOS v10.13.3 (17D47)

STEP 1: 
Make sure your macOS software version is v10.13.3 (17D47). It is important that
you check this first before you install the 387.10.10.10.25.156 Driver. Click
on the Apple icon (upper left corner of the screen) and select About This Mac.
Click the Version number ("Version 10.13.3") to see the exact build version
number (17D47).


* http://www.nvidia.com/download/driverResults.aspx/130460/en-us


To uninstall the NVIDIA Web Driver and the NVIDIA Driver Manager, follow the steps below: 

STEP 1: Open the NVIDIA Driver Manager from the System Preferences or through the menu bar item.

STEP 2: Click on the padlock icon and enter an Administrator password.

STEP 3: Click the Open Uninstaller button.

STEP 4: Click Uninstall and then Continue Uninstallation on the Warning screen:
The Warning screen lets you know that you will need to restart your system once
the installation process is complete.

STEP 5: Re-enter an Administrator password and click OK. Once the NVIDIA Web
Driver and NVIDIA Driver Manager have been removed from the system, click
Restart.



NOTE: If for any reason you are unable to boot your system to the Desktop and
wish to restore your original macOS v10.13.3 (17D47) driver, you can do so by
clearing your Mac’s NVRAM: 

STEP 1: Restart your Macintosh computer and simultaneously hold down the
“Command” (apple) key, the “Option” key, the “P” key and the “R” key before the
gray screen appears. 

STEP 2: Keep the keys held down until you hear the startup chime for the second
time. Release the keys and allow the system to boot to the desktop.

STEP 3: The original macOS v10.13.3 (17D47) driver will be restored upon
booting, although the NVIDIA Web Driver and NVIDIA Driver Manager will not be
uninstalled from the system.


How to reset NVRAM on your Mac
--------------------------------

* https://support.apple.com/en-us/HT204063

NVRAM (nonvolatile random-access memory) is a small amount of memory that your
Mac uses to store certain settings and access them quickly. Settings that can
be stored in NVRAM include sound volume, display resolution, startup-disk
selection, time zone, and recent kernel panic information. The settings stored
in NVRAM depend on your Mac and the devices you're using with your Mac.


What settings of NVRAM pertain to the GPU ?
---------------------------------------------

::

    delta:externals blyth$ nvram -p | grep gpu
    gpu-policy  %01



After 9.1 install on epsilon
------------------------------

In *Sys Prefs > CUDA*::

    [Install CUDA Update] greyed out
    No newer CUDA Driver available
    CUDA Driver Version: 387.128   [Update Required] 
    GPU Driver Version: 355.11.10.10.30.120

::

    epsilon:~ blyth$ cuda-osx-kextstat
      152    0 0xffffff7f80d08000 0x2000     0x2000     com.nvidia.CUDA (1.1.0) 4329B052-6C8A-3900-8E83-744487AEDEF1 <4 1>

    ## following a restart
    epsilon:~ blyth$ kextstat | grep -i cuda
      153    0 0xffffff7f81698000 0x2000     0x2000     com.nvidia.CUDA (1.1.0) 4329B052-6C8A-3900-8E83-744487AEDEF1 <4 1>

The kext are in /Library/Extensions not /System/Library/Extensions


Samples Build OK with switch back to xcode-92 : NB dont have the corresponding headers in /usr/include
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    xcode-;xcode-92;xcode-check

    cuda-samples-make    # after dir setup and mving samples from HOME to there



About CUDA driver 387.128
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Searching for the CUDA driver : http://www.nvidia.com/object/macosx-cuda-387.128-driver.html

::

    Version: 387.128
    Release Date: 2018.01.25
    Operating System: Mac OS
    Language: English (U.S.)
    File Size: 39.7 MB

    New Release 387.128

    CUDA driver update to support CUDA Toolkit 9.1, macOS 10.13.3 
    and NVIDIA display driver 378.10.10.10.25.156
    macOS CUDA driver version format change
    The macOS CUDA driver version now uses the format xxx.xx compare to x.x.x
    to be consistent with our Linux and Windows driver version naming convention.

    Recommended CUDA version(s):
          CUDA 9.1
    Supported macOS
         10.13.x
    An alternative method to download the latest CUDA driver is within macOS
    environment.  Access the latest driver through System Preferences > Other >
    CUDA.  Click 'Install CUDA Update'



CUDA Driver
~~~~~~~~~~~~~

::

    Version:	387.10.10.10.25.156
    Release Date:	2018.1.24
    Operating System:	macOS High Sierra 10.13.3
    CUDA Toolkit:	9.1
    Language:	English (US)
    File Size:	60.95 MB


*CUDA Application Support:*

In order to run macOS Applications that leverage the CUDA architecture of
certain NVIDIA graphics cards, users will need to download and install the
driver for Mac located here.

*Installation Note:*

Because of improvements in macOS security, the Security & Privacy Preferences
may open during the installation process. If it does, click “Allow” in order
for the NVIDIA Graphics Driver to load, then return to the Installer and click
“Restart”.

*New in Release 387.10.10.10.25.156:*

* Graphics driver updated for macOS High Sierra 10.13.3 (17D47)
* Contains performance improvements and bug fixes for a wide range of applications.
* Includes NVIDIA Driver Manager preference pane.
* Includes BETA support for iMac and MacBook Pro systems with NVIDIA graphics

*Release Notes Archive:*

This driver update is for::

    Mac Pro 5,1 (2010), 
    Mac Pro 4,1 (2009) and 
    Mac Pro 3,1 (2008) users.

BETA support is for::

    iMac 14,2 / 14,3 (2013), 
    iMac 13,1 / 13,2 (2012) and
    MacBook Pro 11,3 (2013),       <<<<< 
    MacBook Pro 10,1 (2012), and 
    MacBook Pro  9,1 (2012) users.

Installing the Driver
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:~ blyth$ l ~/Downloads/WebDriver-387.10.10.10.25.158.pkg
    -rw-r--r--@ 1 blyth  staff  63913283 Mar 31 18:51 /Users/blyth/Downloads/WebDriver-387.10.10.10.25.158.pkg

    ## thus one said 10.13.4 not compatible 
       
    epsilon:~ blyth$ l ~/Downloads/WebDriver-387.10.10.10.30.103.pkg 
    -rw-r--r--@ 1 blyth  staff  63913281 Mar 31 18:55 /Users/blyth/Downloads/WebDriver-387.10.10.10.30.103.pkg


Thank you for purchasing an NVIDIA® graphics card for Mac. 

This software package contains:

* A driver designed for your NVIDIA graphics card.
* A preference pane in System Preferences that includes options to:
  
  * Switch between the NVIDIA Web Graphics Driver or Default macOS Graphics Driver.
  * Enable Error Correcting Codes (ECC) on supported graphics cards.
  * Automatically check for NVIDIA software updates for your graphics card.
  * An optional menu bar item that provides easy access to the most common preferences.

If, for any reason, you are unable to boot your system to the desktop and wish
to restore your original drivers, you can do so by clearing your Mac’s NVRAM.
To clear NVRAM, power on your Mac and hold down Command (⌘), Option, P, and R
simultaneously before the gray screen appears. Keep the keys held down until
you hear the startup chime for the second time. The system will now boot with
the original drivers restored.

This will take : 196.7 MB 

Installed OK, but still::

    epsilon:~ blyth$ cuda-deviceQuery 
    running /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/bin/x86_64/darwin/release/deviceQuery
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/bin/x86_64/darwin/release/deviceQuery Starting...

     CUDA Device Query (Runtime API) version (CUDART static linking)

    cudaGetDeviceCount returned 35
    -> CUDA driver version is insufficient for CUDA runtime version
    Result = FAIL
    epsilon:~ blyth$ 
    epsilon:~ blyth$ 




OptiX 5.0.1 Release Notes Recommends CUDA 9.0
--------------------------------------------------

* https://developer.nvidia.com/cuda-90-download-archive

CUDA 9.0 Mac
~~~~~~~~~~~~~~

* http://developer.download.nvidia.com/compute/cuda/9.0/Prod/docs/sidebar/CUDA_Installation_Guide_Mac.pdf

 
To use CUDA on your system, you need to have:

* a CUDA-capable GPU
* Mac OS X 10.12
* the Clang compiler and toolchain installed using Xcode

::

   Xcode 8.3.3 
   Apple LLVM 8.1.0   
   macOS 10.12      (Sierra)


CUDA 9.1 Mac
~~~~~~~~~~~~~~

* http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

Driver support for Mac OS 10.13.3 is added in this release.
Xcode 9.2 is now supported as a host compiler on Mac OS.

* https://developer.nvidia.com/cuda-downloads?target_os=MacOSX&target_arch=x86_64&target_version=1013&target_type=dmgnetwork
* https://developer.download.nvidia.com/compute/cuda/9.1/Prod/docs/sidebar/CUDA_Installation_Guide_Mac.pdf

1.1. System Requirements

To use CUDA on your system, you need to have:

* a CUDA-capable GPU
* Mac OS X 10.13
* the Clang compiler and toolchain installed using Xcode
* the NVIDIA CUDA Toolkit (available from the CUDA Download page)

::

    Xcode 9.2
    Apple LLVM  9.0.0
    macOS 10.13.2   (High Sierra)

A supported version of Xcode must be installed on your system. The list of supported
Xcode versions can be found in the System Requirements section. The latest version of
Xcode can be installed from the Mac App Store.
Older versions of Xcode can be downloaded from the Apple Developer Download Page.
Once downloaded, the Xcode.app folder should be copied to a version-specific folder
within /Applications. For example, Xcode 6.2 could be copied to /Applications/
Xcode_6.2.app.
Once an older version of Xcode is installed, it can be selected for use by running the
following command, replacing <Xcode_install_dir> with the path that you copied
that version of Xcode to::

    sudo xcode-select -s /Applications/<Xcode_install_dir>/Contents/Developer


Installer::

  CUDA Driver  (kext)
  CUDA Toolkit
  CUDA Samples







Curious about how graphics drivers work
-----------------------------------------

* https://people.freedesktop.org/~marcheu/linuxgraphicsdrivers.pdf


CUDA.kext
-----------

::

    delta:home blyth$ ls -lt  /System/Library/Extensions/
    total 0
    drwxr-xr-x  3 root  wheel  102 Mar  6  2015 CUDA.kext
    drwxr-xr-x  3 root  wheel  102 Jan 10  2015 AppleUSBEthernetHost.kext
    drwxr-xr-x  6 root  wheel  204 Jul 14  2014 System.kext
    drwxr-xr-x  6 root  wheel  204 Jul 14  2014 AppleBacklightExpert.kext
    ...

::

    delta:home blyth$ file /System/Library/Extensions/CUDA.kext/Contents/MacOS/CUDA 
    /System/Library/Extensions/CUDA.kext/Contents/MacOS/CUDA: Mach-O universal binary with 1 architecture
    /System/Library/Extensions/CUDA.kext/Contents/MacOS/CUDA (for architecture x86_64): Mach-O 64-bit kext bundle x86_64


List loaded kexts::

    delta:~ blyth$ kextstat -k | grep nvidia
      100    2 0xffffff7f80bb4000 0x274000   0x274000   com.apple.nvidia.driver.NVDAResman (8.2.6) <85 77 75 11 5 4 3 1>
      101    0 0xffffff7f80e33000 0x1ad000   0x1ad000   com.apple.nvidia.driver.NVDAGK100Hal (8.2.6) <100 11 4 3>
      124    0 0xffffff7f81d26000 0x2000     0x2000     com.nvidia.CUDA (1.1.0) <4 1>

* switching off *Energy Saver > Auto Graphics Switching* made no difference

::

    delta:~ blyth$ cat /System/Library/Extensions/CUDA.kext/Contents/Info.plist 
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
      ..
      <key>CFBundleIdentifier</key>
      <string>com.nvidia.CUDA</string>
      <key>CFBundleInfoDictionaryVersion</key>
      <string>6.0</string>
      <key>CFBundleName</key>
      <string>CUDA</string>
      <key>CFBundlePackageType</key>
      <string>KEXT</string>
      <key>CFBundleVersion</key>
      <string>1.1.0</string>
      <key>OSBundleLibraries</key>
      <dict>
        <key>com.apple.kpi.bsd</key>
        <string>8.0.0</string>
        <key>com.apple.kpi.libkern</key>
        <string>8.0.0</string>
        </dict>
    </dict>
    </plist>
    delta:~ blyth$ 




*System Preferences > CUDA* says::

    CUDA 7.5.30 Driver update is available

    CUDA Driver Version: 7.0.29
    GPU Driver Version: 8.26.26 310.40.45f01


Accessing this while running opensnoop, shows access to installed CUDA.framework Info.plist::

    delta:~ blyth$ sudo opensnoop -vg 
    Password:
    STRTIME                UID    PID  FD PATH                 ARGS
    2018 Mar 26 19:11:32     0     12  -1 //private/var/run/installd.commit.pid kextd\0
    ..
    2018 Mar 26 19:11:40   501   7168  10 /Library/Frameworks/CUDA.framework/Resources/Info.plist System Preferenc\0
    2018 Mar 26 19:11:40   501   7168  10 /Library/PreferencePanes/CUDA Preferences.prefPane System Preferenc\0
    2018 Mar 26 19:11:40   501   7168  10 /Users/blyth/Library/Keychains/login.keychain System Preferenc\0
    2018 Mar 26 19:11:40   501   7168  10 /Library/Keychains/System.keychain System Preferenc\0
    2018 Mar 26 19:11:40     0     67  12 /Library/PreferencePanes/CUDA Preferences.prefPane mds\0
    2018 Mar 26 19:11:40   501   7168  10 /Library/Frameworks/CUDA.framework/Resources/Info.plist System Preferenc\0
    2018 Mar 26 19:11:40   501   7168  10 /System/Library/CoreServices/SystemVersion.plist System Preferenc\0
    ^C

::

    delta:~ blyth$ cat /Library/Frameworks/CUDA.framework/Resources/Info.plist
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        ...
        <key>CFBundleIdentifier</key>
        <string>com.nvidia.CUDA</string>
        <key>CFBundleInfoDictionaryVersion</key>
        <string>6.0</string>
        <key>CFBundleName</key>
        <string>CUDA</string>
        <key>CFBundlePackageType</key>
        <string>FMWK</string>
        <key>CFBundleShortVersionString</key>
        <string>7.0.29</string>
        ...
        <key>CFBundleVersion</key>
        <string>7.0.29</string>
        <key>MaxSupportedNVDAResmanVersion</key>
        <string>34302</string>
        <key>NSHumanReadableCopyright</key>
        <string>© NVIDIA, 2015</string>
        <key>NVDAResmanVersions</key>
        <dict>
            <key>310.40.05</key>
            <string>310.40.05</string>
            <key>310.40.15</key>
            <string>310.40.15</string>
            <key>310.40.25</key>
            ...





EOU
}
cudamac-dir(){ echo $(local-base)/home/sysadmin/sysadmin-cudamac ; }
cudamac-cd(){  cd $(cudamac-dir); }
cudamac-mate(){ mate $(cudamac-dir) ; }
cudamac-get(){
   local dir=$(dirname $(cudamac-dir)) &&  mkdir -p $dir && cd $dir

}
