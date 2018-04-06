cudamac-source(){   echo $BASH_SOURCE; }
cudamac-vi(){       vi $(cudamac-source) ; }
cudamac-env(){      olocal- ; }
cudamac-usage(){ cat << \EOU

CUDA on Mac
==============

Guide
--------

* http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html



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


nvidia display driver (aka GPU driver)
----------------------------------------

The GPU driver is normally provided by the vendor (apple) 
as typically you do not update GPUs in Macs.  BUT the CUDA
driver requires a newer GPU driver than the old one
provided by apple.

GPU drivers are kernel extensions (forcing caution), and requiring a 
precise match between the GPU driver and macOS version+build number.   


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


NVIDIA GPU driver
-------------------

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
