cudalin-source(){   echo $BASH_SOURCE; }
cudalin-vi(){       vi $(cudalin-source) ; }
cudalin-env(){      olocal- ; }
cudalin-usage(){ cat << \EOU

CUDA on Linux : Version specific notes 
=========================================

See cuda- for more general info.

See Also
----------

cuda- 
   general CUDA notes
cudamac-
   version specifics on macOS


Overview on version dependencies
---------------------------------

Versions need to be carefully aligned, driven 
by OptiX, the release notes of which identify the
development CUDA version used and the minimum 
GPU driver version.  The CUDA version release notes
identify the minimum Linux kernel version.

* OptiX version 

  * GPU driver version (kernel extension, forcing stringent version alignment)

    * Linux kernel version 

  * CUDA version

    * CUDA driver version
    * Linux kernel version


nvidia display driver (aka GPU driver)
----------------------------------------

The GPU driver is normally provided by the vendor, 
BUT the CUDA driver requires a newer GPU driver than the old one
provided by the vendor/distribution.


Curious about how graphics drivers work
-----------------------------------------

* https://people.freedesktop.org/~marcheu/linuxgraphicsdrivers.pdf



GPU Drivers
-------------


Linux x64 (AMD64/EM64T) Display Driver : 418.43
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.nvidia.com/Download/driverResults.aspx/142958/en-us
 
Version:    418.43
Release Date:   2019.2.22
Operating System:   Linux 64-bit
Language:   English (US)
File Size:  101.71 MB
    





EOU
}

