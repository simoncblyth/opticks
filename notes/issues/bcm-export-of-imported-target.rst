bcm-export-of-imported-target
===============================


Context
---------

* :doc:`packaging-opticks-and-externals-for-use-on-gpu-cluster`

* bcm-;bcm-vi

* :google:`cmake export an imported target`


NEXT
--------

* provide the start of an opticks-config.py interface, so 
  can start non-CMake building with some Makefiles perhaps 
  within subdirectories of the examples/Use* 

  * need to unify the parsed config from:

    1. real CMake exported targets
    2. TOPMETA fallback for non-native CMake imported targets


Issue
---------

Trying to export found targets, in order to have all targets in 
a standard form for easy parsing with bcm.py.


sysrap/CMakeLists.txt::

    141 list(REMOVE_DUPLICATES OPTICKS_TARGETS_FOUND)
    142 message(STATUS " OPTICKS_TARGETS_FOUND : ${OPTICKS_TARGETS_FOUND} " )
    143 bcm_deploy(TARGETS ${OPTICKS_TARGETS_FOUND} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
    144 



Try just accessing the properties and dumping them
---------------------------------------------------

* :google:`kitware CMake INTERFACE_LIBRARY targets may only have whitelisted properties`
* https://gitlab.kitware.com/cmake/cmake/issues/18463
* possibly a problem from 3.13 series only 

::

    [blyth@lxslc701 sysrap]$ cmake3 --version
    cmake3 version 3.13.4



::

    tgt='Opticks::PLog' prop='INTERFACE_INCLUDE_DIRECTORIES' defined='0' set='1' value='/hpcfs/juno/junogpu/blyth/local/opticks/externals/plog/include' 

    tgt='Opticks::PLog' prop='INTERFACE_BLAH' defined='0' set='1' value='Just testing' 

    CMake Error at /hpcfs/juno/junogpu/blyth/opticks/cmake/Modules/EchoTarget.cmake:13 (get_property):
      INTERFACE_LIBRARY targets may only have whitelisted properties.  The
      property "IMPORTED_LOCATION" is not allowed.
    Call Stack (most recent call first):
      /hpcfs/juno/junogpu/blyth/opticks/cmake/Modules/EchoTarget.cmake:39 (echo_target_property)
      CMakeLists.txt:159 (echo_target)



Try adding ini format TOPMETA to the TOPMATTER
-------------------------------------------------

* added cmake/Modulues/TopMetaTarget.cmake top_meta_target which collects 
  metadata on targets passed as arguments 

* including this metadata into the bcm_deploy TOPMATTER

* bin/bcm.py parses this using ConfigParser


Problems
~~~~~~~~~~~~~~~~~

1. get_property refuses for non-whitelisted properties, workaround by duplicating some with "INTERFACE_" prefix 
2. with boost are using the CMake internal finding, so cannot workaround the non-whitelisting easily 




Issue : boost has horrible FindBoost 
----------------------------------------


Issue : YoctoGL is CMake installed directly into Opticks prefix
---------------------------------------------------------------------

* thats my fork of YoctoGL on github

::

   oyoctogl-c  # to see my CMakeLists.txt which uses BCM already 


::

    oyoctogl--
    ...

    -- Up-to-date: /home/blyth/local/opticks/externals/lib/libYoctoGL.so
    -- Up-to-date: /home/blyth/local/opticks/externals/lib/pkgconfig/yoctogl.pc
    -- Up-to-date: /home/blyth/local/opticks/externals/lib/cmake/yoctogl/properties-yoctogl-targets.cmake
    -- Up-to-date: /home/blyth/local/opticks/externals/lib/cmake/yoctogl/yoctogl-targets.cmake
    -- Installing: /home/blyth/local/opticks/externals/lib/cmake/yoctogl/yoctogl-targets-debug.cmake
    -- Up-to-date: /home/blyth/local/opticks/externals/lib/cmake/yoctogl/yoctogl-config.cmake
    -- Up-to-date: /home/blyth/local/opticks/externals/lib/cmake/yoctogl/yoctogl-config-version.cmake
    -- Installing: /home/blyth/local/opticks/lib/ygltf_reader
    -- Set runtime path of "/home/blyth/local/opticks/lib/ygltf_reader" to "/home/blyth/opticks/lib64:/home/blyth/opticks/externals/lib:/home/blyth/opticks/externals/lib64:/home/blyth/opticks/externals/OptiX/lib64"
    (base) [blyth@gilda03 UseYoctoGL]$ 
    (base) [blyth@gilda03 UseYoctoGL]$ 


This is already in the release.

::

    (base) [blyth@gilda03 opticks]$ bcm.py -t yoctogl -p
    yoctogl :  :  
    /home/blyth/local/opticks/externals/lib/cmake/yoctogl/yoctogl-config.cmake
    (base) [blyth@gilda03 opticks]$ vi /home/blyth/local/opticks/externals/lib/cmake/yoctogl/yoctogl-config.cmake   ## no TOPMETA yet 

::

    (base) [blyth@gilda03 opticks]$ bcm.py -t assimp -p
    assimp :  :  
    /home/blyth/local/opticks/externals/lib/cmake/assimp-3.1/assimp-config.cmake           ## non-BCM targets 












