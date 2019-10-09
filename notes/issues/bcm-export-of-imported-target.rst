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









Take a look at CMake generated pkgconfig
---------------------------------------------

* not really expecting this to be a complete solution, but can learn from it 

::

    (base) [blyth@gilda03 pkgconfig]$ PKG_CONFIG_PATH=. pkg-config sysrap --libs
    Empty package name in Requires or Conflicts in file './sysrap.pc'


    (base) [blyth@gilda03 pkgconfig]$ l
    total 84
    -rw-r--r--. 1 blyth blyth 270 Sep 13 22:57 g4ok.pc
    -rw-r--r--. 1 blyth blyth 262 Sep 13 22:56 okg4.pc
    -rw-r--r--. 1 blyth blyth 281 Sep 13 22:56 cfg4.pc
    -rw-r--r--. 1 blyth blyth 260 Sep 13 22:56 extg4.pc
    -rw-r--r--. 1 blyth blyth 272 Sep 13 22:56 ok.pc
    -rw-r--r--. 1 blyth blyth 281 Sep 13 22:56 opticksgl.pc
    -rw-r--r--. 1 blyth blyth 284 Sep 13 22:56 oglrap.pc
    -rw-r--r--. 1 blyth blyth 263 Sep 13 22:55 okop.pc
    -rw-r--r--. 1 blyth blyth 295 Sep 13 22:55 optixrap.pc
    -rw-r--r--. 1 blyth blyth 288 Sep 13 22:54 optickscore.pc
    -rw-r--r--. 1 blyth blyth 236 Sep 13 22:50 okconf.pc
    -rw-r--r--. 1 blyth blyth 373 Sep  9 21:08 thrustrap.pc
    -rw-r--r--. 1 blyth blyth 368 Sep  9 21:08 cudarap.pc
    -rw-r--r--. 1 blyth blyth 307 Sep  9 21:08 opticksgeo.pc
    -rw-r--r--. 1 blyth blyth 295 Sep  9 21:08 openmeshrap.pc
    -rw-r--r--. 1 blyth blyth 276 Sep  9 21:08 assimprap.pc
    -rw-r--r--. 1 blyth blyth 277 Sep  9 21:03 ggeo.pc
    -rw-r--r--. 1 blyth blyth 275 Sep  9 21:02 yoctoglrap.pc
    -rw-r--r--. 1 blyth blyth 369 Sep  9 21:02 npy.pc
    -rw-r--r--. 1 blyth blyth 286 Sep  9 21:02 boostrap.pc
    -rw-r--r--. 1 blyth blyth 282 Sep  9 21:02 sysrap.pc
    (base) [blyth@gilda03 pkgconfig]$ pwd
    /home/blyth/local/opticks/lib64/pkgconfig
    (base) [blyth@gilda03 pkgconfig]$ cd
    (base) [blyth@gilda03 ~]$ 
    (base) [blyth@gilda03 ~]$ 
    (base) [blyth@gilda03 ~]$ 
    (base) [blyth@gilda03 ~]$ PKG_CONFIG_PATH=/home/blyth/local/opticks/lib64/pkgconfig pkg-config sysrap --libs
    Empty package name in Requires or Conflicts in file '/home/blyth/local/opticks/lib64/pkgconfig/sysrap.pc'
    (base) [blyth@gilda03 ~]$ 



::

     01 
      2 prefix=/home/blyth/local/opticks
      3 exec_prefix=${prefix}
      4 libdir=${exec_prefix}/lib64
      5 includedir=${exec_prefix}/include/SysRap
      6 Name: sysrap
      7 Description: No description
      8 Version: 0.1.0
      9 
     10 Cflags:  -I${includedir} -DOPTICKS_SYSRAP
     11 Libs: -L${libdir}  -lSysRap ssl crypto
     12 Requires: okconf,,


Removing the ",,"::

    (base) [blyth@gilda03 pkgconfig]$ PKG_CONFIG_PATH=/home/blyth/local/opticks/lib64/pkgconfig pkg-config sysrap --libs
    ssl crypto -L/home/blyth/local/opticks/lib64 -lSysRap  


::

    (base) [blyth@gilda03 pkgconfig]$ grep "Requires:" *.pc
    assimprap.pc:Requires: ggeo
    boostrap.pc:Requires: sysrap,,
    cfg4.pc:Requires: extg4,opticksgeo,thrustrap
    cudarap.pc:Requires: ,,,,sysrap,okconf,
    extg4.pc:Requires: ggeo
    g4ok.pc:Requires: cfg4,extg4,okop
    ggeo.pc:Requires: optickscore,yoctoglrap
    npy.pc:Requires: sysrap,boostrap,yoctogl,dualcontouringsample
    oglrap.pc:Requires: opticksgeo,
    okconf.pc:Requires: 
    okg4.pc:Requires: ok,cfg4
    okop.pc:Requires: optixrap
    ok.pc:Requires: opticksgl
    openmeshrap.pc:Requires: optickscore,ggeo
    optickscore.pc:Requires: npy,okconf
    opticksgeo.pc:Requires: optickscore,assimprap,openmeshrap
    opticksgl.pc:Requires: oglrap,okop
    optixrap.pc:Requires: okconf,opticksgeo,thrustrap
    sysrap.pc:Requires: okconf
    thrustrap.pc:Requires: ,,,,optickscore,cudarap
    yoctoglrap.pc:Requires: npy
    (base) [blyth@gilda03 pkgconfig]$ 



* :google:`"Empty package name in Requires or Conflicts in file"`

BCM is generating these files::
   
    bcm- 
    bcm-cd
    vi share/bcm/cmake/BCMPkgConfig.cmake

    19 function(bcm_generate_pkgconfig_file)
    20     set(options)
    21     set(oneValueArgs NAME LIB_DIR INCLUDE_DIR DESCRIPTION)
    22     set(multiValueArgs TARGETS CFLAGS LIBS REQUIRES)
    23 


::

    (base) [blyth@gilda03 cmake]$ grep auto *.cmake
    BCMDeploy.cmake:    bcm_auto_pkgconfig(TARGET ${PARSE_TARGETS})
    BCMDeploy.cmake:    bcm_auto_export(TARGETS ${PARSE_TARGETS} NAMESPACE ${PARSE_NAMESPACE} COMPATIBILITY ${PARSE_COMPATIBILITY} TOPMATTER ${PARSE_TOPMATTER})
    BCMExport.cmake:function(bcm_auto_export)
    BCMPkgConfig.cmake:function(bcm_auto_pkgconfig_each)
    BCMPkgConfig.cmake:        message(SEND_ERROR "Target is required for auto pkg config")
    BCMPkgConfig.cmake:function(bcm_auto_pkgconfig)
    BCMPkgConfig.cmake:        bcm_auto_pkgconfig_each(TARGET ${PARSE_TARGET} NAME ${PARSE_NAME})
    BCMPkgConfig.cmake:            bcm_auto_pkgconfig_each(TARGET ${TARGET} NAME ${TARGET})
    (base) [blyth@gilda03 cmake]$ 


::

    204 function(bcm_auto_pkgconfig)
    205     set(options)
    206     set(oneValueArgs NAME)
    207     set(multiValueArgs TARGET) # TODO: Rename to TARGETS
    208 
    209     cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    210 
    211     list(LENGTH PARSE_TARGET TARGET_COUNT)
    212 
    213     if(TARGET_COUNT EQUAL 1)
    214         bcm_auto_pkgconfig_each(TARGET ${PARSE_TARGET} NAME ${PARSE_NAME})
    215     else()
    216         string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
    217         set(PACKAGE_NAME ${PROJECT_NAME})
    218 
    219         if(PARSE_NAME)
    220             set(PACKAGE_NAME ${PARSE_NAME})
    221         endif()
    222 
    223         string(TOLOWER ${PACKAGE_NAME} PACKAGE_NAME_LOWER)
    224 
    225         set(GENERATE_PROJECT_PC On)
    226         foreach(TARGET ${PARSE_TARGET})
    227             if("${TARGET}" STREQUAL "${PACKAGE_NAME_LOWER}")
    228                 set(GENERATE_PROJECT_PC Off)
    229             endif()
    230             bcm_auto_pkgconfig_each(TARGET ${TARGET} NAME ${TARGET})
    231         endforeach()
    232 
    233         string(REPLACE ";" "," REQUIRES "${PARSE_TARGET}")
    ^^^^^^^^  REQUIRES is output with ; -> ,
    234         # TODO: Get description from project
    235         set(DESCRIPTION "No description")
    236 
    237         if(GENERATE_PROJECT_PC)
    238             file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME_LOWER}.pc CONTENT
    239 "
    240 Name: ${PACKAGE_NAME_LOWER}
    241 Description: ${DESCRIPTION}
    242 Version: ${PROJECT_VERSION}
    243 Requires: ${REQUIRES}
    244 "
    245             )
    246         endif()
    247     endif()




