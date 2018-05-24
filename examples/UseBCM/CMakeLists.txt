cmake_minimum_required(VERSION 3.5)

set(name UseBCM)

project(${name})

find_package(BCM)


message( STATUS "BCM_FOUND  : ${BCM_FOUND} ")
message( STATUS "BCM_CONFIG : ${BCM_CONFIG} ")

#[=[

Initially 
   /usr/local/opticks-cmake-overhaul/share/bcm/cmake/BCMConfig.cmake 

But after changing bcm-prefix now install into externals 
    /usr/local/opticks-cmake-overhaul/externals/share/bcm/cmake/BCMConfig.cmake 

Its more logical to place BCM in PREFIX/externals/share not in PREFIX/share 
as for opticks variants the typical action is to symbolic link 

::

    epsilon:opticks-cmake-overhaul blyth$ l
    total 8
    drwxr-xr-x  407 blyth  staff  - 13024 May 24 15:12 lib
    drwxr-xr-x   36 blyth  staff  -  1152 May 24 15:12 build
    drwxr-xr-x    5 blyth  staff  -   160 May 24 13:44 installcache
    drwxr-xr-x   28 blyth  staff  -   896 May 23 22:30 include
    drwxr-xr-x   20 blyth  staff  -   640 May 23 18:38 gl
    drwxr-xr-x    3 blyth  staff  -    96 May 21 21:18 geocache
    -rw-r--r--    1 blyth  staff  -    94 May 21 16:06 OpticksIDPATH.log
    lrwxr-xr-x    1 blyth  staff  -    22 May 21 16:06 opticksdata -> ../opticks/opticksdata
    drwxr-xr-x    7 blyth  staff  -   224 May 15 19:13 config
    lrwxr-xr-x    1 blyth  staff  -    20 May 15 16:31 externals -> ../opticks/externals
    epsilon:opticks-cmake-overhaul blyth$     



So it gets found via 
    CMAKE_PREFIX_PATH=$(opticks-prefix)/externals  
rather than 
    CMAKE_INSTALL_PATH=$(opticks-prefix)


#]=]

