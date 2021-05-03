cmake_minimum_required(VERSION 3.5)
set(name UseBCM)
project(${name} VERSION 0.1.0)

find_package(BCM)

message( STATUS "BCM_FOUND  : ${BCM_FOUND} ")
message( STATUS "BCM_CONFIG : ${BCM_CONFIG} ")

#[=[

BCMConfig.cmake::

    epsilon:UseBCM blyth$ find /usr/local/opticks -name BCMConfig.cmake
    /usr/local/opticks/externals/bcm/bcm/share/bcm/cmake/BCMConfig.cmake
    /usr/local/opticks/externals/share/bcm/cmake/BCMConfig.cmake

So BCM gets found via::

    CMAKE_PREFIX_PATH=$(opticks-prefix)/externals  

rather than::

    CMAKE_INSTALL_PATH=$(opticks-prefix)


#]=]

