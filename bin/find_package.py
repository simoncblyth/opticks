#!/usr/bin/env python
"""
find_package.py
=================

Following the simple heuristic of looking for *Config.cmake or *-config.cmake 
in directories of the CMAKE_PREFIX_PATH envvar attempt to 
predict which package CMake will find without asking CMake.

::

   unset CMAKE_PREFIX_PATH
   export CMAKE_PREFIX_PATH=$(opticks-prefix):${CMAKE_PREFIX_PATH}:$(opticks-prefix)/externals
   find_package.py 

   CMAKE_PREFIX_PATH=$(opticks-prefix):$(opticks-prefix)/externals find_package.py 
   CMAKE_PREFIX_PATH=$(opticks-prefix) find_package.py 
   CMAKE_PREFIX_PATH=$(opticks-prefix)/externals find_package.py 
   # NB do not include the "lib" in the prefix

::

   find_package.py Boost --first --libdir


NB this is only useful if it can be done simply and is usually correct, 
otherwise might as well use CMake by generating a CMakeLists.txt 
and get the definitive answer by parsing CMake tealeaves.

Thoughts
---------

Do not want to rely on this script at first order as it is adding 
another resolver to the already two that need to be matched.  
Better to arrange that the two resolvers (CMake and pkg-config) 
yield matched results by making sure that PKG_CONFIG_PATH is set appropriately 
based on CMAKE_PREFIX_PATH.  junotop/bashrc.sh will do this so long as
the directories exist.

BUT : Boost and Geant4 lack lib/pkgconfig/name.pc files, xerces-c has one

So this script can have the role of fixing these omitted pc 
as a workaround until the Boost and Geant4 installs do 
the correct thing themselves. Do not hold breath it has been 
years since people have been asking for this. 


"""
from findpkg import Main

if __name__ == '__main__':
    Main(default_mode="cmake")


