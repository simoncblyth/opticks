cmake_target_link_libraries_for_imported_target
==================================================


Issue reported by Elias 
-----------------------

However, I am still finding the error where it is unable to link
GLFW. In the following code::

   OpticksGLFW_FOUND AND NOT TARGET Opticks::OpticksGLFW

resolves to TRUE, so I know it can find the library,
but for some reason the function call to create the target fails. 
This is the last line of the script:: 

   target_link_libraries(${tgt} INTERFACE GL)

Also the error is:: 
 
    CMake Error at /root/razOpticks/opticks/cmake/Modules/FindOpticksGLFW.cmake:50
    (target_link_libraries): Cannot specify link libraries for target "Opticks::OpticksGLFW" which is not built by this project.


Observations
-------------

Comparing "cmake/Modules/FindOpticksGLFW.cmake" with others in "cmake/Modules"
shows that this one is exceptional in that it uses *target_link_libraries* 
on an imported target, whereas the others just set properties on the target.  
It is possible that this might be exploiting a CMake bug, making it not work 
with some CMake versions : resulting in the issue for Elias. 

To replace the *target_link_libraries* with a *set_target_properties* need to
find the appropriate property names, perhaps



IMPORTED_LINK_INTERFACE_LIBRARIES is deprecated use INTERFACE_LINK_LIBRARIES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    set_target_properties(LibD
      PROPERTIES
        IMPORTED_LINK_INTERFACE_LIBRARIES "LibA;LibB"
    )


* https://cmake.org/cmake/help/v3.0/prop_tgt/IMPORTED_LINK_INTERFACE_LIBRARIES.html


INTERFACE_LINK_LIBRARIES
~~~~~~~~~~~~~~~~~~~~~~~~~

* https://cmake.org/cmake/help/v3.3/prop_tgt/INTERFACE_LINK_LIBRARIES.html

* :google:`CMake INTERFACE_LINK_LIBRARIES imported target`



interesting cloning of a target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/conan-io/conan/issues/2125



perhaps not a bug, a very recent feature ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://gitlab.kitware.com/cmake/cmake/merge_requests/1264

The commands target_compile_definitions, target_compile_features,
target_compile_options, target_include_directories, target_link_libraries and
target_sources can now be used with IMPORTED libraries.

ALIASing an IMPORTED library is now possible, too, as long as the aliased
IMPORTED target is globally visible (aka was created with option GLOBAL).

Thereby, there is no longer any difference in behavior between normal INTERFACE
targets and IMPORTED INTERFACE targets. Only the behavior of IMPORTED INTERFACE
targets that are non-GLOBAL differs slightly: they still cannot be aliased.

This merge-request fixes #15689 (closed), #15569 (closed) and #17197 (closed).


* https://gitlab.kitware.com/cmake/cmake/issues/17197

* https://stackoverflow.com/questions/38534215/cmake-include-dependencies-of-imported-library-without-linking




Reference
----------

Imported targets

* https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/Exporting-and-Importing-Targets

* :google:`target_link_libraries link libraries vs set_target_properties`

* https://cmake.org/cmake/help/v3.0/manual/cmake-buildsystem.7.html#transitive-usage-requirements

* http://cmake.3232098.n2.nabble.com/Using-SET-TARGET-PROPERTIES-and-IMPORTED-LINK-INTERFACE-LIBRARIES-td7596792.html



cmake/Modules/FindOpticksGLFW.cmake
-------------------------------------
 
Also notably, APPLE resolves to FALSE, is that a problem?::

     29 if(OpticksGLFW_FOUND AND NOT TARGET Opticks::OpticksGLFW)
     30     set(tgt Opticks::OpticksGLFW)
     31     add_library(${tgt} UNKNOWN IMPORTED)
     32     set_target_properties(${tgt} PROPERTIES IMPORTED_LOCATION "${OpticksGLFW_LIBRARY}")
     33 
     34     if(APPLE)
     35        find_library( Cocoa_FRAMEWORK NAMES Cocoa )
     36        find_library( OpenGL_FRAMEWORK NAMES OpenGL )
     37        find_library( IOKit_FRAMEWORK NAMES IOKit )
     38        find_library( CoreFoundation_FRAMEWORK NAMES CoreFoundation )
     39        find_library( CoreVideo_FRAMEWORK NAMES CoreVideo )
     40 
     41        ## NB cannot just use "-framework Cocoa" etc, theres some secret distinguishing frameworks apparently 
     42        target_link_libraries(${tgt} INTERFACE
     43            ${Cocoa_FRAMEWORK}
     44            ${OpenGL_FRAMEWORK}
     45            ${IOKit_FRAMEWORK}
     46            ${CoreFoundation_FRAMEWORK}
     47            ${CoreVideo_FRAMEWORK}
     48       )
     49     else()
     50        target_link_libraries(${tgt} INTERFACE GL)
     51 
     52     endif()
     53 
     54     set_target_properties(${tgt} PROPERTIES
     55         INTERFACE_INCLUDE_DIRECTORIES "${OpticksGLFW_INCLUDE_DIR}"
     56         INTERFACE_FIND_PACKAGE_NAME "OpticksGLFW MODULE REQUIRED"
     57     )
     58 
     59     ## Above target_properties INTERFACE_FIND_PACKAGE_NAME kludge tees up the arguments 
     60     ## to find_dependency in BCM generated exports 
     61     ## so downstream targets will automatically do the required find_dependency
     62     ## and call this script again to revive the targets.
     63     ## NB INTERFACE_FIND_PACKAGE_NAME is a BCM defined property, not a standard one, see bcm-
     64 
     65 endif()



