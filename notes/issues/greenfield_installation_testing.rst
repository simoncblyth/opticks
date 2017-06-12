Greenfield Installation Testing
==================================


Checking Green Field Opticks Installation
--------------------------------------------

* add to ~/.bash_profile a temporary envvar, export OPTICKS_GREENFIELD_TEST=1

This changes the result of opticks-prefix, so can test geenfield building 
into a day stamped folder.


::

    simon:~ blyth$ opticks-prefix
    /usr/local/opticks
    simon:~ blyth$ OPTICKS_GREENFIELD_TEST= opticks-prefix
    /usr/local/opticks
    simon:~ blyth$ OPTICKS_GREENFIELD_TEST=1 opticks-prefix
    /usr/local/opticks20170612



BUT CMake has some hardcoded paths ??
----------------------------------------

Some opticks/cmake/Modules/ were using `$ENV`, for "official" opticks
externals : should not depend on environment in this way...


Tests using optional externals, need detection of the externals presence
-------------------------------------------------------------------------

* some currently optionals need to be moved to standard externals
* many tests need optional inclusion


::

    -- cfg4._line #define G4VERSION_NUMBER  1021 ===> 1021 
    -- Configuring okg4
    CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
    Please set them or make sure they are set and tested correctly in the CMake files:
    ImplicitMesher_LIBRARIES
        linked by target "NPY" in directory /Users/blyth/opticks/opticksnpy
        linked by target "NPolygonizerTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NCSGBSPTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NuvTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NOpenMeshCombineTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NOpenMeshCfgTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NOpenMeshFindTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NOpenMeshTest" in directory /Users/blyth/opticks/opticksnpy/tests

    -- Configuring incomplete, errors occurred!
    See also "/tmp/blyth/opticks20170612/build/CMakeFiles/CMakeOutput.log".
    make: *** No rule to make target `install'.  Stop.
    === opticks-full : DONE Mon Jun 12 13:41:56 CST 2017
    simon:~ blyth$ 








