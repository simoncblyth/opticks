Quickstart Examples showing how to integrate Opticks acceleration with your simulation
========================================================================================

.. contents:: Table of Contents
   :depth: 2


G4OKTest.cc : Basic Example
-----------------------------

* https://bitbucket.org/simoncblyth/opticks/src/master/g4ok/tests/G4OKTest.cc

This basic example uses “fake” gensteps. 
The G4OKTest executable is built by the standard opticks build::

    epsilon:~ blyth$ which G4OKTest   ## not yet in path
    epsilon:~ blyth$ oe               ## environment setup
    epsilon:~ blyth$ which G4OKTest 
    /usr/local/opticks/lib/G4OKTest
    epsilon:~ blyth$ t oe            ## use the typeset shortcut to introspect the bash function "oe"
    oe () 
    { 
        oe- 2> /dev/null
    }


The G4OKTest executable will use the geometry identified by the OPTICKS_KEY envvar 
or will create the geometry directly from GDML file identified by the `--gdmlpath` 
argument (NB there are two dashes before almost all commandline options used by Opticks
but the html conversion and also email clients are sometimes double-dash challenged).


CerenkovMinimal : Extended Example
------------------------------------

* https://bitbucket.org/simoncblyth/opticks/src/master/examples/Geant4/CerenkovMinimal/

A larger example showing collection of real Geant4 gensteps. 



Building Opticks against your simulation frameworks externals : boost, geant4, xercesc, clhep
-----------------------------------------------------------------------------------------------

When integrating software packages having multiple versions of external packages linked together 
must be avoided as that leads to incompatible interfaces that at best fail to compile and 
and worst succeed to compile but cause difficult to find bugs.

For this reason the Opticks build uses the so called foreign external package 
identified by the CMAKE_PREFIX_PATH envvar mechanism:: 

    epsilon:~ blyth$ opticks-foreign 
    boost
    clhep
    xercesc
    g4

This allows Opticks to be build against the externals of your detector simulation 
framework which can be installed into any directory. The envvars CMAKE_PREFIX_PATH, LD_LIBRARY_PATH, PATH, 
PKG_CONFIG_PATH can be conveniently prepended using the *opticks-prepend-prefix* function or
can be manually setup by your detector simulation framework machinery.::

    ## hookup paths to access "foreign" externals, not yet existing dirs just give warnings
    opticks-prepend-prefix /usr/local/mysimframework_externals/clhep
    opticks-prepend-prefix /usr/local/mysimframework_externals/xercesc
    opticks-prepend-prefix /usr/local/mysimframework_externals/g4_1042
    opticks-prepend-prefix /usr/local/mysimframework_externals/boost


