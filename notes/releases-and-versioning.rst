releases-and-versioning
===========================

Opticks repositories on Bitbucket and Github
-----------------------------------------------

Day-to-day Opticks development uses the bitbucket repository

* https://bitbucket.org/simoncblyth/opticks/commits/

Less frequenty Opticks is pushed to the github repository. 

* https://github.com/simoncblyth/opticks

Git "snapshot" tags are occasionally made and pushed to 
both bitbucket and github.  Github provides a convenient way 
to download tar.gz or zip archives of those tagged versions from the below page. 

* https://github.com/simoncblyth/opticks/tags 


Snapshot History : Resuming tagging after 2 years with v0.2.0
-----------------------------------------------------------------

+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+  
| date       | tag     | OPTICKS_VERSION_NUMBER  | GEOCACHE_CODE_VERSION      | Notes                                                                           |
+============+=========+=========================+============================+=================================================================================+  
| 2023/10/12 | v0.2.0  | 20                      | NO CURRENT EQUIVALENT      | Resume tagging after 2 years of changes : huge change from prior release        |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+  


For a record of ancient tags see the "Snapshot pre-History" section at the end of this page. 


Workflow for adding "snapshot" tag to github and bitbucket
------------------------------------------------------------

Follow the workflow documented within the "~/opticks/addtag.sh" script 



OpticksVersionNumber.hh from OKConf package
------------------------------------------------

::

    epsilon:opticks blyth$ tail -15 okconf/OpticksVersionNumber.hh
    #pragma once

    /**
    OpticksVersionNumber
    =====================

    Definition of version integer

    **/


    #define OPTICKS_VERSION_NUMBER 10



Using **OPTICKS_VERSION_NUMBER**  to navigate API changes
----------------------------------------------------------

::

    epsilon:opticks blyth$ cat sysrap/tests/SOpticksVersionNumberTest.cc

    #include <cstdio>
    #include "OpticksVersionNumber.hh"

    int main()
    {
    #if OPTICKS_VERSION_NUMBER < 10
        printf("OPTICKS_VERSION_NUMBER < 10 \n"); 
    #elif OPTICKS_VERSION_NUMBER == 10
        printf("OPTICKS_VERSION_NUMBER == 10 \n"); 
    #elif OPTICKS_VERSION_NUMBER > 10
        printf("OPTICKS_VERSION_NUMBER > 10 \n"); 
    #else
        printf("OPTICKS_VERSION_NUMBER unexpected \n"); 
    #endif
        return 0 ; 
    }


OKConf/tests related to versioning
---------------------------------------

OpticksVersionNumberTest converts the macro into a string::

    epsilon:okconf blyth$ cat tests/OpticksVersionNumberTest.cc 
    #include <cstdio>
    #include "OpticksVersionNumber.hh"

    #define xstr(s) str(s)
    #define str(s) #s

    int main()
    {
        printf("%s\n",xstr(OPTICKS_VERSION_NUMBER)); 
        return 0 ; 
    }


The exeutable enables bash scripts to access the version::

    epsilon:opticks blyth$ ver=$(OpticksVersionNumberTest)
    epsilon:opticks blyth$ echo $ver
    10


OKConfTest dumps version integers using static functions such as  OKConf::OpticksVersionInteger()::

    epsilon:opticks blyth$ OKConfTest 
    OKConf::Dump
                      OKConf::OpticksVersionInteger() 10
                       OKConf::OpticksInstallPrefix() /usr/local/opticks
                            OKConf::CMAKE_CXX_FLAGS()  -fvisibility=hidden -fvisibility-inlines-hidden -fdiagnostics-show-option -Wall -Wno-unused-function -Wno-unused-private-field -Wno-shadow
                         OKConf::CUDAVersionInteger() 9010
                   OKConf::ComputeCapabilityInteger() 30
                            OKConf::OptiXInstallDir() /usr/local/optix
                         OKCONF_OPTIX_VERSION_INTEGER 50001
                        OKConf::OptiXVersionInteger() 50001
                         OKCONF_OPTIX_VERSION_MAJOR   5
                          OKConf::OptiXVersionMajor() 5
                         OKCONF_OPTIX_VERSION_MINOR   0
                          OKConf::OptiXVersionMinor() 0
                         OKCONF_OPTIX_VERSION_MICRO   1
                          OKConf::OptiXVersionMicro() 1
                       OKConf::Geant4VersionInteger() 1042
                       OKConf::ShaderDir()            /usr/local/opticks/gl

     OKConf::Check() 0



Git tags
-----------

List tags with "git tag" or "git tag -l"::

    epsilon:opticks blyth$ git tag -l
    v0.0.0-rc1
    v0.0.0-rc2
    v0.0.0-rc3
    v0.1.0-rc1
    v0.1.0-rc2




Snapshot pre-History 
----------------------

* *NB : IT WOULD BE VERY UNWISE TO ATTEMPT TO USE ANY OF THESE ANCIENT SNAPSHOTS*

+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+  
| date       | tag     | OPTICKS_VERSION_NUMBER  | GEOCACHE_CODE_VERSION      | Notes                                                                           |
+============+=========+=========================+============================+=================================================================================+  
| 2021/08/28 | v0.1.1  | 11                      | 14                         | Fermilab Geant4 team request, severe Cerenkov Wavelength bug found, DO NOT USE  | 
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+  
| 2021/08/30 | v0.1.2  | 12                      | 14                         | Fixed Cerenkov wavelength bug                                                   |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+  
| 2021/09/02 | v0.1.3  | 13                      | 14                         | Fixed minor CManager bug                                                        |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+  
| 2021/09/24 | v0.1.4  | 14                      | 14                         | Changes for Geant4 1100 beta, 4 cfg4 test fails remain, needing G4 GDML read fix|
|            |         |                         |                            | see notes/issues/Geant4_1100_GDML_AddProperty_error.rst                         |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+  
| 2021/09/30 | v0.1.5  | 15                      | 14                         | All use of G4PhysicsVector::SetSpline removed due to Geant4 API change,         |
|            |         |                         |                            | see notes/issues/Geant4_Soon_SetSpline_change.rst                               |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+  
| 2021/10/06 | v0.1.6  | 16                      | 14                         | More updates for Geant4 API in flux and fixing test fails,                      |
|            |         |                         |                            | see notes/issues/Geant4_Soon_GetMinLowEdgeEnergy.rst                            |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+  





