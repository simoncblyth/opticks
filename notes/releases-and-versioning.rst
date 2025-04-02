releases-and-versioning
===========================

Opticks repositories on Bitbucket and Github
-----------------------------------------------

Day-to-day Opticks development uses the bitbucket repository

* https://bitbucket.org/simoncblyth/opticks/commits/

Less frequenty Opticks is pushed to the github repository.

* https://github.com/simoncblyth/opticks



How to download a snapshot .tar.gz or .zip
---------------------------------------------

Git "snapshot" tags are occasionally made and pushed to
both bitbucket and github.

Snapshot .tar.gz/.zip for each tag can be downloaded from GitHub,
see the below page to find the URLs.

* https://github.com/simoncblyth/opticks/tags

For example download the v0.2.2 zip with::

    curl -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.2.zip

Note that the tarball does not include git information, it simply provides
a snapshot of the state of the repository at a particular commit that has been
marked by having an associated tag.


How to checkout snapshot tag into a new branch
------------------------------------------------

An alternative way to use a tag starting from a clone clone

::

    git fetch origin
    git fetch --all --tags               # fetch from upstream
    git tag                              # list the tags
    git checkout tags/v0.2.2 -b v022     # create branch for a tag and checkout into it
    git branch                           # list branches
    git checkout master                  # return to master with the latest




git clone https://github.com/simoncblyth/opticks.git
cd opticks
git checkout v0.2.2





Snapshot Tags History
----------------------

+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| date       | tag     | OPTICKS_VERSION_NUMBER  | Notes                                                                           |
+============+=========+=========================+=================================================================================+
| 2025/04/02 | v0.3.4  | 34                      | wayland viz fix, handle no CUDA device detected with opticksMode 1              |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2025/03/17 | v0.3.3  | 33                      | try to hide non-zero rc in bashrc from the set -e used by gitlab-ci             |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2025/03/17 | v0.3.2  | 32                      | okdist-- installed tree fixes                                                   |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2025/01/11 | v0.3.1  | 31                      | fixes BR/BT reversion in v0.3.0                                                 |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2025/01/08 | v0.3.0  | 30                      | many changes, including jump to Philox RNG + addition of out-of-core running    |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2024/02/01 | v0.2.7  | 27                      | tag requested by Hans, just for some convenience OpticksPhoton methods          |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2024/01/25 | v0.2.6  | 26                      | fix VRAM leak by using default CUDA stream for every launch                     |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2023/12/19 | v0.2.5  | 25                      | fix off-by-one sensor identifier bug                                            |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2023/12/18 | v0.2.4  | 24                      | fix for tests installation                                                      |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2023/12/18 | v0.2.3  | 23                      | Addition of smonitor GPU memory monitoring, explicit reset API in QSim and G4CX |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2023/12/14 | v0.2.2  | 22                      | Addition of profiling machinery, introduce Release build, fix CK generation bug |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2023/10/20 | v0.2.1  | 21                      | Fix stale dependencies issue reported by Hans, remove opticksaux from externals |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+
| 2023/10/12 | v0.2.0  | 20                      | Resume tagging after 2 years of changes : huge change from prior release        |
+------------+---------+-------------------------+---------------------------------------------------------------------------------+

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





