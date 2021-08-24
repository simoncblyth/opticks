releases-and-versioning
===========================


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


Opticks repositories on Bitbucket and Github
-----------------------------------------------

Day-to-day Opticks development uses the bitbucket repository

* https://bitbucket.org/simoncblyth/opticks/commits/
* https://bitbucket.org/simoncblyth/opticks/src/v0.1.0-rc2/

Infrequently the lastest Opticks is pushed to the github repository. 

* https://github.com/simoncblyth/opticks
* https://github.com/simoncblyth/opticks/tags 

Both Bitbucket and Github provide web interfaces listing tags.
However Github also provides a convenient way to download 
tar.gz or zip of the tagged versions of Opticks.


Workflow for making "snapshot" tags
--------------------------------------

Infrequently (seasonally) or when users request it. Consider making a "snapshot" tag:

1. commit and push any changes to bitbucket
2. check how many *opticks-t* test fails
3. decide if now is a good time to "snapshot" tag

When it is a good time to snapshot. Make the tag::

1. Following a consistent pattern, decide on the next tag string eg "v0.1.0-rc2" and corresponding incremented *OPTICKS_VERSION_NUMBER* eg 10
2. set the incremented *OPTICKS_VERSION_NUMBER* in okconf/OpticksVersionNumber.hh, commit and push to BOTH bitbucket and github::

      git push 
      git push github 

3. make the git tag, and push tags to both bitbucket and github::

      git tag -a v0.1.0-rc3 -m "Mid-summer snapshot corresponding to OPTICKS_VERSION_NUMBER 10"
      git push --tags
      git push github --tags
 

4. check the web interfaces

   * https://github.com/simoncblyth/opticks/tags
   * https://bitbucket.org/simoncblyth/opticks/src/v0.1.0-rc2/






