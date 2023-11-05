#!/usr/bin/env bash
usage(){ cat << EOU
STestRunner.sh
================

Use this from CMakeLists.txt with::

    set(BASH_RUN_TEST_SOURCES
        SEnvTest_FAIL.cc
        SEnvTest_PASS.cc
        SSimTest.cc
        SBndTest.cc
    )

    find_program(BASH_EXECUTABLE NAMES bash REQUIRED)
    message(STATUS "BASH_EXECUTABLE : ${BASH_EXECUTABLE}")

    foreach(SRC ${BASH_RUN_TEST_SOURCES})
        get_filename_component(TGT ${SRC} NAME_WE)
        add_executable(${TGT} ${SRC})
        target_link_libraries(${TGT} SysRap)
        install(TARGETS ${TGT} DESTINATION lib)
        add_test(
           NAME ${name}.${TGT} 
           COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/STestRunner.sh ${TGT}
        )   
    endforeach()

Following:

* https://enccs.github.io/cmake-workshop/
* https://enccs.github.io/cmake-workshop/hello-ctest/
* https://github.com/ENCCS/cmake-workshop/blob/main/content/code/day-1/06_bash-ctest/solution/CMakeLists.txt

Dev::

   om 
   om-cd  # to the bdir 

OR after okdist-install-tests::

   cd /usr/local/opticks/tests

Then try ctest from the installed tree::

   ctest -N             # list 
   ctest -R SEnvTest_PASS  --output-on-failure
   ctest -R SEnvTest_FAIL  --output-on-failure

EOU
}

EXECUTABLE="$1"
shift
ARGS="$@"



geomscript=$HOME/.opticks/GEOM/GEOM.sh
[ -s $geomscript ] && source $geomscript


vars="HOME PWD GEOM BASH_SOURCE EXECUTABLE ARGS"
for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 

#env 
$EXECUTABLE $@
[ $? -ne 0 ] && echo $BASH_SOURCE : FAIL from $EXECUTABLE && exit 1 

exit 0

