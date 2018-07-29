x4gen-source(){ echo $BASH_SOURCE ; }
x4gen-vi(){ vi $(x4gen-source) ; }
x4gen-env(){  olocal- ; opticks- ; }
x4gen-usage(){ cat << EOU

X4Gen Usage 
===================


1. *OKX4Test --g4codegen* 

   Generate the geometry serialization mains 
   for each solid of the geometry beneath a directory 
   like the below for live running:

   /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/g4codegen
   (formerly $TMP/x4gen/tests)

2. x4gen-- 

   Generate the CMake project to build and install the executables and then do so.

3. x4gen-run

   Run all the executables, writing test geometries 



EOU
}


x4gen-dir(){ echo $(dirname $(x4gen-source)) ; }
x4gen-cd(){  cd $(x4gen-dir) ; }
x4gen-c(){   cd $(x4gen-dir) ; }


x4gen-testconfig () 
{ 
    local testconfig;
    local testname;
    if [ -n "$TESTCONFIG" ]; then
        testconfig=${TESTCONFIG};
    fi;
    echo $testconfig

}


x4gen-csg---()
{
    local cmdline=$*;
    local testconfig=$(x4gen-testconfig)
    echo testconfig $testconfig

    op.sh $cmdline \
          --rendermode +global,+axis \
          --animtimemax 20 \
          --timemax 20 \
          --geocenter \
          --eye 1,0,0 \
          --dbganalytic \
          --test \
          --testconfig "$testconfig"  \
          --tracer \
          --printenabled 
           

}
x4gen-csg--(){ x$(x4gen-lvf) ; }
x4gen-csg-(){ $FUNCNAME- 2>&1 1>/dev/null ; }
x4gen-csg(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME-) x4gen-csg--- $* ;}  


x4gen-lv(){ echo ${LV:-000} ; }
x4gen-lvf(){ echo $(printf "%0.3d" $(x4gen-lv)) ; }
x4gen-solids(){ echo $(x4gen-base)/solids.txt  ; }

x4gen-info(){
    local lvf=$(x4gen-lvf)
    x$lvf
    grep "lv:$lvf" $(x4gen-solids)
}

x4gen-deep(){
   grep mx:1 $(x4gen-solids) 
   grep mx:09 $(x4gen-solids) 
   grep mx:08 $(x4gen-solids) 
   echo ----------------------------- height greater than 7 skipped in kernel ------- 
   grep mx:07 $(x4gen-solids) 
   grep mx:06 $(x4gen-solids) 
   grep mx:05 $(x4gen-solids) 
   grep mx:04 $(x4gen-solids) 
   grep mx:03 $(x4gen-solids) 
   #grep mx:02 $(x4gen-solids) 
   #grep mx:01 $(x4gen-solids) 
   #grep mx:00 $(x4gen-solids) 
}

## TESTCONFIG is a capture of stderr with stdout ignored 





x4gen-paths(){
    local arg
    for arg in $* 
    do
        echo $(x4gen-path $arg)
    done
}

x4gen-ed(){ 
   [[ $# -eq 0 ]] && echo expecting one or more lvidx integer arguments && return 
   local paths=$(x4-nnt-paths $*)
   vi $paths
}



x4gen-cd(){   cd $(x4gen-base) ; }
#x4gen-base(){ echo $TMP/x4gen ; }
x4gen-base(){ echo  /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/g4codegen ; }

x4gen-name(){ echo x$(printf "%0.3d" ${1:-0}) ; }
x4gen-path(){ echo $(x4gen-base)/tests/$(x4gen-name $1).cc ; }

x4gen-hh(){ cat << EOC

#pragma once
struct X4Gen 
{
    X4Gen(); 
};

EOC
}

x4gen-cc(){ cat << EOC

#include "X4Gen.hh"
#include "PLOG.hh"

X4Gen::X4Gen()
{
   LOG(info) << "." ; 
}

EOC
}


x4gen-CMakeLists(){ cat << EOH

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name X4Gen)
project(\${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(ExtG4 REQUIRED CONFIG)  

set( SOURCES X4Gen.cc )
set( HEADERS X4Gen.hh )
    
add_library( \${name}  SHARED \${SOURCES} \${HEADERS} )
target_link_libraries( \${name} PUBLIC
    Opticks::ExtG4
)

target_include_directories( \${name} PUBLIC
   $<BUILD_INTERFACE:\${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_definitions( \${name} PUBLIC OPTICKS_X4GEN )

install(FILES \${HEADERS}  DESTINATION \${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS \${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)

EOH
}

x4gen-CMakeLists-tests-head(){ cat << EOH

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name X4GenTest)
project(\${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(TEST_SOURCES

EOH
}

x4gen-CMakeLists-tests-tail(){ cat << EOT
)
foreach(TEST_CC_SRC \${TEST_SOURCES})
    get_filename_component(TGT \${TEST_CC_SRC} NAME_WE)
    add_executable(\${TGT} \${TEST_CC_SRC})

    set(testname \${name}.\${TGT})
    add_test(\${testname} \${TGT})

    target_link_libraries(\${TGT} X4Gen )
    install(TARGETS \${TGT} DESTINATION lib)
endforeach()

EOT
}

x4gen-idx(){ cat << EOI
1
EOI
}

x4gen-go(){ cat << EOG
#!/bin/bash -l

opticks-

sdir=\$(pwd)
name=\$(basename \$sdir) 
bdir=/tmp/\$USER/opticks/\$name/build 

rm -rf \$bdir && mkdir -p \$bdir && cd \$bdir && pwd 

cmake \$sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=\$(opticks-prefix)/externals \
    -DCMAKE_INSTALL_PREFIX=\$(opticks-prefix) \
    -DCMAKE_MODULE_PATH=\$(opticks-home)/cmake/Modules


make
make install   

EOG
}

x4gen--(){
   local msg="=== $FUNCNAME :"
   local base=$(x4gen-base)

   echo $msg generating project into $base 

   mkdir -p $base/tests
   local iwd=$PWD
   cd $base

   x4gen-go > go.sh
   chmod ugo+x go.sh
   x4gen-hh > X4Gen.hh
   x4gen-cc > X4Gen.cc
   x4gen-CMakeLists > CMakeLists.txt

   x4gen-CMakeLists-tests-head  > tests/CMakeLists.txt
   ( cd tests ; ls -1 x*.cc )  >> tests/CMakeLists.txt
   x4gen-CMakeLists-tests-tail >> tests/CMakeLists.txt

   echo $msg invoking go.sh to compile/build/install
   ./go.sh 

   cd $iwd
}


x4gen-run()
{
   local x
   ( cd $(opticks-bindir) ; ls -1 x* ) | while read x 
   do 
        echo x $x 
        $x
   done
}

