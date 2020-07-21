##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

x4gen-source(){ echo $BASH_SOURCE ; }
x4gen-vi(){ vi $(x4gen-source) ; }
x4gen-env(){  olocal- ; opticks- ; }
x4gen-usage(){ cat << EOU

X4Gen Usage 
==============

Geocache created with "--g4codegen" option contains g4codegen/tests/*.{cc,gdml} 
This x4gen- script adds the CMake and bash machinery to convert the 
sources into executables. Done this way as CMake and bash machinery ages 
much more rapidly than the code for creating solids.

::

   x4gen-
   x4gen--



ISSUES
--------

* x016.cc generated with zero content

::

    2019-06-13 22:00:53.292 INFO  [385857] [X4PhysicalVolume::convertSolid@500]  [ 16 lFasteners0x4c012d0
    G4GDML: Writing solids...
    G4GDML: Writing solids...
    2019-06-13 22:00:53.294 INFO  [385857] [NTreeBalance<T>::create_balanced@40] op_mask union intersection 
    2019-06-13 22:00:53.294 INFO  [385857] [NTreeBalance<T>::create_balanced@41] hop_mask union 
    2019-06-13 22:00:53.294 INFO  [385857] [NTreeBalance<T>::create_balanced@65]  bileafs 2 otherprim 9
    2019-06-13 22:00:53.294 ERROR [385857] [NTreeBuilder<T>::init@169]  num_subs 2 num_otherprim 9 num_prim 13 height 4 mode MIXED operator union
    2019-06-13 22:00:53.302 ERROR [385857] [X4CSG::generateTestMain@236]  skip as no g4code 




TODO:

* perhaps more useful to generate a library with an executable that
  takes an argument to the index, rather than executables for every index



1. *OKX4Test --g4codegen* 

    Runinng OKX4Test with --g4codegen option generates an 
    g4codegen dir within the geocache keydir, 
    containing G4VSolid making mains::

        blyth@localhost 1]$ find g4codegen
        g4codegen
        g4codegen/tests
        g4codegen/tests/x000.cc
        g4codegen/tests/x001.cc
        g4codegen/tests/x002.cc
        g4codegen/tests/x003.cc
        g4codegen/tests/x004.cc
        g4codegen/tests/x005.cc
        g4codegen/tests/x006.cc
        ...

The mains are all the same, with a different make_solid::

    int main( int argc , char** argv )
    {
        OPTICKS_LOG(argc, argv);

        const char* exename = PLOG::instance->args.exename() ; 

        G4VSolid* solid = make_solid() ; 

        std::string csgpath = BFile::FormPath(X4::X4GEN_DIR, exename) ; 

        X4CSG::Serialize( solid, csgpath.c_str() ) ;

        return 0 ; 
    }

Currently::

    const char* X4::X4GEN_DIR = "$TMP/x4gen" ;

    026 void X4CSG::Serialize( const G4VSolid* solid, const char* csgpath ) // static
     27 {
     28     X4CSG xcsg(solid);
     29     std::cerr << xcsg.save(csgpath) << std::endl ;   // NB only stderr emission to be captured by bash 
     30     xcsg.dumpTestMain();
     31 }

    131 std::string X4CSG::configuration(const char* csgpath) const
    132 {
    133     std::stringstream ss ;
    134     ss << "analytic=1_csgpath=" << csgpath ;
    135     return ss.str();
    136 }
    137 
    138 std::string X4CSG::save(const char* csgpath)
    139 {
    140     ls = NCSGList::Create( trees, csgpath , verbosity );
    141     ls->savesrc();
    142     return configuration(csgpath);
    143 }


2. x4gen-- 

   Generate the CMake project to build and install the executables and then do so.

3. x4gen-run

   Run all the executables, writing test geometries 



EOU
}


x4gen--notes(){ cat << EON

EON
}


x4gen-dir(){ echo $(dirname $(x4gen-source)) ; }


x4gen-testconfig () 
{ 
    local testconfig;
    local testname;
    if [ -n "$TESTCONFIG" ]; then
        testconfig=${TESTCONFIG};
    fi;

    echo $testconfig
    #echo ${testconfig/analytic=1/analytic=0}    not so easy, trips assert in GGeoTest::initCreateCSG
}


x4gen-csg---()
{
    local cmdline=$*;
    local testconfig=$(x4gen-testconfig)
    echo testconfig $testconfig

    op.sh $cmdline \
          --envkey \
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
x4gen-csg-(){ $FUNCNAME- 2>&1 1>/dev/null ; } ## stderr with stdout ignored 
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
x4gen-base(){ x4gen-base-fromkey ; } 
x4gen-base-fromkey(){ geocache- ; echo $(geocache-keydir) ; }  # requires OPTICKS_KEY envvar 


x4gen-name(){ echo x$(printf "%0.3d" ${1:-0}) ; }
x4gen-path(){ echo $(x4gen-base)/g4codegen/tests/$(x4gen-name $1).cc ; }

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

x4gen-go-(){ cat << EOG
#!/bin/bash -l

opticks-
om-

sdir=\$(pwd)
name=\$(basename \$sdir) 
bdir=/tmp/\$USER/opticks/\$name/build 

rm -rf \$bdir && mkdir -p \$bdir && cd \$bdir && pwd 

om-cmake \$sdir 

make
make install   

echo sdir \$sdir
echo bdir \$bdir

EOG
}




x4gen--(){
   local msg="=== $FUNCNAME :"
   local base=$(x4gen-base)/g4codegen

   echo $msg generating project into $base 

   mkdir -p $base/tests
   local iwd=$PWD
   cd $base

   x4gen-go- > go.sh
   chmod ugo+x go.sh
   x4gen-hh > X4Gen.hh
   x4gen-cc > X4Gen.cc
   x4gen-CMakeLists > CMakeLists.txt

   x4gen-CMakeLists-tests-head  > tests/CMakeLists.txt
   ( cd tests ; find * -name 'x???.cc' -size +0 ) >> tests/CMakeLists.txt
   ## skip the zero length x016.cc 

   x4gen-CMakeLists-tests-tail >> tests/CMakeLists.txt

   echo $msg invoking go.sh to compile/build/install
   ./go.sh 

   cd $iwd
}

x4gen-go()
{
   cd $(x4gen-base)/g4codegen  
   ./go.sh 

}




#x4gen-run-(){ ( cd $(opticks-bindir) ; ls -1 x* )  ; }
x4gen-run-(){ ( cd $(opticks-bindir) ; ls -1 x00* x01* x02* x03* )  ; }
x4gen-run()
{
   local x
   x4gen-run- | while read x 
   do 
        echo x $x 
        $x
   done
}

