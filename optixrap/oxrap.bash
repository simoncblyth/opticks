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

# === func-gen- : optixrap/optixrap fgp optixrap/optixrap.bash fgn optixrap fgh optixrap
oxrap-rel(){      echo optixrap  ; }
oxrap-src(){      echo optixrap/optixrap.bash ; }
oxrap-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oxrap-src)} ; }
oxrap-vi(){       vi $(oxrap-source) ; }
oxrap-usage(){ cat << EOU

OptiX Engine
==============

Hmm OptiX CMakeLists are kinda compilicated making it difficult 
to do an equvalent to oglrap- but maybe some of the
component needed can be stuffed into a library without the 
full CMakeLists machinery for compiling .cu to .ptx etc..

Porting on GPU photon generation to OptiX
--------------------------------------------

Python prototype: 

* /usr/local/env/chroma_env/src/chroma/chroma/gpu/photon_hit.py


OptiX 510 CUDA 10.1 internal header warning for host_defines.h
-----------------------------------------------------------------


::

    [ 83%] Building CXX object CMakeFiles/OpticksGL.dir/OKGLTracer.cc.o
    In file included from /usr/local/OptiX_510/include/optixu/../internal/optix_datatypes.h:33:0,
                     from /usr/local/OptiX_510/include/optixu/optixu_math_namespace.h:57,
                     from /usr/local/OptiX_510/include/optix_world.h:71,
                     from /home/blyth/local/opticks/include/OptiXRap/OXPPNS.hh:13,
                     from /home/blyth/local/opticks/include/OptiXRap/OContext.hh:19,
                     from /home/blyth/opticks/opticksgl/OAxisTest.cc:10:
    /usr/local/cuda-10.1/include/host_defines.h:54:2: warning: #warning "host_defines.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead." [-Wcpp]
     #warning "host_defines.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
      ^
    In



* https://gcc.gnu.org/onlinedocs/gcc-4.8.4/gcc/Warning-Options.html

-Wno-cpp
    (C, Objective-C, C++, Objective-C++ and Fortran only)

    Suppress warning messages emitted by #warning directives. 



* https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html

    #pragma GCC diagnostic kind option

    Modifies the disposition of a diagnostic. Note that not all diagnostics are
    modifiable; at the moment only warnings (normally controlled by ‘-W…’) can be
    controlled, and not all of them. Use -fdiagnostics-show-option to determine
    which diagnostics are controllable and which option controls them. 




OptiX 501 CUDA 9.1 macOS 10.13.4 Xcode 9.2
----------------------------------------------

::

    [ 75%] Linking CXX shared library libOptiXRap.dylib
    ld: warning: object file (CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBuf_.cu.o) was built for newer OSX version (10.13) than being linked (10.8)
    ld: warning: object file (CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBufBase_.cu.o) was built for newer OSX version (10.13) than being linked (10.8)
    ld: warning: object file (CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBufPair_.cu.o) was built for newer OSX version (10.13) than being linked (10.8)
    [ 75%] Built target OptiXRap



OptiX 400
-----------

::

    [ 86%] Building CXX object optixrap/CMakeFiles/OptiXRap.dir/OConfig.cc.o
    /Users/blyth/opticks/optixrap/OConfig.cc:204:11: warning: 4 enumeration values not handled in switch: 'RT_FORMAT_HALF', 'RT_FORMAT_HALF2', 'RT_FORMAT_HALF3'... [-Wswitch]
       switch(format)
              ^


Warnings
----------

::

    /Users/blyth/opticks/optixrap/cu/generate.cu(254): warning: variable "gencode" was declared but never referenced

    /Users/blyth/opticks/optixrap/cu/generate.cu(258): warning: variable "s" was set but never used

    /Users/blyth/opticks/optixrap/cu/LTminimalTest.cu(29): warning: variable "photon_offset" was declared but never referenced

    /Users/blyth/opticks/optixrap/cu/seedTest.cu(35): warning: variable "ghead" was set but never used

    /Users/blyth/opticks/optixrap/cu/OEventTest.cu(34): warning: variable "ghead" was set but never used




OptiX Version Isolation ?
---------------------------

TODO: rearrange OptiX use to facilitate easier version switching

* avoid having to rebuild all of Opticks just to use a different OptiX version ?
* try to firewall the change to just oxrap- ?
* how far can forward decalarations of OptiX types in oxrap- get me so other
  packages do not need to include any OptiX headers ?


Only opticksgl- and oxrap- use optix types?::

    simon:opticks blyth$ opticks-find optix:: -l | sort 
    ...
    ./opticksgl/OAxisTest.cc
    ...
    ./opticksgl/tests/OOAxisAppCheck.cc
    ./optixrap/OAccel.cc
    ./optixrap/OAccel.hh
    ...
    ./optixrap/tests/bufferTest.cc


But lots of inclusion of OptiX headers in oxrap- headers::

    simon:opticks blyth$ opticks-find OXPPNS | grep \.hh
    ./optixrap/OScene.cc:#include "OXPPNS.hh"
    ./optixrap/tests/OOtex0Test.cc:#include "OXPPNS.hh"
    ./optixrap/tests/OOtexTest.cc:#include "OXPPNS.hh"
    ./optixrap/tests/OPropertyLibTest.cc:#include "OXPPNS.hh"
    ./optixrap/OAccel.hh:#include "OXPPNS.hh"
    ./optixrap/OBndLib.hh:#include "OXPPNS.hh"
    ./optixrap/OColors.hh:#include "OXPPNS.hh"
    ./optixrap/OConfig.hh:#include "OXPPNS.hh"
    ./optixrap/OContext.hh:#include "OXPPNS.hh"
    ./optixrap/OEvent.hh:#include "OXPPNS.hh"
    ./optixrap/OGeo.hh:#include "OXPPNS.hh"
    ./optixrap/OLaunchTest.hh:#include "OXPPNS.hh"
    ./optixrap/OPropagator.hh:#include "OXPPNS.hh"
    ./optixrap/OPropertyLib.hh:#include "OXPPNS.hh"
    ./optixrap/OptiXUtil.hh:#include "OXPPNS.hh"
    ./optixrap/OScintillatorLib.hh:#include "OXPPNS.hh"
    ./optixrap/OSourceLib.hh:#include "OXPPNS.hh"
    ./optixrap/OTracer.hh:#include "OXPPNS.hh"


pimpl : hopefully forward decls will avoid going full pimpl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.drdobbs.com/cpp/making-pimpl-easy/205918714
* http://stackoverflow.com/questions/3597693/how-does-the-pimpl-idiom-reduce-dependencies
* http://stackoverflow.com/questions/13103311/hiding-library-dependencies-from-library-users



JPMT memory issue
-------------------

::

    op --jpmt --dbg 
    ...
    2016-07-12 14:00:17.002 INFO  [162808] [OEngineImp::preparePropagator@170] OEngineImp::preparePropagator DONE 
    2016-07-12 14:00:17.002 INFO  [162808] [OpEngine::preparePropagator@102] OpEngine::preparePropagator DONE 
    2016-07-12 14:00:17.003 INFO  [162808] [OpSeeder::seedPhotonsFromGensteps@65] OpSeeder::seedPhotonsFromGensteps
    2016-07-12 14:00:17.003 INFO  [162808] [OpSeeder::seedPhotonsFromGenstepsViaOpenGL@79] OpSeeder::seedPhotonsFromGenstepsViaOpenGL
    seedDestination src : dev_ptr 0xa00bbbe00 size 24 num_bytes 96 stride 24 begin 3 end 24 
    seedDestination dst : dev_ptr 0xa00da0000 size 1600000 num_bytes 6400000 stride 16 begin 0 end 1600000 
    iexpand  counts_size 1 output_size 100000
    2016-07-12 14:00:17.099 INFO  [162808] [OpZeroer::zeroRecords@61] OpZeroer::zeroRecords
    2016-07-12 14:00:17.118 INFO  [162808] [OEngineImp::propagate@176] OEngineImp::propagate
    2016-07-12 14:00:17.118 INFO  [162808] [OPropagator::prelaunch@290] OPropagator::prelaunch count 0 size(100000,1)
    2016-07-12 14:00:17.118 INFO  [162808] [OConfig::getNumEntryPoint@102] OConfig::getNumEntryPoint m_raygen_index 2 m_exception_index 2
    2016-07-12 14:00:17.118 INFO  [162808] [OContext::close@185] OContext::close numEntryPoint 2
    2016-07-12 14:00:17.118 INFO  [162808] [OConfig::dump@141] OContext::close m_raygen_index 2 m_exception_index 2
    OProg R 0 pinhole_camera.cu.ptx pinhole_camera 
    OProg E 0 pinhole_camera.cu.ptx exception 
    OProg M 1 constantbg.cu.ptx miss 
    OProg R 1 generate.cu.ptx generate 
    OProg E 1 generate.cu.ptx exception 
    2016-07-12 14:00:17.317 INFO  [162808] [OContext::launch@211] OContext::launch entry 1 width 100000 height 1
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Insufficient device memory. GPU does not support paging., [16515528])
    Process 27816 stopped
    * thread #1: tid = 0x27bf8, 0x00007fff94cdb866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff94cdb866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff94cdb866:  jae    0x7fff94cdb870            ; __pthread_kill + 20
       0x7fff94cdb868:  movq   %rax, %rdi
       0x7fff94cdb86b:  jmp    0x7fff94cd8175            ; cerror_nocancel
       0x7fff94cdb870:  retq   
    (lldb) bt
    * thread #1: tid = 0x27bf8, 0x00007fff94cdb866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff94cdb866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8c37835c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff930c8b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff92988f31 libc++abi.dylib`abort_message + 257
        frame #4: 0x00007fff929ae93a libc++abi.dylib`default_terminate_handler() + 240
        frame #5: 0x00007fff92ce6322 libobjc.A.dylib`_objc_terminate() + 124
        frame #6: 0x00007fff929ac1d1 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff929abc5b libc++abi.dylib`__cxa_throw + 124
        frame #8: 0x00000001030547a9 libOptiXRap.dylib`optix::ContextObj::checkError(this=0x00000004e1873430, code=RT_ERROR_UNKNOWN) const + 121 at optixpp_namespace.h:1840
        frame #9: 0x00000001030666d7 libOptiXRap.dylib`optix::ContextObj::compile(this=0x00000004e1873430) + 55 at optixpp_namespace.h:2376
        frame #10: 0x0000000103065ed4 libOptiXRap.dylib`OContext::launch(this=0x00000004e1873450, lmode=14, entry=1, width=100000, height=1, times=0x0000000c96332e70) + 660 at OContext.cc:237
        frame #11: 0x0000000103076fdc libOptiXRap.dylib`OPropagator::prelaunch(this=0x0000000c990b9b20) + 1388 at OPropagator.cc:301
        frame #12: 0x0000000103074467 libOptiXRap.dylib`OEngineImp::propagate(this=0x00000004dbd53c40) + 247 at OEngineImp.cc:178
        frame #13: 0x0000000103553ef9 libOpticksOp.dylib`OpEngine::propagate(this=0x00000004db9efc30) + 25 at OpEngine.cc:151
        frame #14: 0x00000001036449d5 libGGeoView.dylib`App::propagate(this=0x00007fff5fbfe958) + 309 at App.cc:1014
        frame #15: 0x000000010000af3e GGeoViewTest`main(argc=3, argv=0x00007fff5fbfeae8) + 1710 at GGeoViewTest.cc:114
        frame #16: 0x00007fff9014e5fd libdyld.dylib`start + 1
        frame #17: 0x00007fff9014e5fd libdyld.dylib`start + 1
    (lldb) 




Link error
------------

::

    [ 85%] Linking CXX shared library libOptiXRap.so
    /usr/bin/ld: CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBuf_.cu.o: relocation R_X86_64_32S against `.bss' can not be used when making a shared object; recompile with -fPIC
    CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBuf_.cu.o: could not read symbols: Bad value
    collect2: error: ld returned 1 exit status
    gmake[2]: *** [optixrap/libOptiXRap.so] Error 1
    gmake[1]: *** [optixrap/CMakeFiles/OptiXRap.dir/all] Error 2
    gmake: *** [all] Error 2
    [simonblyth@optix opticks]$ 



Remaining Warnings from nvcc
-----------------------------

TODO: dump the flags that nvcc is seeing.

* http://stackoverflow.com/questions/26867352/cant-get-rid-of-warning-command-line-option-std-c11-using-nvcc-cuda-cma
* https://cmake.org/Bug/view.php?id=15240

::

	[ 14%] Building NVCC (Device) object optixrap/CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBufPair_.cu.o
	cc1: warning: command line option ‘-Wno-non-virtual-dtor’ is valid for C++/ObjC++ but not for C [enabled by default]
	cc1: warning: command line option ‘-Woverloaded-virtual’ is valid for C++/ObjC++ but not for C [enabled by default]
	cc1: warning: command line option ‘-Wno-non-virtual-dtor’ is valid for C++/ObjC++ but not for C [enabled by default]
	cc1: warning: command line option ‘-Woverloaded-virtual’ is valid for C++/ObjC++ but not for C [enabled by default]




first run crashes in OTracerTest and GGeoViewTest
---------------------------------------------------

Seems to happen on first run, from a cold GPU.  Subsequent runs dont crash.
Suspect may be related to coding style that is cavalier about 
returning objects by value. 

I believed that the objects were some fancy pointer thingies that made this 
safe, but perhaps that is not entirely true.

Adjusting style to avoid passing objects around in texture creation 
seems to have prevented crashes in the texture creation. Instead now
get crashes in geometry conversion.

TODO: adopt minimal object copy around style:

* create objects in same scope where they will be placed into context
* modularize by "setup/configure" methods rather than "make/create" 
  passed references to the objects
  



::

    14  liboptix.1.dylib                0x000000010364f884 0x103593000 + 772228
    15  liboptix.1.dylib                0x00000001035a9001 rtProgramCreateFromPTXFile + 545
    16  libOptiXRap.dylib               0x0000000104916f5c optix::ContextObj::createProgramFromPTXFile(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&) + 620 (optixpp_namespace.h:2166)
    17  libOptiXRap.dylib               0x000000010491543f OConfig::createProgram(char const*, char const*) + 1999 (OConfig.cc:40)
    18  libOptiXRap.dylib               0x000000010491c3b0 OContext::createProgram(char const*, char const*) + 48 (OContext.cc:144)
    19  libOptiXRap.dylib               0x000000010492e1ca OGeo::makeTriangulatedGeometry(GMergedMesh*) + 138 (OGeo.cc:538)
    20  libOptiXRap.dylib               0x000000010492c51f OGeo::makeGeometry(GMergedMesh*) + 127 (OGeo.cc:429)
    21  libOptiXRap.dylib               0x000000010492bbd3 OGeo::convert() + 771 (OGeo.cc:184)
    22  libOpticksOp.dylib              0x000000010480ebaf OpEngine::prepareOptiX() + 4431 (OpEngine.cc:132)
    23  libGGeoView.dylib               0x0000000104519ad6 App::prepareOptiX() + 326 (App.cc:964)
    24  GGeoViewTest                    0x0000000101d26ea7 main + 1559 (GGeoViewTest.cc:90)
    25  libdyld.dylib                   0x00007fff89e755fd start + 1


::

    15  liboptix.1.dylib                0x00000001057f1540 0x105728000 + 824640
    16  liboptix.1.dylib                0x000000010572c3da rtAccelerationSetTraverser + 122
    17  libOptiXRap.dylib               0x0000000106acbac5 optix::ContextObj::createAcceleration(char const*, char const*) + 149 (optixpp_namespace.h:1893)
    18  libOptiXRap.dylib               0x0000000106ac8cb7 OGeo::makeAcceleration(char const*, char const*) + 583 (OGeo.cc:393)
    19  libOptiXRap.dylib               0x0000000106ac84c4 OGeo::makeRepeatedGroup(GMergedMesh*) + 1204 (OGeo.cc:258)
    20  libOptiXRap.dylib               0x0000000106ac6eb0 OGeo::convert() + 1504 (OGeo.cc:193)
    21  libOpticksOp.dylib              0x00000001069afbaf OpEngine::prepareOptiX() + 4431 (OpEngine.cc:132)
    22  libGGeoView.dylib               0x00000001066aead6 App::prepareOptiX() + 326 (App.cc:964)
    23  OTracerTest                     0x0000000103ec1802 main + 994 (OTracerTest.cc:51)
    24  libdyld.dylib                   0x00007fff89e755fd start + 1





macOS Warning
---------------

::

    [ 85%] Building NVCC (Device) object optixrap/CMakeFiles/OptiXRap.dir/OptiXRap_generated_OBufPair_.cu.o
    /Developer/OptiX/include/optixu/optixpp_namespace.h(593): 
    warning: overloaded virtual function "optix::APIObj::checkError" 
    is only partially overridden in class "optix::ContextObj"


* http://stackoverflow.com/questions/21462908/warning-overloaded-virtual-function-baseprocess-is-only-partially-overridde



ISSUE : rendering much slower than samples
-------------------------------------------------

Observations: 

* optix samples are much more responsive than ggv

* decreasing resolution of OptiX renders by large factors 
  has little impact on fps speed, but the time is all going 
  in the tracing ..

::

    [2015-Oct-19 16:26:34.259418]:info: OEngine::render
     trace_count             10
     trace_prep      0.000795775
     trace_time         14.4034
     avg trace_prep  7.95775e-05
     avg trace_time     1.44034
     render_count            10
     render_prep     0.000499662
     render_time     0.00064173
     avg render_prep 4.99662e-05
     avg render_time 6.4173e-05



Actions:

* note that tracing is being done even when no change in viewpoint


How can this be ? 

* maybe buffer access is rate determining 


Timing validate/compile/prelaunch/launch with OContext 
suggests that accel structures are being rebuilt for every launch::

    [2015-Oct-20 12:45:00.070613]:info: OContext::launch 
     count    31 
     validate      0.0347     0.0011 
     compile       0.4049     0.0131 
     prelaunch    25.0761     0.8089 
     launch       44.4527     1.4340 


Immediately repeating pre-launches and launches are 
taking time every time, without any context changes.  

From manual:

    rtContextCompile does allow the user to control the timing of the compilation,
    but the context should normally be finalized before compilation because any
    subsequent changes will cause a recompile within rtContextLaunch.


TODO:

* try simpler geometries, like a single analytic PMT





ENHANCEMENT : Analytic PMT Intersection 
---------------------------------------

* http://uk.mathworks.com/matlabcentral/answers/73606-intersection-of-3d-ray-and-surface-of-revolution

Intersection with a surface of revolution turns into finding the roots of a possibly high-ish 
order eqn, depending on the eqn of the profile.

Prereqisite for analytic geometry exploration is including GDML (CSG geometry)  
in G4DAE exports.


ISSUE: restricting bouncemax prevents recsel selection operation
----------------------------------------------------------------------
   
The index is constructed, but selection do


OptiX Model
------------

Seven different user supplied program types are compiled together
using GPU and ray tracing domain expertise to create a single
optimized CUDA kernel.  

Initially anyhow only need to implement two:

#. *Ray Generation* programs provides the entry point into the ray tracing pipeline,
   they start the trace and store results into output buffers.

#. *Closest hit* programs are invoked once traversal has found the closest
   intersection of a ray with the scene geometry. They can cast new rays
   and store results into the ray payload.


The other five mostly implement themselves when using triangle mesh::

#. *Intersection* programs implement ray-geometry intersection tests which are
   invoked to perform a geometric queries as acceleration structures are traversed.
   Simple ray triangle intersection could be provided but also
   analytic geometry intersection is possible.

#. *Bounding box* programs compute the bounds associated with each primitive to
   enable acceleration structures over arbitrary geometry

#. *Any Hit* programs are called during traversal for every ray-object
   intersection, the default of no operation is often appropriate.

#. *Miss* programs are executed when the ray does not intersect any geometry

#. *Exception* programs are called when problems such as stack overflow occur


Chroma while stepping loop
-----------------------------

Chroma steers propagation with while stepping loop /usr/local/env/chroma_env/src/chroma/chroma/cuda/propagate_vbo.cu
In pseudo-code this is structured::

    generate photons from gen steps, setup RNG

    while (steps < max_steps) {

       steps++;

       check for geometry intersection 
       if (no_intersection) -> out to bookeeping and exit 

       ------------inside closest hit ? ------------------------
       lookup wavelength dependant properties
       based on material at current photon location

           absorption_length
           scattering_length
           reemission_probability


       propagate_to_boundary 

            Random draws dictate what happens on the way 

            * time and position are advanced based on refractive index

            ABSORB   end trace

            REEMIT   direction, wavelength, polarization changed 
            SCATTER  direction, polarization changed 
                     -> continue to next step  

            PASS      to boundary 


       propagate_at_boundary/propagate_at_surface 

       -------------------------------------------------

   RNG state recording 
   record photon 



Porting GPU photon propagation to OptiX
-----------------------------------------

* Chroma while stepping loop needs to be chopped up to work with OptiX

* pre-loop and post-loop obviously in "Ray Generation"

* Where to draw the line between "Ray Generation" and "Closest Hit" ? 

  * one option would be to minimalize "Closest Hit" and just 
    use it to pass information back to "Ray Generation" via PerRayData

* What needs to live in per-ray data struct 
  




EOU
}


oxrap-env(){  
   olocal- 
   #optix-
   #optix-export 
   opticks-
}


oxrap-sdir(){ echo $(opticks-home)/optixrap ; }
oxrap-tdir(){ echo $(opticks-home)/optixrap/tests ; }
oxrap-idir(){ echo $(opticks-idir); }
oxrap-bdir(){ echo $(opticks-bdir)/$(oxrap-rel) ; }

oxrap-c(){   cd $(oxrap-sdir)/$1 ; }
oxrap-cd(){   cd $(oxrap-sdir)/$1 ; }
oxrap-scd(){  cd $(oxrap-sdir); }
oxrap-tcd(){  cd $(oxrap-tdir); }
oxrap-icd(){  cd $(oxrap-idir); }
oxrap-bcd(){  cd $(oxrap-bdir); }

oxrap-name(){ echo OptiXRap ; }
oxrap-tag(){  echo OXRAP ; }

oxrap-tests(){ tests.py $(oxrap-tdir)/CMakeLists.txt ; }

oxrap-tests-run(){
    local t
    local rc
    local log 
    oxrap-tests | while read t ; do
        which $t
        log=otr_$t.log 
        which $t > $log
        $t >> $log 2>&1
        rc=$?
        echo $rc
    done
}



oxrap-apihh(){  echo $(oxrap-sdir)/$(oxrap-tag)_API_EXPORT.hh ; }
oxrap---(){     touch $(oxrap-apihh) ; oxrap--  ; } 


oxrap-wipe(){ local bdir=$(oxrap-bdir) ; rm -rf $bdir ; } 

oxrap--(){                   opticks-- $(oxrap-bdir) ; oxrap-f64 ; } 
oxrap-t(){                   opticks-t $(oxrap-bdir) $* ; } 
oxrap-genproj() { oxrap-scd ; opticks-genproj $(oxrap-name) $(oxrap-tag) ; } 
oxrap-gentest() { oxrap-tcd ; opticks-gentest ${1:-OExample} $(oxrap-tag) ; } 
oxrap-txt(){ vi $(oxrap-sdir)/CMakeLists.txt $(oxrap-tdir)/CMakeLists.txt ; } 





oxrap-lsptx(){
   ls -l $(oxrap-bdir)/*.ptx
   ls -l $(opticks-prefix)/ptx/*.ptx
}

oxrap-rmptx(){
   rm $(oxrap-bdir)/*.ptx
   rm $(opticks-prefix)/ptx/*.ptx
}


oxrap-f64-notes(){ cat << EON
$FUNCNAME
====================

see ptx- ptx-vi


EON
}

oxrap-f64-(){ ptx.py $(opticks-prefix)/installcache/PTX $*  | c++filt ; }
oxrap-f64(){  $FUNCNAME- --exclude exception $*  ; }

################# OLD FUNCS ####################

oxrap-cmake-deprecated(){
   local iwd=$PWD

   local bdir=$(oxrap-bdir)
   mkdir -p $bdir
  
   oxrap-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(oxrap-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(oxrap-sdir)

   cd $iwd
}


oxrap-bin(){ echo $(oxrap-idir)/bin/${1:-OptiXRapTest} ; }
oxrap-export()
{
   echo -n
   #export SHADER_DIR=$(oxrap-sdir)/glsl
}
oxrap-run(){
   local bin=$(oxrap-bin)
   oxrap-export
   $bin $*
}











oxrap-ptxs(){
   local name=${1:-boundaryTest}
   local ptx=OptiXRap_generated_${name}.cu.ptx
   find $(oxrap-bdir) -name $ptx
   find $(opticks-prefix)/installcache/PTX -name $ptx
}


oxrap-cu()
{
   ## workaround for suspected lack of dependency setup for OptiX ptx 
   ## by touching the cu and cc of the test to force rebuilding 

   local name=${1:-boundaryTest} 
   local cu=cu/$name.cu
   local cc=tests/OO$name.cc

   oxrap-cd

   [ ! -f $cc ] && echo no such cc $cc && return 
   [ ! -f $cu ] && echo no such cu $cu && return 
  
   #rm $ptxs

   touch $cu $cc
   oxrap--

   local ptxs=$(oxrap-ptxs $name)
   ls -l $ptxs

   local tst=OO$name
   echo running tst $tst

   $tst
}



