# === func-gen- : optixrap/optixrap fgp optixrap/optixrap.bash fgn optixrap fgh optixrap
optixrap-rel(){      echo optixrap  ; }
optixrap-src(){      echo optixrap/optixrap.bash ; }
optixrap-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(optixrap-src)} ; }
optixrap-vi(){       vi $(optixrap-source) ; }
optixrap-usage(){ cat << EOU

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


optixrap-env(){  
   olocal- 
   optix-
   optix-export 
   opticks-
}


optixrap-sdir(){ echo $(opticks-home)/optixrap ; }
optixrap-tdir(){ echo $(opticks-home)/optixrap/tests ; }
optixrap-idir(){ echo $(opticks-idir); }
optixrap-bdir(){ echo $(opticks-bdir)/$(optixrap-rel) ; }

optixrap-cd(){   cd $(optixrap-sdir); }
optixrap-scd(){  cd $(optixrap-sdir); }
optixrap-tcd(){  cd $(optixrap-tdir); }
optixrap-icd(){  cd $(optixrap-idir); }
optixrap-bcd(){  cd $(optixrap-bdir); }

optixrap-name(){ echo OptiXRap ; }
optixrap-tag(){  echo OXRAP ; }


optixrap-apihh(){  echo $(optixrap-sdir)/$(optixrap-tag)_API_EXPORT.hh ; }
optixrap---(){     touch $(optixrap-apihh) ; optixrap--  ; } 


optixrap-wipe(){ local bdir=$(optixrap-bdir) ; rm -rf $bdir ; } 

optixrap--(){                   opticks-- $(optixrap-bdir) ; } 
optixrap-ctest(){               opticks-ctest $(optixrap-bdir) $* ; } 
optixrap-genproj() { optixrap-scd ; opticks-genproj $(optixrap-name) $(optixrap-tag) ; } 
optixrap-gentest() { optixrap-tcd ; opticks-gentest ${1:-OExample} $(optixrap-tag) ; } 
optixrap-txt(){ vi $(optixrap-sdir)/CMakeLists.txt $(optixrap-tdir)/CMakeLists.txt ; } 



optixrap-lsptx(){
   ls -l $(optixrap-bdir)/*.ptx
   ls -l $(opticks-prefix)/ptx/*.ptx
}

optixrap-rmptx(){
   rm $(optixrap-bdir)/*.ptx
   rm $(opticks-prefix)/ptx/*.ptx
}



################# OLD FUNCS ####################

optixrap-cmake-deprecated(){
   local iwd=$PWD

   local bdir=$(optixrap-bdir)
   mkdir -p $bdir
  
   optixrap-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(optixrap-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(optixrap-sdir)

   cd $iwd
}


optixrap-bin(){ echo $(optixrap-idir)/bin/${1:-OptiXRapTest} ; }
optixrap-export()
{
   echo -n
   #export SHADER_DIR=$(optixrap-sdir)/glsl
}
optixrap-run(){
   local bin=$(optixrap-bin)
   optixrap-export
   $bin $*
}



