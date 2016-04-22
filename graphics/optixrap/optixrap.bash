# === func-gen- : graphics/optixrap/optixrap fgp graphics/optixrap/optixrap.bash fgn optixrap fgh graphics/optixrap
optixrap-rel(){      echo graphics/optixrap  ; }
optixrap-src(){      echo graphics/optixrap/optixrap.bash ; }
optixrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optixrap-src)} ; }
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
   elocal- 
   optix-
   optix-export 
   OPTICKS-
}


optixrap-sdir(){ echo $(env-home)/graphics/optixrap ; }

optixrap-idir(){ echo $(OPTICKS-idir); }
optixrap-bdir(){ echo $(OPTICKS-bdir)/$(optixrap-rel) ; }


optixrap-scd(){  cd $(optixrap-sdir); }
optixrap-cd(){   cd $(optixrap-sdir); }

optixrap-icd(){  cd $(optixrap-idir); }
optixrap-bcd(){  cd $(optixrap-bdir); }
optixrap-name(){ echo OptiXRap ; }

optixrap-wipe(){
   local bdir=$(optixrap-bdir)
   rm -rf $bdir
}

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


optixrap-make(){
   local iwd=$PWD

   optixrap-bcd 
   make $*

   cd $iwd
}

optixrap-install(){
   optixrap-make install
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



optixrap--()
{
    optixrap-make clean
    optixrap-make
    optixrap-install
}







