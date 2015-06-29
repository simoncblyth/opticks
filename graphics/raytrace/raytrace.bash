# === func-gen- : graphics/raytrace/raytrace fgp graphics/raytrace/raytrace.bash fgn raytrace fgh graphics/raytrace
raytrace-src(){      echo graphics/raytrace/raytrace.bash ; }
raytrace-source(){   echo ${BASH_SOURCE:-$(env-home)/$(raytrace-src)} ; }
raytrace-vi(){       vi $(raytrace-source) ; }
raytrace-usage(){ cat << EOU

Test Combination of Assimp and OptiX
======================================

::

    raytrace-
    raytrace-v         # default phong shader, 3-colorful can see things inside AD once navigate there
    raytrace-v -l 0    # default phong shader, lights off : black silhouette against blue bkgd 


* press "ctrl" and drag up/down to zoom out/in 

* for fps display press "r" and then "d"


Blogs on Ray Tracing
----------------------

* http://raytracey.blogspot.tw


Visual Effects : bloom
------------------------

* http://prideout.net/archive/bloom/


CUDA kdtree
------------

* https://github.com/unvirtual/cukd



Specular Ray Trace
---------------------

* :google:`specular raytrace`

* how to create an RGB image from mono single wavelength images ?

* http://www.pjreddie.com/media/files/Redmon_Thesis.pdf


Rendering Dispersion with composite spectral model 

* https://www.cs.sfu.ca/~mark/ftp/Cgip00/dispersion_CGIP00.pdf




* optix helpers.h has XYZ2rgb  (from CIE XYZ to RGB)

::

    static __host__ __device__ __inline__ optix::float3 XYZ2rgb( const optix::float3& xyz)



Raytrace with OpenGL compute shaders
--------------------------------------

* https://github.com/LWJGL/lwjgl3-wiki/wiki/2.6.1.-Ray-tracing-with-OpenGL-Compute-Shaders-(Part-I)


Acceleration Cache Flakiness
------------------------------

Sometimes code changes somehow invalidate the cached acceleration structure 
resulting in no error but causing a blank render.  Solution is 
to recreate the acceleration cache by running without the "--cache" option
ie::

    raytrace-run --g4dae $DAE_NAME_DYB 

As opposed to normal faster running::

    raytrace-run --cache --g4dae $DAE_NAME_DYB 

The **raytrace-clean** function wipes and rebuilds from scratch and then runs 
without **--cache** in order to recreate everything (actually curand cache is not recreated).  


std::string in interface causing linker problems
-------------------------------------------------

Hmm, I've seen this before switching std::string arguments for const char* 
from interface to G4DAELoader avoids linking problems.  This suggests incompatible 
compiler or compilation option differences between raytrace- and optixrap-. 
Switching workaround fine in this case, but are more issues lurking ?

* Moral : avoid std::string in public interfaces ?

::

    Undefined symbols for architecture x86_64:
      "G4DAELoader::isMyFile(std::string const&)", referenced from:
          MeshViewer::initGeometry() in MeshViewer.cpp.o
      "G4DAELoader::G4DAELoader(std::string const&, optix::Handle<optix::ContextObj>, optix::Handle<optix::GeometryGroupObj>, optix::Handle<optix::MaterialObj>, char const*, char const*, char const*, bool)", referenced from:
          MeshViewer::initGeometry() in MeshViewer.cpp.o
    ld: symbol(s) not found for architecture x86_64
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    make[2]: *** [MeshViewer] Error 1


Strings args have different messier symbols invoking basic_string in the lib::

    nm /usr/local/env/graphics/optixrap/lib/libOptiXRap.dylib | grep G4DAE | c++filt


compilation units
------------------

OptiXGeometry
    abstract base holder of OptiX context, geometries, materials, geometry intances  

AssimpOptiXGeometry
    deprecated specialization of OptiXGeometry using initial direct from Assimp
    approach, lacks material/surface property handling : the complexity of this
    inspired creation of GGeo 

GGeoOptiXGeometry
    specialization of OptiXGeometry, depending only on RayTraceConfig, GGeo, OptiX  
    provides conversion of GGeo geometry into OptiX geometry including 
    material and surface optical properties packed into textures 

RayTraceConfig
    holder of OptiX context and usage utilities (eg compiling OptiX programs) 
    uses compiled in configuration header RayTraceConfigInc propagating values
    such as RAYTRACE_SRC_DIR, RAYTRACE_PTX_DIR etc from CMakeLists.txt 
    into header defines

G4DAELoader
    Mainlyneeded to integrate with MeshViewer(?) 
    Uses AssimpWrap and GGeoOptixGeometry to load and convert geometry
    into form needed by OptiX.

MeshViewer/MeshScene
    Main and MeshViewer class, based on messy GLUT base OptiX sample code 



ortho mode
-----------

* need to hit "e" to adjust scene_epsilon to see any geometry


curand RNG cache handling
---------------------------

Tried two ways of doing OptiX/CUDA interop, eventually 
avoided interop via a cache file.

* **--rng-cuda** CUDA Owned curand buffer

  * caching the curand buffer works, allowing fast starts
  * resizing does not work, it causes hangs forcing hardware restart
    DUE to this disabled this mode

* **--rng-optix** OptiX Owned curand buffer

  * loading the cached curand buffer can be made to work using 
    host side map/memcpy/unmap, but only once. 
    Resizing causes crashes. To workaround this adopt the maximum
    buffer size needed for full screen and set that up once only.

  * thusly CUDA/OptiX interop is avoided by writing the curandStates 
    in a pure CUDA process and reading the cache from the OptiX process



RTprogram Re-compilation Flakiness
-------------------------------------------------

* sometimes observe following a cu change and quick 
  recompile with raytrace-x 
  that the change does not take effect on first raytrace-x 
  but does on 2nd try ? 

  * cmake Makefile sees the cu change and rebuilds the ptx, but 
    it seems that is not remaking the GPU program until the second call

  * optix reads ptx, cmake ptx rebuild dependency issue ? 

  * so far raytrace-x-clean that does a full rebuild including the caches
    has not shown flakiness

  * suspect the postage stamp 128x128 image that initially appears 
    prior to rng cache loading is unhealthy 


Launch failed : No binary for GPU
-----------------------------------

::

    OptiX Error: Launch failed (Details: Function "RTresult
    _rtContextCompile(RTcontext)" caught exception: : error: Encountered a CUDA
    error: createFromMemoryPtx returned (209): No binary for GPU, [7340351]
    [7340352]) delta:env blyth$ 


Next Steps 
-----------

* taming the flakiness

  * flakiness manifests in that some minor code change results in a 
    menagerie of errors : this is probably partial rebuild dependency
    problem 

  * make each ptx as simple as possible, minimize the number of functions
    and top level declarations in each cu 

* port Chroma cuda/photon.h:fill_state to OptiX 

  * geometricNormal 




OptiX Questions
-----------------

* What are pros and cons of passing info between RTprogram via attribute or PerRayData 

  * geometricNormal more natural as attribute, as this will be changing as the ray 
    bounces around ?


Pre-requisites
--------------

* NVIDIA CUDA 5.5, cuda- 
* NVIDIA OptiX 3.5.1 or higher, optix-
* assimp-  C++ COLLADA importer
* assimpwrap- My wrapping of Assimp  


Installing Pre-requisites
----------------------------

Assimp and AssimpWrap do not use CUDA or OptiX so 
they can be build from another node onto filesystem 
shared with the GPU node::

    assimp-
    assimp-get
    assimp-cmake
    assimp-make
    assimp-install

    assimpwrap-
    assimpwrap-cmake
    assimpwrap-make
    assimpwrap-install

    assimpwrap-run   ## test G4DAE geometry import by Assimp


Building on GPU node
---------------------

* CUDA and OptiX need to be installed in consultation with sysadmin 

::

   raytrace-
   raytrace-cmake
   raytrace-make
  

Testing on headless node
-------------------------

Raytrace on compute only nodes by writing ppm files, which 
need no OpenGL context. The ppm are created and converted to png with::

   raytrace-
   raytrace-benchmark 

View the png on graphics capable node with::

   raytrace-
   raytrace-benchmark-get



Mouse interaction
-------------------

left mouse           
          Camera Rotate/Orbit 
middle mouse         
          Camera Pan/Truck
          (pan means rotate camera direction)
 
right mouse          
          Camera Dolly  
          (dolly means translate camera)

right mouse + shift  
          Camera FOV 


Mac trackpad
------------

While running MeshView, visit Preferences
and set to emulate 3-button mouse.

            
rotate
      drag around 

translate          
      hold option key down while dragging around

change fov
      hold shift key down while making a firm two finger drag up/down



Standard keystrokes:
  q Quit
  f Toggle full screen
  r Toggle continuous mode (progressive refinement, animation, or benchmark)
  R Set progressive refinement to never timeout and toggle continuous mode
  b Start/stop a benchmark
  d Toggle frame rate display
  s Save a frame to 'out.ppm'
  m Toggle memory usage printing
  c Print camera pose




Memory Requirements
---------------------

Default geometry selection requires approx 520MB of GPU memory free, otherwise crashes
as show below. To free up some GPU memory, close non-used windows/apps and 
sleep/wake cycle.::

    Time to load geometry: 2.87298 s.
    OptiX Error: Unknown error (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Insufficient device memory. GPU does not support paging., [16515528])
    delta:env blyth$ 
    delta:env blyth$ cu
    timestamp                Thu Feb  5 20:25:36 2015
    tag                      default
    name                     GeForce GT 750M
    compute capability       (3, 0)
    memory total             2.1G
    memory used              1.6G
    memory free              500.9M


* https://forums.adobe.com/thread/1326404



Code Layout
------------

Note that OptiX program .cu sources can be arranged 
however is convenient, they just need to be listed 
in CMakeLists.txt in order to get compiled by nvcc 
into .ptx placed in bdir/lib/ptx from which OptiX 
pulls the ptx and assembles it into 
the CUDA kernel.


Class Layout
-------------

::

    MeshViewer
        MeshScene
            SampleScene         


   OPTIX_SDK_DIR/sutil/SampleScene.h

        


Next Steps
------------


* measure merged meshes performance

* all parameters that impact geometry need to 
  be included into the accelcache digest : for proper identity 
  to avoid crashes from mismatched geometry/accel structure

  Digest needs to include:

  * geom query selection
  * geometry conversion maxdepth, extract this from query 
  * assimp import flags 
  * mesh merging approach control parameter

  Alternatively could make digest of actual geometry rather 
  than parameters.

* flesh out geometry selection, similar to g4daeview.sh 

* try instanced geometry with transforms and check performance

* add transparency 

* investigate OpenGL interop 

* better interactive controls 

  * yfov
  * tmin (scene_epsilon)
  * orthographic/projection

* get assimp to load material/surface extra properties

* test curand with OptiX 

* review chroma and try to shoe horn some aspects 
  into OptiX approach   

From 301 to 370b2
--------------------

Mesh handling consolidated into OptiXMesh::

    vimdiff OptiX_301_sample6.cpp OptiX_370b2_sample6.cpp


Very different loader structure, MeshBase.h::

    353   void loadDataFromObj( const std::string& filename );
    354 
    355   /**
    356    * Similar to loadFromObj() bur for .ply files
    357    */
    358   void loadFromPly( const std::string& filename );
    359 


std::string symbol mismatch
------------------------------

Mismatch between the libsutil.dylib symbols regarding std::string and those in MeshViewer.
To avoid this the compiler settings from the OptiX samples were adopted for MeshViewer.
::

    delta:raytrace blyth$ nm /usr/local/env/cuda/OptiX_370b2_sdk_install/lib/libsutil.dylib | c++filt  | grep GLUTDisplay::run
    0000000000009450 T GLUTDisplay::runBenchmarkNoDisplay()
    0000000000008e90 T GLUTDisplay::run(std::string const&, SampleScene*, GLUTDisplay::contDraw_E)

    delta:raytrace blyth$ nm /usr/local/env/cuda/OptiX_301/lib/libsutil.dylib | c++filt  | grep GLUTDisplay::run
    0000000000009ed0 T GLUTDisplay::runBenchmarkNoDisplay()
    00000000000098d0 T GLUTDisplay::run(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, SampleScene*, GLUTDisplay::contDraw_E)

::

    delta:raytrace blyth$ nm CMakeFiles/MeshViewer.dir/MeshViewer.cpp.o | c++filt | grep GLUTDisplay 
                     U GLUTDisplay::printUsage()
    000000000003fbb0 S GLUTDisplay::isBenchmark()
                     U GLUTDisplay::m_app_continuous_mode
                     U GLUTDisplay::m_cur_continuous_mode
                     U GLUTDisplay::run(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, SampleScene*, GLUTDisplay::contDraw_E)
                     U GLUTDisplay::init(int&, char**)
    delta:raytrace blyth$ 


* http://stackoverflow.com/questions/8454329/why-cant-clang-with-libc-in-c0x-mode-link-this-boostprogram-options-examp
* http://stackoverflow.com/questions/16352833/linking-with-clang-on-os-x-generates-lots-of-symbol-not-found-errors




EOU
}

raytrace-odir(){ echo $(local-base)/env/graphics/$(optix-name)/raytrace_out ; }
raytrace-bdir(){ echo $(local-base)/env/graphics/$(optix-name)/raytrace ; }
raytrace-sdir(){ echo $(env-home)/graphics/raytrace ; }

raytrace-cd(){  cd $(raytrace-sdir); }
raytrace-cu(){  cd $(raytrace-sdir)/cu; }
raytrace-scd(){  cd $(raytrace-sdir); }
raytrace-bcd(){  cd $(raytrace-bdir); }
raytrace-ocd(){  cd $(raytrace-odir); }

raytrace-src-dir(){ echo $(raytrace-sdir) ; }
raytrace-ptx-dir(){ echo $(raytrace-bdir)/lib/ptx ; }
raytrace-rng-dir(){ echo $(cudawrap-rng-dir) ; }

raytrace-ptx(){ ls -l $(raytrace-ptx-bdir) ; }
raytrace-rng(){ ls -l $(raytrace-rng-dir) ; }


raytrace-clean(){
   raytrace-wipe
   raytrace-cmake
   raytrace-make
   raytrace-run --g4dae $DAE_NAME_DYB
}

raytrace-env(){      
    elocal-  
    assimp-
    optix-
    optix-export
    export-
    export-export
    cudawrap-
}

raytrace-name(){ echo MeshViewer ; }



raytrace-wipe(){
   local bdir=$(raytrace-bdir)
   rm -rf $bdir
}


# hmm these settings not propagated from cmake -Dargs, 
# setting manually in CMakeLists.txt works
raytrace-curand(){   echo 1 ; }
raytrace-timeview(){ echo 0 ; }
raytrace-config(){ cat $(raytrace-bdir)/RayTraceConfigInc.h ; }

raytrace-cmake(){
   local iwd=$PWD

   local bdir=$(raytrace-bdir)
   mkdir -p $bdir

   raytrace-bcd
   raytrace-export
 
   cmake -DCMAKE_BUILD_TYPE=Debug \
         -DOptiX_INSTALL_DIR=$(optix-install-dir) \
         -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
          $(raytrace-sdir) 

   cd $iwd
}


raytrace-make(){
   local iwd=$PWD
   raytrace-bcd
   local rc

   make $*
   rc=$?

   cd $iwd
   [ $rc -ne 0 ] && echo $FUNCNAME ERROR && return 1 
   return 0
}




raytrace-bin(){ echo $(raytrace-bdir)/$(raytrace-name) ; }

raytrace-export()
{
   local q
   #q="name:__dd__Geometry__PMT__lvPmtHemi--pvPmtHemiVacuum0xc1340e8"
   #q="name:__dd__Geometry__PMT__lvPmtHemi0xc133740"
   #q="name:__dd__Geometry__PMT__lvPmtHemi"
   #q="index:0"
   #q="index:1"
   #q="index:2"
   #q="index:3147"
   #q="index:3153"
   #q="index:3154"
   #q="index:3155"
   #q="range:2153:12221,merge:1" 
   #q="range:3153:12221,merge:0" 
   #q="range:3153:12221,merge:1" 
   #q="index:4998"
   #q="index:5000"
   #q="range:5000:5010"
   #q="range:5000:8000"
   q="range:3153:12221"
   #q="range:4998:5998,merge:1" 
   #q="range:4998:5998,merge:0" 
   export RAYTRACE_QUERY=$q


   unset RAYTRACE_GGCTRL 
   export RAYTRACE_GGCTRL=""

   unset RAYTRACE_RNG_DIR
   unset RAYTRACE_SRC_DIR
   unset RAYTRACE_PTX_DIR

   export RAYTRACE_RNG_DIR=$(raytrace-rng-dir)
   export RAYTRACE_SRC_DIR=$(raytrace-src-dir)
   export RAYTRACE_PTX_DIR=$(raytrace-ptx-dir)


   env | grep RAYTRACE

}


raytrace-run(){
  raytrace-export 
  if [ -n "$DEBUG" ]; then 
      $DEBUG $(raytrace-bin) -- $* 
  else
      $(raytrace-bin) $* 
  fi
}

raytrace-run-manual(){
  local bin=$(raytrace-bin)
  local dir=$(dirname $bin)
  raytrace-export 
  DYLD_LIBRARY_PATH=$dir $bin $*
}



raytrace--(){
  echo $FUNCNAME $*

  #raytrace-wipe 
  #raytrace-cmake

  raytrace-make
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

  raytrace-run --g4dae $DAE_NAME_DYB $*
}



raytrace-v(){
  raytrace-- --cache --g4dae $DAE_NAME_DYB_NOEXTRA $*
}

raytrace-x(){
  #raytrace-- --cache --g4dae $DAE_NAME_DYB --dim=1024x768 --rng-cuda $*
  raytrace-- --cache --g4dae $DAE_NAME_DYB --dim=1024x768 --rng-optix --ortho $*
  #raytrace-- --cache --g4dae $DAE_NAME_DYB --dim=1024x768 --rng-optix  $*
}


raytrace-args(){
    echo --cache --g4dae $DAE_NAME_DYB --dim=1024x768 --rng-optix --ortho $*
}

raytrace-x-manual(){
  raytrace-make
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 
  raytrace-export
  raytrace-run-manual --cache --g4dae $DAE_NAME_DYB $*
}



raytrace-lldb(){
  DEBUG=lldb raytrace-x $*
}





raytrace-benchmark-name(){ echo benchmark ; }
raytrace-benchmark-path(){ echo $(raytrace-odir)/$(raytrace-benchmark-name) ; }
raytrace-benchmark-scaling(){ echo 1 ; }

raytrace-benchmark()
{
   local path=$(raytrace-benchmark-path)
   local dir=$(dirname $path)
   mkdir -p $dir


   if [ "$(raytrace-benchmark-scaling)" == "1" ] ; then
       CUDA_VISIBLE_DEVICES=0 raytrace-v -a --benchmark-no-display=1x2 --save-frames=$path $*
       CUDA_VISIBLE_DEVICES=1 raytrace-v -a --benchmark-no-display=1x2 --save-frames=$path $*
       raytrace-v -a --benchmark-no-display=1x2 --save-frames=$path $*
   else

       raytrace-v -a --benchmark-no-display=1x2 --save-frames=$path $*

   fi

   # TODO: tidy output and record benchmark info named logs 

   raytrace-benchmark-convert
}



raytrace-benchmark-convert()
{ 
   local ppm=$(raytrace-benchmark-path).ppm
   local png=${ppm/.ppm}.png

   [ ! -f "$ppm" ] && echo $msg no ppm $ppm && return

   if [ -n "$(which convert)" ]
   then 
       echo $msg convert-ing ppm $ppm to png $png
       convert $ppm $png 
   else
       echo $msg wpng-ing ppm $ppm to png $png
       libpng-
       cat $ppm | libpng-wpng > $png
       open $png
   fi 
}


raytrace-benchmark-optix(){ echo $(optix-linux-name 370) ; }
raytrace-benchmark-node(){  echo L6 ; }
raytrace-benchmark-get()
{
   local tag=$(raytrace-benchmark-node)
   local rem=$(NODE_TAG=$tag OPTIX_NAME=$(raytrace-benchmark-optix) raytrace-benchmark-path) 
   local loc=$(raytrace-benchmark-path)
   local cmd="scp $tag:$rem.png $loc.$tag.png"
   echo $cmd
   eval $cmd
   open $loc.$tag.png 
}


