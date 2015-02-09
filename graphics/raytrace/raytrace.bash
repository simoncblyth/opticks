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
raytrace-scd(){  cd $(raytrace-sdir); }
raytrace-bcd(){  cd $(raytrace-bdir); }
raytrace-ocd(){  cd $(raytrace-odir); }

raytrace-env(){      
    elocal-  
    assimp-
    optix-
    optix-export
    export-
    export-export
}

raytrace-name(){ echo MeshViewer ; }

raytrace-wipe(){
   local bdir=$(raytrace-bdir)
   rm -rf $bdir
}

raytrace-cmake(){
   local iwd=$PWD

   local bdir=$(raytrace-bdir)
   mkdir -p $bdir

   raytrace-bcd
  
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
   q="range:3153:12221,merge:1" 
   #q="index:4998"
   #q="range:4998:5998,merge:1" 
   #q="range:4998:5998,merge:0" 
   export RAYTRACE_QUERY=$q
}


raytrace-run(){ $DEBUG $(raytrace-bin) $* ; }

raytrace--(){
  raytrace-make
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 
  raytrace-export 
  raytrace-run $*
}

raytrace-v(){
  raytrace-- --cache --g4dae $DAE_NAME_DYB_NOEXTRA $*
}

raytrace-lldb(){
  DEBUG=lldb raytrace-v $*
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


raytrace-benchmark-optix(){ echo $(optix-linux-name 351) ; }
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


