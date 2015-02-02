# === func-gen- : graphics/raytrace/raytrace fgp graphics/raytrace/raytrace.bash fgn raytrace fgh graphics/raytrace
raytrace-src(){      echo graphics/raytrace/raytrace.bash ; }
raytrace-source(){   echo ${BASH_SOURCE:-$(env-home)/$(raytrace-src)} ; }
raytrace-vi(){       vi $(raytrace-source) ; }
raytrace-usage(){ cat << EOU

Test Combination of Assimp and OptiX
======================================

::

    raytrace-;raytrace-v -n

         # MeshViewer with accelcache 

    raytrace-;raytrace-- __dd__Geometry__AD__lvOIL0xbf5e0b8 --dim=256x256

         # old RayTrace has no accelcache support 


* press "ctrl" and drag up/down to zoom out/in 

* for fps display press "r" and then "d"


Next Steps
------------

* try merged meshes and check performance

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


Build Warnings
----------------

::

    [ 22%] Building NVCC ptx file MeshViewer_generated_TriangleMesh.cu.ptx
    /Users/blyth/env/graphics/raytrace/TriangleMesh.cu(34): Warning: Cannot tell what pointer points to, assuming global memory space
    /Users/blyth/env/graphics/raytrace/TriangleMesh.cu(34): Warning: Cannot tell what pointer points to, assuming global memory space
    /Users/blyth/env/graphics/raytrace/TriangleMesh.cu(34): Warning: Cannot tell what pointer points to, assuming global memory space
    /Users/blyth/env/graphics/raytrace/TriangleMesh.cu(36): Warning: Cannot tell what pointer points to, assuming global memory space


    /Users/blyth/env/graphics/raytrace/MeshViewer.cpp:12:10: fatal error: 'PlyLoader.h' file not found
    #include <PlyLoader.h>


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


::

    delta:raytrace blyth$ nm /usr/local/env/cuda/OptiX_370b2_sdk_install/lib/libsutil.dylib | grep ptxpath | c++filt
    000000000003ebd8 bool guard variable for SampleScene::ptxpath(std::string const&, std::string const&)::path
    000000000002a8f0 T SampleScene::ptxpath(std::string const&, std::string const&)
    000000000003ebd0 bool SampleScene::ptxpath(std::string const&, std::string const&)::path
    delta:raytrace blyth$ 

::

    delta:raytrace blyth$ nm /usr/local/env/cuda/OptiX_301/lib/libsutil.dylib | grep ptxpath  | c++filt
    0000000000038f50 bool guard variable for SampleScene::ptxpath(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&)::path
    00000000000286e0 T SampleScene::ptxpath(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&)
    0000000000038f38 bool SampleScene::ptxpath(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&)::path
    delta:raytrace blyth$ 




Mismatch between the libsutil.dylib symbols regarding std::string and those in MeshViewer 

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


::

    delta:OptiX_370b2_sdk blyth$ optix-diff CMakeLists.txt
    diff /Developer/OptiX_301/SDK/CMakeLists.txt /Developer/OptiX/SDK/CMakeLists.txt
    82c82
    < cmake_minimum_required(VERSION 2.6.3 FATAL_ERROR)
    ---
    > cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
    121a122,127
    > # For Xcode 5, gcc is actually clang, so we have to tell CUDA to treat the compiler as
    > # clang, so that it doesn't mistake it for something else.
    > if(USING_CLANG_C)
    >   set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
    > endif()
    > 
    204c210
    <   if ( USING_GCC AND NOT APPLE)
    ---
    >   if ( USING_GNU_C AND NOT APPLE)
    260a267,269
    >   if(USING_GNU_CXX)
    >     target_link_libraries( ${target_name} m ) # Explicitly link against math library (C samples don't do that by default)
    >   endif()



* http://stackoverflow.com/questions/16352833/linking-with-clang-on-os-x-generates-lots-of-symbol-not-found-errors




EOU
}
raytrace-bdir(){ echo $(local-base)/env/graphics/$(optix-name)/raytrace ; }
raytrace-sdir(){ echo $(env-home)/graphics/raytrace ; }
raytrace-cd(){  cd $(raytrace-sdir); }
raytrace-scd(){  cd $(raytrace-sdir); }
raytrace-bcd(){  cd $(raytrace-bdir); }

raytrace-env(){      
    elocal-  
    assimp-
    optix-
    optix-export
    export-
    export-export
}

raytrace-name(){ echo RayTrace ; }


raytrace-wipe(){

   local bdir=$(raytrace-bdir)
   rm -rf $bdir

}

raytrace-cmake(){
   local iwd=$PWD

   local bdir=$(raytrace-bdir)
   mkdir -p $bdir

   raytrace-bcd
  
   cmake -DCMAKE_BUILD_TYPE=Debug -DOptiX_INSTALL_DIR=$(optix-install-dir) -DCUDA_NVCC_FLAGS="-ccbin /usr/bin/clang --use_fast_math" $(raytrace-sdir) 

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



raytrace-geo-bin(){ echo $(raytrace-bdir)/AssimpGeometryTest ; }
raytrace-view-bin(){ echo $(raytrace-bdir)/MeshViewer ; }
raytrace-bin(){ echo $(raytrace-bdir)/$(raytrace-name) ; }

raytrace-run(){ $LLDB $(raytrace-bin) $* ; }
raytrace-dbg(){ LLDB=lldb raytrace-run $* ; }


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
   q="range:3153:12221" 
   #q="index:4998"
   export RAYTRACE_QUERY=$q
}

raytrace--(){
  raytrace-make
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 
  raytrace-export 
  raytrace-run $*
}

raytrace-geo(){
  raytrace-make
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

  raytrace-export 
  $DEBUG $(raytrace-geo-bin) $* 
}

raytrace-v-(){
  raytrace-make
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

  raytrace-export 
  $DEBUG $(raytrace-view-bin) $* 
}

raytrace-v(){
  raytrace-v- --cache --g4dae $DAE_NAME_DYB_NOEXTRA $*
}

raytrace-o(){
  raytrace-v- --cache  $*
}





raytrace-geo-lldb(){
  DEBUG=lldb raytrace-geo
}
