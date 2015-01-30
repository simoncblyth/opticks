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


EOU
}
raytrace-bdir(){ echo $(local-base)/env/graphics/raytrace ; }
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
   #q="range:3153:12221" 
   q="index:4998"
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
