# === func-gen- : graphics/raytrace/raytrace fgp graphics/raytrace/raytrace.bash fgn raytrace fgh graphics/raytrace
raytrace-src(){      echo graphics/raytrace/raytrace.bash ; }
raytrace-source(){   echo ${BASH_SOURCE:-$(env-home)/$(raytrace-src)} ; }
raytrace-vi(){       vi $(raytrace-source) ; }
raytrace-usage(){ cat << EOU

Test Combination of Assimp and OptiX
======================================

::

    raytrace-;raytrace-- __dd__Geometry__AD__lvOIL0xbf5e0b8 --dim=256x256


* press "ctrl" and drag up/down to zoom out/on 

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
   #q="index:3153"
   #q="index:3154"
   q="index:3155"
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

raytrace-view(){
  raytrace-make
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

  raytrace-export 
  $DEBUG $(raytrace-view-bin) $* 
}





raytrace-geo-lldb(){
  DEBUG=lldb raytrace-geo
}
