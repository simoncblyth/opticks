# === func-gen- : optix/optixminimal/optixminimal fgp optix/optixminimal/optixminimal.bash fgn optixminimal fgh optix/optixminimal
optixminimal-src(){      echo optix/optixminimal/optixminimal.bash ; }
optixminimal-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optixminimal-src)} ; }
optixminimal-vi(){       vi $(optixminimal-source) ; }
optixminimal-usage(){ cat << EOU



RPATH hookup

::

    simon:lib64 blyth$ otool -L liboptix.3.8.0.dylib
    liboptix.3.8.0.dylib:
        @rpath/liboptix.1.dylib (compatibility version 1.0.0, current version 3.8.0)
        /System/Library/Frameworks/AGL.framework/Versions/A/AGL (compatibility version 1.0.0, current version 1.0.0)
        /System/Library/Frameworks/OpenGL.framework/Versions/A/OpenGL (compatibility version 1.0.0, current version 1.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 65.1.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 169.3.0)


EOU
}
optixminimal-dir(){ echo $(env-home)/optix/optixminimal ; }
optixminimal-cd(){  cd $(optixminimal-dir); }
optixminimal-mate(){ mate $(optixminimal-dir) ; }

optixminimal-env(){      
   elocal- 
   cuda-
   optix-
}

optixminimal-bin(){ echo /tmp/optixminimal ; }
optixminimal-ptxdir(){ echo /tmp/ptxdir ; }
optixminimal-make(){

   local msg="$FUNCNAME : "
   optixminimal-cd

   local bin=$(optixminimal-bin)
   local name=$(basename $bin)

   echo $msg bin $bin

   clang $name.cpp -o $bin \
         -I$(cuda-prefix)/include   \
         -I$(optix-prefix)/include  \
         -L$(cuda-prefix)/lib -lcudart.7.0  \
         -L$(optix-prefix)/lib64 -loptix.3.8.0 -loptixu.3.8.0  \
         -lc++ \
         -Xlinker -rpath -Xlinker $(cuda-prefix)/lib \
         -Xlinker -rpath -Xlinker $(optix-prefix)/lib64 

   # embed multiple rpath into the binary such that runtime 
   # knows to look for the dylib in the same place as the linker did 
   # avoiding the DYLD_LIBRARY_PATH kludge
}

optixminimal-ptx-make(){

   optixminimal-cd
   local ptxdir=$(optixminimal-ptxdir)
   local name=minimal
   mkdir -p $ptxdir
   nvcc -ptx $name.cu -o $ptxdir/$name.ptx \
         -I$(optix-prefix)/include
}


optixminimal-kludge-run(){
   local bin=$(optixminimal-bin)
   PTXDIR=$(optixminimal-ptxdir) DYLD_LIBRARY_PATH=$(optix-prefix)/lib64:$(cuda-prefix)/lib $bin
}

optixminimal-run(){
   local bin=$(optixminimal-bin)
   PTXDIR=$(optixminimal-ptxdir) $bin
}


