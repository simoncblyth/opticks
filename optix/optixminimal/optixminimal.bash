# === func-gen- : optix/optixminimal/optixminimal fgp optix/optixminimal/optixminimal.bash fgn optixminimal fgh optix/optixminimal
optixminimal-src(){      echo optix/optixminimal/optixminimal.bash ; }
optixminimal-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(optixminimal-src)} ; }
optixminimal-vi(){       vi $(optixminimal-source) ; }
optixminimal-usage(){ cat << EOU

optixminimal-thrust **FAILING**

   Thrust fails to see what OptiX writes into the buffer for unknown reasons

   Over in *optixthrust-* succeed to get Thrust too see what OptiX wrote.
   The code is similar but project layout is very different, or
   it could be that float4 works but unsigned char doesnt ?



EOU
}
optixminimal-dir(){ echo $(opticks-home)/optix/optixminimal ; }
optixminimal-cd(){  cd $(optixminimal-dir); }

optixminimal-env(){      
   elocal- 
   cuda-
   optix-
}

optixminimal-name(){ echo ${OPTIXMINIMAL_NAME:-optixminimal} ; }
optixminimal-compiler(){ echo ${OPTIXMINIMAL_COMPILER:-clang} ; }
optixminimal-objs(){ echo ${OPTIXMINIMAL_OBJS:-""} ; }
optixminimal-bin(){ echo /tmp/$(optixminimal-name) ; }
optixminimal-ptxdir(){ echo /tmp/ptxdir ; }
optixminimal-make(){

   local msg="$FUNCNAME : "
   optixminimal-cd

   local bin=$(optixminimal-bin)
   local name=$(optixminimal-name)
   local compiler="$(optixminimal-compiler)"
   local objs="$(optixminimal-objs)" 

   echo $msg compiler $compiler name $name bin $bin

   clang $name.cpp $objs -o $bin \
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

   local name=${1:-minimal}

   optixminimal-cd
   local ptxdir=$(optixminimal-ptxdir)
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





optixminimal-bdir(){ echo /tmp/optixminimal.bdir ; }

optixminimal-nvcc(){
   local name=${1:-thrust_simple}
   optixminimal-cd
   local bdir=$(optixminimal-bdir)
   mkdir -p $bdir 
   nvcc $name.cu -c -o $bdir/$name.o
}

optixminimal-thrust(){
   local name=thrust_simple
   optixminimal-nvcc $name
   optixminimal-ptx-make minimal

   local bdir=$(optixminimal-bdir)
   OPTIXMINIMAL_NAME=optixminimal_thrust OPTIXMINIMAL_OBJS="$bdir/$name.o" optixminimal-make
}
optixminimal-thrust-run(){
   OPTIXMINIMAL_NAME=optixminimal_thrust optixminimal-run
}



