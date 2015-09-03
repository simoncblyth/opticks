# === func-gen- : optix/optixthrust/optixthrust fgp optix/optixthrust/optixthrust.bash fgn optixthrust fgh optix/optixthrust
optixthrust-src(){      echo optix/optixthrust/optixthrust.bash ; }
optixthrust-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optixthrust-src)} ; }
optixthrust-vi(){       vi $(optixthrust-source) ; }
optixthrust-env(){      elocal- ; }
optixthrust-usage(){ cat << EOU


compilation via cmake is yielding many thousands of warnings



EOU
}
optixthrust-dir(){ echo $(env-home)/optix/optixthrust ; }
optixthrust-cd(){  cd $(optixthrust-dir); }

optixthrust-env(){      
   elocal- 
   cuda-
   optix-
}

optixthrust-ptxdir(){ echo /tmp/optixthrust.ptxdir ; }
optixthrust-sdir(){   echo $(optixthrust-dir) ; }
optixthrust-bdir(){   echo /tmp/optixthrust.bdir ; }
optixthrust-cdir(){   echo /tmp/optixthrust.cdir ; }
optixthrust-idir(){   echo /tmp/optixthrust.idir ; }
optixthrust-ccd(){    cd $(optixthrust-cdir) ; }
optixthrust-bin(){    echo /tmp/optixthrust ; }

optixthrust-prep()
{
   local ptxdir=$(optixthrust-ptxdir)
   mkdir -p $ptxdir
   local bdir=$(optixthrust-bdir)
   mkdir -p $bdir 

   optixthrust-cd 
}

optixthrust-ptx-make(){
   local name=${1:-minimal_float4}
   local msg="$FUNCNAME : "
   optixthrust-prep
   local out=$(optixthrust-ptxdir)/$name.ptx
   echo $msg $name.cu to $out 
   nvcc -ptx $name.cu -o $out -I$(optix-prefix)/include
}

optixthrust-nvcc-make(){
   local msg="$FUNCNAME : "
   local name=${1:-optixthrust_postprocess}
   optixthrust-prep
   local out=$(optixthrust-bdir)/$name.cu.o
   echo $msg $name.cu to $out 
   nvcc $name.cu -c -o $out -I$(optix-prefix)/include
}

optixthrust-o-make(){
   local msg="$FUNCNAME : "
   local name=${1:-main}
   optixthrust-prep
   local out=$(optixthrust-bdir)/$name.cpp.o
   echo $msg $name.cpp to $out 

   clang $name.cpp -c -o $out \
         -I$(cuda-prefix)/include \
         -I$(optix-prefix)/include
}


optixthrust-link(){

   local objs="$*" 
   local msg="$FUNCNAME : "
   optixthrust-cd

   local bin=$(optixthrust-bin)

   echo $mss objs $objs bin $bin

   clang $objs -o $bin \
         -L$(cuda-prefix)/lib -lcudart.7.0  \
         -L$(optix-prefix)/lib64 -loptix.3.8.0 -loptixu.3.8.0  \
         -lc++ \
         -Xlinker -rpath -Xlinker $(cuda-prefix)/lib \
         -Xlinker -rpath -Xlinker $(optix-prefix)/lib64 
}

optixthrust-make-manual()
{
    optixthrust-ptx-make minimal_float4
    # not linked : loaded by OptiX at runtime

    optixthrust-nvcc-make optixthrust_postprocess

    optixthrust-o-make main

    optixthrust-o-make optixthrust

    local bdir=$(optixthrust-bdir)
    optixthrust-link $bdir/main.cpp.o $bdir/optixthrust.cu.o $bdir/optixthrust.cpp.o
}


optixthrust-run-manual(){
   local bin=$(optixthrust-bin)
   PTXDIR=$(optixthrust-ptxdir) $bin
}


optixthrust-cmake()
{
   local iwd=$PWD

   local cdir=$(optixthrust-cdir)
   mkdir -p $cdir

   optix-export
  
   optixthrust-ccd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(optixthrust-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(optixthrust-sdir)

   cd $iwd 

}


optixthrust-make(){
   local iwd=$PWD

   optixthrust-ccd 
   make $*

   cd $iwd 
}

optixthrust-run(){

   local cdir=$(optixthrust-cdir)
   local ptxdir=$cdir/lib/ptx
   local idir=$(optixthrust-idir)
   local ibin=$idir/bin/OptiXThrustMinimal

   PTXDIR=$ptxdir $ibin $*
}


optixthrust-wipe(){
   local cdir=$(optixthrust-cdir)
   rm -rf $cdir 
}

optixthrust--()
{
   local cdir=$(optixthrust-cdir)
   [ ! -d "$cdir" ] && optixthrust-cmake

   optixthrust-make 
   optixthrust-make install
   optixthrust-run
}

