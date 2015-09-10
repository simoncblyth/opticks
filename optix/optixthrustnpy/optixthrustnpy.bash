# === func-gen- : optix/optixthrustnpy/optixthrustnpy fgp optix/optixthrustnpy/optixthrustnpy.bash fgn optixthrustnpy fgh optix/optixthrustnpy
optixthrustnpy-src(){      echo optix/optixthrustnpy/optixthrustnpy.bash ; }
optixthrustnpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optixthrustnpy-src)} ; }
optixthrustnpy-vi(){       vi $(optixthrustnpy-source) ; }
optixthrustnpy-env(){      elocal- ; }
optixthrustnpy-usage(){ cat << EOU

OptiX/CUDA/Thrust + NPY Interop
==================================

The *optixthrustnpy-* package is a testing ground for OptiX/CUDA/Thrust interop,
together with NPY but without any *OpenGL* complications.

see also
---------

* optixthrust-

OptiX with Thrust linking difficulties
----------------------------------------

Some trickery regards optix vector types and CUDA ones ? 
Means have to include vector_types.h before optix headers otherwise
the vector types get stuffed into optix namespace.

See https://devtalk.nvidia.com/default/topic/574078/?comment=3896854

Before doing this there is symbol discrepancy and hence linker error::

    simon:optixthrustnpy blyth$ optixthrustnpy-;optixthrustnpy-symbols0 | grep OBuf
    /tmp/optixthrustnpy.cdir/lib/ptx/OptiXThrustNPY_generated_OBuf_.cu.o
    00000000000006c0 unsigned short __sti____cudaRegisterAll_40_tmpxft_000128ff_00000000_7_OBuf__cpp1_ii_cf3c6bc2()
    0000000000049cf0 S OBuf<float4>::getDevicePtr()
    0000000000049e30 S OBuf<float4>::dump(char const*, unsigned int, unsigned int)
    0000000000049d80 S OBuf<float4>::getSize()
    0000000000049cc0 S OBuf<float4>::OBuf(optix::Handle<optix::BufferObj>&)
    0000000000049c50 S OBuf<float4>::OBuf(optix::Handle<optix::BufferObj>&)

    simon:optixthrustnpy blyth$ optixthrustnpy-;optixthrustnpy-symbols1 | grep OBuf
                     U OBuf<optix::float4>::dump(char const*, unsigned int, unsigned int)
                     U OBuf<optix::float4>::OBuf(optix::Handle<optix::BufferObj>&)
    0000000000028310 S OBuf<optix::float4>::~OBuf()
    000000000002b010 S OBuf<optix::float4>::~OBuf()
    simon:optixthrustnpy blyth$ 

After reordering header includes, thrust headers persumably include vector_types too
there is agreement and linking succeeds::

    simon:thrustrap blyth$ optixthrustnpy-;optixthrustnpy-symbols0 | grep OBuf
    /tmp/optixthrustnpy.cdir/lib/ptx/OptiXThrustNPY_generated_OBuf_.cu.o
    00000000000006c0 unsigned short __sti____cudaRegisterAll_40_tmpxft_00012e19_00000000_7_OBuf__cpp1_ii_cf3c6bc2()
    00000000000615c0 S OBuf<float4>::getDevicePtr()
    0000000000061700 S OBuf<float4>::dump(char const*, unsigned int, unsigned int)
    0000000000061650 S OBuf<float4>::getSize()
    0000000000061590 S OBuf<float4>::OBuf(optix::Handle<optix::BufferObj>&)
    0000000000061520 S OBuf<float4>::OBuf(optix::Handle<optix::BufferObj>&)
    simon:thrustrap blyth$ 
    simon:thrustrap blyth$ optixthrustnpy-;optixthrustnpy-symbols1 | grep OBuf
                     U OBuf<float4>::dump(char const*, unsigned int, unsigned int)
                     U OBuf<float4>::OBuf(optix::Handle<optix::BufferObj>&)
    0000000000028500 S OBuf<float4>::~OBuf()
    000000000002b200 S OBuf<float4>::~OBuf()
    simon:thrustrap blyth$ 


EOU
}
optixthrustnpy-dir(){ echo $(env-home)/optix/optixthrustnpy ; }
optixthrustnpy-cd(){  cd $(optixthrustnpy-dir); }

optixthrustnpy-env(){      
   elocal- 
   cuda-
   optix-
}

optixthrustnpy-ptxdir(){ echo /tmp/optixthrustnpy.ptxdir ; }
optixthrustnpy-sdir(){   echo $(optixthrustnpy-dir) ; }
optixthrustnpy-bdir(){   echo /tmp/optixthrustnpy.bdir ; }
optixthrustnpy-cdir(){   echo /tmp/optixthrustnpy.cdir ; }
optixthrustnpy-idir(){   echo /tmp/optixthrustnpy.idir ; }
optixthrustnpy-ccd(){    cd $(optixthrustnpy-cdir) ; }
optixthrustnpy-bin(){    echo /tmp/optixthrustnpy ; }

optixthrustnpy-prep()
{
   local ptxdir=$(optixthrustnpy-ptxdir)
   mkdir -p $ptxdir
   local bdir=$(optixthrustnpy-bdir)
   mkdir -p $bdir 

   optixthrustnpy-cd 
}


optixthrustnpy-cmake()
{
   local iwd=$PWD

   local cdir=$(optixthrustnpy-cdir)
   mkdir -p $cdir

   optix-export
  
   optixthrustnpy-ccd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(optixthrustnpy-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(optixthrustnpy-sdir)

   cd $iwd 
}


optixthrustnpy-make(){
   local iwd=$PWD

   optixthrustnpy-ccd 
   make $*

   cd $iwd 
}

optixthrustnpy-run(){

   local cdir=$(optixthrustnpy-cdir)
   local ptxdir=$cdir/lib/ptx
   local idir=$(optixthrustnpy-idir)
   local ibin=$idir/bin/OptiXThrustNPY

   PTXDIR=$ptxdir $ibin $*
}


optixthrustnpy-wipe(){
   local cdir=$(optixthrustnpy-cdir)
   rm -rf $cdir 
}

optixthrustnpy--()
{
   local cdir=$(optixthrustnpy-cdir)
   [ ! -d "$cdir" ] && optixthrustnpy-cmake

   optixthrustnpy-make 
   optixthrustnpy-make install
   #optixthrustnpy-run
}


optixthrustnpy-symbols0()
{
   optixthrustnpy-symbols- /tmp/optixthrustnpy.cdir/lib/ptx
}
optixthrustnpy-symbols1()
{
   optixthrustnpy-symbols- /tmp/optixthrustnpy.cdir/CMakeFiles/OptiXThrustNPY.dir
}

optixthrustnpy-symbols-()
{
   local dir=$1
   local objs=$dir/*.o
   for obj in $objs ; do 
       echo $obj 
       nm $obj | c++filt 
   done 
}


