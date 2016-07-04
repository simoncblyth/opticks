# === func-gen- : optix/optixthrustuse/optixthrustuse fgp optix/optixthrustuse/optixthrustuse.bash fgn optixthrustuse fgh optix/optixthrustuse
optixthrustuse-src(){      echo optix/optixthrustuse/optixthrustuse.bash ; }
optixthrustuse-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(optixthrustuse-src)} ; }
optixthrustuse-vi(){       vi $(optixthrustuse-source) ; }
optixthrustuse-env(){      elocal- ; }
optixthrustuse-usage(){ cat << EOU

OptiXThrustUse
==========================

Testing usage of library-ified optixthrust-

cmake testing
-------------

::

   optixthrustuse-;optixthrustuse-wipe;VERBOSE=1 optixthrustuse--


EOU
}
optixthrustuse-dir(){ echo $(opticks-home)/optix/optixthrustuse ; }
optixthrustuse-cd(){  cd $(optixthrustuse-dir); }

optixthrustuse-env(){      
   elocal- 
   cuda-
   optix-
}

optixthrustuse-sdir(){   echo $(optixthrustuse-dir) ; }
optixthrustuse-bdir(){   echo /tmp/optixthrustuse.bdir ; }
optixthrustuse-idir(){   echo /tmp/optixthrustuse.idir ; }
optixthrustuse-scd(){    cd $(optixthrustuse-sdir) ; }
optixthrustuse-bcd(){    cd $(optixthrustuse-bdir) ; }
optixthrustuse-bin(){    echo /tmp/optixthrustuse ; }


optixthrustuse-cmake()
{
   local iwd=$PWD

   local bdir=$(optixthrustuse-bdir)
   mkdir -p $bdir

   optix-export
  
   optixthrustuse-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCMAKE_INSTALL_PREFIX=$(optixthrustuse-idir) \
       $(optixthrustuse-sdir)

   cd $iwd 
}

optixthrustuse-make(){
   local iwd=$PWD

   optixthrustuse-bcd 
   make $*

   cd $iwd 
}

optixthrustuse-run(){

   local idir=$(optixthrustuse-idir)
   local ibin=$idir/bin/OptiXThrustUse

   $ibin $*
}

optixthrustuse-wipe(){
   local bdir=$(optixthrustuse-bdir)
   rm -rf $bdir 
}

optixthrustuse--()
{
   local bdir=$(optixthrustuse-bdir)
   [ ! -d "$bdir" ] && optixthrustuse-cmake

   optixthrustuse-make 
   optixthrustuse-make install
   #optixthrustuse-run
}

