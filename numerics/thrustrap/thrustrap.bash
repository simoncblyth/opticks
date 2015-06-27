# === func-gen- : numerics/thrustrap/thrustrap fgp numerics/thrustrap/thrustrap.bash fgn thrustrap fgh numerics/thrustrap
thrustrap-src(){      echo numerics/thrustrap/thrustrap.bash ; }
thrustrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(thrustrap-src)} ; }
thrustrap-vi(){       vi $(thrustrap-source) ; }
thrustrap-env(){      elocal- ; }
thrustrap-usage(){ cat << EOU


Observations from the below match my experience

* https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks

CUDA 5.5 has problems with c++11 ie libc++ on Mavericks, 
can only get to compile and run without segv only by 

* targetting the older libstdc++::

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -stdlib=libstdc++")

* not linking to other libs built against c++11 ie libc++

delta:tests blyth$ otool -L /usr/local/env/numerics/thrustrap/bin/ThrustEngineTest
/usr/local/env/numerics/thrustrap/bin/ThrustEngineTest:
    @rpath/libcudart.5.5.dylib (compatibility version 0.0.0, current version 5.5.28)
    /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 60.0.0)
    /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)



CUDA 7 to the rescue ?
------------------------

* http://devblogs.nvidia.com/parallelforall/cuda-7-release-candidate-feature-overview/




Can a C interface provide a firewall to allow interop between compilers ?
===========================================================================






EOU
}

thrustrap-sdir(){ echo $(env-home)/numerics/thrustrap ; }
thrustrap-idir(){ echo $(local-base)/env/numerics/thrustrap ; }
thrustrap-bdir(){ echo $(thrustrap-idir).build ; }

thrustrap-scd(){  cd $(thrustrap-sdir); }
thrustrap-cd(){   cd $(thrustrap-sdir); }

thrustrap-icd(){  cd $(thrustrap-idir); }
thrustrap-bcd(){  cd $(thrustrap-bdir); }
thrustrap-name(){ echo ThrustRap ; }

thrustrap-wipe(){
   local bdir=$(thrustrap-bdir)
   rm -rf $bdir
}

thrustrap-env(){  
   elocal- 
   cuda-
   cuda-export
   #optix-
   #optix-export
   thrust-
   thrust-export 
}

thrustrap-cmake(){
   local iwd=$PWD
   local msg="=== $FUNCNAME : "
   local bdir=$(thrustrap-bdir)
   mkdir -p $bdir
  
   thrustrap-bcd 

   local flags=$(cuda-nvcc-flags)
   echo $msg using CUDA_NVCC_FLAGS $flags

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(thrustrap-idir) \
       -DCUDA_NVCC_FLAGS="$flags" \
       $(thrustrap-sdir)

   cd $iwd
}

thrustrap-make(){
   local iwd=$PWD

   thrustrap-bcd
   make $*

   cd $iwd
}

thrustrap-install(){
   thrustrap-make install
}

thrustrap-bin(){ echo $(thrustrap-idir)/bin/$(thrustrap-name)Test ; }
thrustrap-export()
{ 
   echo -n 
}
thrustrap-run(){
   local bin=$(thrustrap-bin)
   thrustrap-export
   $bin $*
}



thrustrap--()
{
    thrustrap-wipe
    thrustrap-cmake
    thrustrap-make
    thrustrap-install

}



