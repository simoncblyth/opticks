# === func-gen- : cuda/ocelot fgp cuda/ocelot.bash fgn ocelot fgh cuda
ocelot-src(){      echo cuda/ocelot.bash ; }
ocelot-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ocelot-src)} ; }
ocelot-vi(){       vi $(ocelot-source) ; }
ocelot-env(){      elocal- ; }
ocelot-usage(){ cat << EOU

GPU OCELOT, CUDA EMULATION
=============================

* :google:`CUDA emulation ocelot`
* http://code.google.com/p/gpuocelot/
* http://code.google.com/p/gpuocelot/wiki/Installation

May need from gcc 4.5+

* http://barefeg.wordpress.com/2012/06/16/how-to-install-gpuocelot-in-ubuntu-12-04/


Belle7 gcc44 g++44 alternates
-------------------------------

Try most recent gcc/g++ that SL5.1 repo provides, namely 44

Alternates Setup, see gcc-


Other Emulation Approaches
----------------------------

OCELOT is painful to install

* http://www.drdobbs.com/parallel/running-cuda-code-natively-on-x86-proces/231500166
* http://www.pgroup.com/



EOU
}
ocelot-dir(){ echo $(local-base)/env/cuda/gpuocelot ; }
ocelot-cd(){  cd $(ocelot-dir); }
ocelot-mate(){ mate $(ocelot-dir) ; }
ocelot-get(){
   local dir=$(dirname $(ocelot-dir)) &&  mkdir -p $dir && cd $dir

   svn checkout http://gpuocelot.googlecode.com/svn/trunk/ gpuocelot 

}
