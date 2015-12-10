# === func-gen- : graphics/ggeoview/cfg4/cfg4 fgp graphics/ggeoview/cfg4/cfg4.bash fgn cfg4 fgh graphics/ggeoview/cfg4
cfg4-src(){      echo graphics/ggeoview/cfg4/cfg4.bash ; }
cfg4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cfg4-src)} ; }
cfg4-vi(){       vi $(cfg4-source) ; }
cfg4-env(){      elocal- ; }
cfg4-usage(){ cat << EOU

Comparisons against Geant4
===========================

Objectives
------------

* Construct Geant4 test geometries and light sources from the same commandline
  arguments as ggv invokations like ggv-rainbow, ggv-prism.

* Add requisite step action(?) to record photon steps in the same format as
  optixrap- using NPY 


1st approach : try to follow Chroma g4py use of Geant4 
---------------------------------------------------------

* /usr/local/env/chroma_env/src/chroma/chroma/generator
* ~/env/chroma/chroma_geant4_integration.rst


2nd approach : C++ following Geant4 examples 
----------------------------------------------

* reuse ggeo- machinery as much as possible






EOU
}
cfg4-dir(){ echo $(env-home)/graphics/ggeoview/cfg4 ; }
cfg4-cd(){  cd $(cfg4-dir); }

cfg4-i(){
   cfg4-cd
   i
}

