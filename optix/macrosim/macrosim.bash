# === func-gen- : optix/macrosim/macrosim fgp optix/macrosim/macrosim.bash fgn macrosim fgh optix/macrosim
macrosim-src(){      echo optix/macrosim/macrosim.bash ; }
macrosim-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(macrosim-src)} ; }
macrosim-vi(){       vi $(macrosim-source) ; }
macrosim-env(){      elocal- ; }
macrosim-usage(){ cat << EOU

MacroSim
=========

MacroSim is ITOs (Institut fur Technische Optik) 
open source, GPU accelerated ray tracing engine. 
Originally developed for fast non-sequential 
stray light analysis of a spectrometer system

* http://www.uni-stuttgart.de/ito/software/macrosim/index.en.html


Large Project Using OptiX
----------------------------

Iteresting to see the organization approach 
of a large project using CUDA and OptiX.  Command line 
and GUI functionality are strictly separated, and
claims to be able to drop in different OptiX versions.

::

    delta:src blyth$ grep RT_PROGRAM *.cu | wc -l
         268




EOU
}
macrosim-dir(){ echo $(local-base)/env/optix/macrosim ; }
macrosim-cd(){  cd $(macrosim-dir); }
macrosim-mate(){ mate $(macrosim-dir) ; }
macrosim-get(){
   local dir=$(dirname $(macrosim-dir)) &&  mkdir -p $dir && cd $dir

   git clone git@bitbucket.org:itom/macrosim.git

}
