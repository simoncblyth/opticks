# === func-gen- : numerics/thrust/thrusthello fgp numerics/thrust/thrusthello.bash fgn thrust fgh numerics/thrust
thrusthello-src(){      echo numerics/thrust/thrusthello.bash ; }
thrusthello-source(){   echo ${BASH_SOURCE:-$(env-home)/$(thrusthello-src)} ; }
thrusthello-vi(){       vi $(thrusthello-source) ; }
thrusthello-usage(){ cat << EOU

Thrust
=======



Refs
-----

* https://developer.nvidia.com/Thrust
* https://developer.nvidia.com/gpu-accelerated-libraries

* http://astronomy.swin.edu.au/supercomputing/thrust.pdf



Thrust cmake
-------------

* http://stackoverflow.com/questions/28968277/compilation-error-using-findcuda-cmake-and-thrust-with-thrust-device-system-omp
* http://stackoverflow.com/questions/13073717/building-cuda-object-files-using-cmake

Thrust OptiX interop
--------------------

* https://devtalk.nvidia.com/search/more/sitecommentsearch/optix%20thrust/

* https://github.com/thrust/thrust/issues/204

  Using thrust with some OptiX types

* https://devtalk.nvidia.com/default/topic/574078/optix/compiler-errors-when-including-both-optix-and-thrust/


Thrust CUDA interop
--------------------

* https://github.com/thrust/thrust/blob/master/examples/cuda/wrap_pointer.cu

Frequency Indexing, Histogramming
-----------------------------------

* http://stackoverflow.com/questions/8792926/finding-the-number-of-occurrences-of-keys-and-the-positions-of-first-occurrences
* https://code.google.com/p/thrust/source/browse/examples/histogram.cu

 



EOU
}

thrusthello-name(){ echo hello ; }

thrusthello-sdir(){ echo $(env-home)/numerics/thrust/$(thrusthello-name)  ; }
thrusthello-bdir(){ echo $(local-base)/env/numerics/thrust/$(thrusthello-name).build  ; }
thrusthello-idir(){ echo $(local-base)/env/numerics/thrust/$(thrusthello-name)  ; }

thrusthello-scd(){  cd $(thrusthello-sdir)/$1; }
thrusthello-bcd(){  cd $(thrusthello-bdir)/$1; }
thrusthello-icd(){  cd $(thrusthello-idir)/$1; }

thrusthello-cd(){   cd $(thrusthello-sdir)/$1; }

thrusthello-env(){      elocal- ; cuda- ; }
thrusthello-samples-dir(){
    echo $(cuda-dir)/samples
}
thrust-pdf(){
    open $(thrust-samples-dir)/doc/Thrust_Quick_Start_Guide.pdf
}

thrust-html(){ open $(cuda-dir)/doc/html/thrust/index.html ; }


thrusthello-nvcc-flags(){ echo "" ; }


thrusthello-wipe(){
   local bdir=$(thrusthello-bdir)
   rm -rf $bdir
}


thrusthello-cmake(){
   local iwd=$PWD

   local bdir=$(thrusthello-bdir)
   mkdir -p $bdir

   thrusthello-bcd
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(thrusthello-idir) \
       -DCUDA_NVCC_FLAGS="$(thrusthello-nvcc-flags)" \
       $(thrusthello-sdir)

   cd $iwd
}

thrusthello-make(){
   local iwd=$PWD

   thrusthello-bcd
   make $*

   cd $iwd
}

thrusthello-install(){
   thrusthello-make install
}

thrusthello--()
{
    thrusthello-wipe
    thrusthello-cmake
    thrusthello-make
    thrusthello-install
}






