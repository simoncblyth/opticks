# === func-gen- : numerics/thrust/thrust fgp numerics/thrust/thrust.bash fgn thrust fgh numerics/thrust
thrust-src(){      echo numerics/thrust/thrust.bash ; }
thrust-source(){   echo ${BASH_SOURCE:-$(env-home)/$(thrust-src)} ; }
thrust-vi(){       vi $(thrust-source) ; }
thrust-usage(){ cat << EOU

Thrust
=======



Refs
-----

* https://developer.nvidia.com/Thrust

* https://code.google.com/p/thrust/wiki/QuickStartGuide

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


thrust-idir(){ echo $(cuda-idir)/thrust ; }
thrust-icd(){  cd $(thrust-idir) ; }
thrust-cd(){   cd $(thrust-idir) ; }

thrust-env(){      
   elocal- ; 
   cuda- ; 
}

thrust-samples-dir(){ echo $(cuda-dir)/samples ; }
thrust-pdf(){  open $(thrust-samples-dir)/doc/Thrust_Quick_Start_Guide.pdf ; }
thrust-html(){ open $(cuda-dir)/doc/html/thrust/index.html ; }





