CUDA OSX Driver
===============

* https://devtalk.nvidia.com/default/topic/528725/forcing-retina-macbook-pro-to-use-intel-graphics-for-desktop-freeing-memory-on-cuda-device-/


January 2015
--------------

Considering upgrading from Mavericks to Yosemite, any CUDA implications ?


OSX Syspref says::

   CUDA 6.5.36 Driver update is available

   CUDA Driver Version : 5.5.47

   GPU Driver Version : 8.26.26 310.40.45f01


Release notes:

* http://www.nvidia.com/object/macosx-cuda-6.5.25-driver.html

* http://www.nvidia.com/object/macosx-cuda-6.5.36-driver.html 2015.01.14 52.8 MB

  Supports all NVIDIA products available on Mac HW.
  Note: this driver does not support GeForce GTX980 and GTX970. 
  Please download the equivalent CUDA driver 6.5.37 which supports 
  GeForce GTX980 and GTX970.

* http://www.nvidia.com/object/macosx-cuda-6.5.37-driver.html 2015.01.14  52.3 MB

  Supports all NVIDIA products available on Mac HW.
  Note: this driver does not support the older generation GPUs with SM1.x.
  Please download the equivalent CUDA driver 6.5.36 which supports SM1.x.


Some chatter regards problems, but suspect issue with addon hardware on Mac Pro

* :google:`yosemite cuda`

* https://forums.geforce.com/default/topic/787648/geforce-drivers/osx-10-10-yosemite-and-cuda-when-will-it-work-/

* http://www.xlr8yourmac.com/index.html#NvidiaF02DriversOSX10.10.0



March 12, 2014
---------------

Updated (using `System Preferences > CUDA` ) to the latest available
in hope of avoiding frequent GPU kernel panics in response to 
running Chroma ray tracing camera.

From::

    CUDA Driver Version: 5.5.28
    GPU Driver Version: 8.24.9 310.40.25f01

To::

    CUDA Driver Version: 5.5.47
    GPU Driver Version: 8.24.9 310.40.25f01


5.5.47
~~~~~~~

* :google:`cuda mac driver 5.5.47`

* http://www.nvidia.com/object/macosx-cuda-5.5.47-driver.html

* https://developer.nvidia.com/sites/default/files/akamai/cuda/files/CUDADownloads/CUDA_Toolkit_Release_Notes_55RC.pdf


Hardware
---------

MacBook Pro, Retina, Late 2013, NVIDIA GeForce GT 750M 2048 MB

PyCUDA Debugging
------------------

* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions#Is_it_possible_to_use_cuda-gdb_with_PyCUDA.3F
* hmm, no gdb does that still work on OSX 10.9.2
* also does it work with virtual python ?

::

    cuda-gdb --args python -m pycuda.debug


::

    cuda-gdb --args python -m pycuda.debug simplecamera.py -s3199 -d3 -f10 --eye=0,1,0 --lookat=10,0,10  -i




System Preferences
-------------------

`System Preferences > Energy Saver` deselect **Automatic graphics switching** when on Power Adapter

  * this means the discrete GPU is always used rather than the integrated one
  * perhaps the switch contributes to problems 



Simultaneous Integrated and Discrete GPU Operation ?
------------------------------------------------------

Is it possible to configure:

* integrated Intel Iris GPU for OpenGL apps like pygame and OSX Desktop 
* discrete NVIDIA GPU for CUDA compute only 

Integrated
~~~~~~~~~~~

Technical Q&A QA1734: Allowing OpenGL applications to utilize the integrated GPU

* https://developer.apple.com/library/mac/qa/qa1734/_index.html


