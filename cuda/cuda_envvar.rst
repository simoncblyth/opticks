CUDA envvar
=============

* http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars

CUDA_CACHE_DISABLE   
--------------------

::

      0 or 1 (default is 0)   
      Disables caching (when set to 1) or enables caching (when set to 0) for just-in-time-compilation. 
      When disabled, no binary code is added to or retrieved from the cache.

CUDA_DEVICE_WAITS_ON_EXCEPTION  
-------------------------------

Maybe this can be used to avoid panic despite pycuda blase error handling.

* :google:`CUDA_DEVICE_WAITS_ON_EXCEPTION`
* :google:`cuda_device_waits_on_exception freezes GUI`




::

      0 or 1 (default is 0)   
      When set to 1, a CUDA application will halt when a device exception occurs, 
      allowing a debugger to be attached for further debugging.






