CUDA Texture Memory 
=====================


* http://cuda-programming.blogspot.tw/2013/02/texture-memory-in-cuda-what-is-texture.html


* https://devtalk.nvidia.com/default/topic/381335/setting-up-for-tex1d-how-to-load-cuda-array-for-tex1d-/



Reference
-----------

* :google:`cuda 5 reference texture`



* http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api



Texture Objects
----------------

* http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/


Kepler GPUs and CUDA 5.0 introduce a new feature called texture objects
(sometimes called bindless textures, since they don’t require manual
binding/unbinding) that greatly improves the usability and programmability of
textures. Texture objects use the new cudaTextureObject_t class API, whereby
textures become first-class C++ objects and can be passed as arguments just as
if they were pointers.  There is no need to know at compile time which textures
will be used at run time, which enables much more dynamic execution and
flexible programming, as shown in the following code.



