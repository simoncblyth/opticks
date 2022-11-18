optix_gcc_cuda_cpp_version_sets
=================================



The recommended version sets to use are driven 
by what the OptiX version release notes. 
They specify the development CUDA version used to build 
each OptiX version and the minimum NVIDIA driver version.:: 

    OptiX          *7.0.0*          7.5.0             7.6.0 
    NVIDIA Driver   435.21          515+              520+ 
    CUDA            10.1            11.7              11.8

Beyond that the CUDA version then constrains the versions 
of gcc and c++ dialect that can be used::

    CUDA            10.1            11.7              11.8
    nvcc c++        c++03,11,14     c++03,11,14,17    ? 
    gcc             8.3.0           11.2              ?

As Opticks compiles against both OptiX and CUDA you
will likely get issues if you don’t stick to the recommended
version sets. 

The Opticks "reference" OptiX version is still 7.0.0 
so I recommend you use that set of versions.

There has been some work, by Hans Wenzel, to get Opticks 
to compile with the OptiX 7.5 version set, but that has
not been tested. 

Please report any issues to the mailing list.
 




 

    


