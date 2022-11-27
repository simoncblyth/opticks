optix_gcc_cuda_cpp_version_sets
=================================


I do not recommend the use of OptiX < 7 currently, 
as all Opticks development effort is currently 
on using the all new NVIDIA OptiX 7 API.     
Note that the NVIDIA OptiX 6->7 transition is like an 
entirely different project, practically all Opticks
code needs change to accomodate this API transition.

The Opticks "reference" NVIDIA OptiX version is currently 7.0.0
so I recommend you use the first column of versions in the below. 

The recommended version sets to use are driven by the OptiX version release notes.
They specify the development CUDA version used to build each OptiX version
and the minimum NVIDIA driver version.::

    OptiX          *7.0.0*          7.5.0             7.6.0 
    NVIDIA Driver   435.21+         515+              520+ 
    CUDA            10.1            11.7              11.8

Beyond that the CUDA version then constrains the versions 
of gcc and c++ dialect that can be used::

    CUDA            10.1            11.7              11.8
    nvcc c++        c++03,11,14     c++03,11,14,17    ?   
    gcc             8.3.0           11.2              ?   

As Opticks compiles against both OptiX and CUDA you 
will likely get issues if you don’t stick to the recommended
version sets. 

There has been some work, by Hans Wenzel, to get Opticks
to compile with the OptiX 7.5 version set, but that has
not been tested.

Please report any issues to the mailing list.
Find instructions for joining the mailing list
and links to publications, presentations and documentation from any of:

https://simoncblyth.bitbucket.io
https://simoncblyth.github.io
https://juno.ihep.ac.cn/~blyth/




 




 

    


