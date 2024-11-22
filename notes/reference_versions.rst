reference_versions
===================

As Opticks builds against both NVIDIA OptiX and NVIDIA CUDA the requirements 
on choice of versions are more stringent than with simple CUDA usage.   
The current reference NVIDIA OptiX version used by Opticks is now 7.5.0 
which was built with CUDA 11.7 and requires NVIDIA Driver of at least 515.43


Advice on which version set to use
------------------------------------

+-----------------+----------------+-------------------+------------------+
|                 |  Old Reference | Current Reference | Future Reference |
+=================+================+===================+==================+
| NVIDIA OptiX    |  7.0.0         |    7.5.0          |   8.0.0          | 
+-----------------+----------------+-------------------+------------------+
| NVIDIA Driver   |  435.21        |    515.43.04      |    550.76        |
+-----------------+----------------+-------------------+------------------+
| NVIDIA CUDA     |  10.1          |    11.7           |    12.4          |
+-----------------+----------------+-------------------+------------------+
| gcc             |  8.3.0         |    11.2.0         |    11.4.1        |
+-----------------+----------------+-------------------+------------------+
|  OS             |  CentOS 7      |  CentOS 7         | AlmaLinux 9      | 
+-----------------+----------------+-------------------+------------------+


* If you can use an NVIDIA Driver version that supports OptiX 8.0.0 then use the future version set.
* If your NVIDIA Driver cannot yet handle OptiX 8.0.0 yet then use the OptiX 7.5.0 version set.  
* If your NVIDIA Driver cannot yet handle OptiX 7.5.0 yet then use the OptiX 7.0.0 version set.  
* OptiX prior to 7.0.0 was a totally different API that I am no longer supporting. 
  
Using version sets other than reference ones is development activity.
I do not recommend new users to do that unless you have lots of experience with CUDA,
in which case your feedback on testing other version sets is appreciated. Please report
failures or successes to the mailing list including all the version numbers 
and copy/pasting error messages.  



Standard Version Sets Gleaned from NVIDIA OptiX Release Notes
----------------------------------------------------------------


+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Release Date    |   NVIDIA OptiX    |  Notes          |  Driver        |  CUDA   |  gcc    |                                |   
+==================+===================+=================+================+=========+=========+================================+
|  July 2019       |   7.0.0           |  NEW API        | 435.12(435.21) |  10.1   |  8.3.0  | OLD REFERENCE SET              |
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  June 2020       |   7.1.0           | Added Curves    | 450            |  11.0   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Oct 2020        |   7.2.0           | Specialization  | 455            |  11.1   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Apr 2021        |   7.3.0           |                 | 465            |  11.1   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Oct 2021        |   7.4.0           | Catmull-Rom     | 495            |  11.4   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  June 2022       | :b:`7.5.0` [1]    | Debug, Sphere   | 515            |  11.7   |         | CURRENT REFERENCE SET          |
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Oct 2022        |   7.6.0 [1]       |                 | 520            |  11.8   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Mar 2023        |   7.7.0           | More Curves     | 530            |  12.0   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Aug 2023        |   8.0.0           | SER, Perf       | 535            |  12.0   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+


* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
* https://docs.nvidia.com/cuda/archive/11.8.0/
* https://gist.github.com/ax3l/9489132



CUDA Installation via runfile, not via package managers
---------------------------------------------------------

* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

The NVIDIA CUDA runfile based installation now includes the 
appropriate NVIDIA Driver. In order to use versions combinations
that are as standard as possible (ie used by very many others already)
the runfile approach is recommended. While package managers claim to 
be able to install CUDA, I do not trust them. Also package 
managers have a tendency to update inappropriately.  More control is 
needed when building against packages that build against CUDA. 



Driver Requirements for CUDA Releases
---------------------------------------

* https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id5





