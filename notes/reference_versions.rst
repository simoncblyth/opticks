reference_versions
===================

As Opticks builds against both NVIDIA OptiX and NVIDIA CUDA the requirements 
on choice of versions are more stringent than with simple CUDA usage.   
The current reference NVIDIA OptiX version used by Opticks is now 7.5.0 
which was built with CUDA 11.7 and requires NVIDIA Driver of at least 515.43

If your GPU is unable to work with Driver version 515.43 then I recommend
usage of the old reference set. If you are brave and want to use a version 
set somewhere inbetween those reference points then please report issues (or success)
on the mailing list, including all the version numbers and copy/pasting the error.  


+-----------------+----------------+-------------------+
|                 |  Old Reference | Current Reference |
+=================+================+===================+
| NVIDIA OptiX    |  7.0.0         |    7.5.0          | 
+-----------------+----------------+-------------------+
| NVIDIA Driver   |  435.21        |    515.43.04      |
+-----------------+----------------+-------------------+
| NVIDIA CUDA     |  10.1          |    11.7           |
+-----------------+----------------+-------------------+
| gcc             |  8.3.0         |    11.2.0         |
+-----------------+----------------+-------------------+


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
managers have a tendency to update inappropriately. 
As Opticks builds against both CUDA and OptiX and the OptiX implementation
is provided within the driver the version matching is more  



