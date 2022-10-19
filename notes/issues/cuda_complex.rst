cuda_complex : cuda/std/complex
================================


libcudacxx : nascent "standard" library with cuda/std/complex
----------------------------------------------------------------

Looks like the below are with cudacxx 1.4.0 which is maybe not yet released with CUDA yet::

    <cuda/std/complex>
    <cuda/std/ccomplex>



* https://stackoverflow.com/questions/72876099/cuda-no-operator-for-volatile-cudastdcomplexfloat
* https://nvidia.github.io/libcudacxx/
* https://stackoverflow.com/questions/68075342/how-do-i-properly-include-files-from-the-nvidia-c-standard-library
* https://nvidia.github.io/libcudacxx/standard_api.html
* https://github.com/NVIDIA/libcudacxx
* https://stackoverflow.com/questions/17473826/cuda-how-to-work-with-complex-numbers

In the last few years, NVIDIA has been developing a "standard" library
for CUDA (libcu++) that mimics some aspects of std::. This library includes
complex functionality, here is an example of usage.

Robert Crovella, 2013



* https://github.com/NVIDIA/libcudacxx/blob/main/include/cuda/std/complex
* https://github.com/NVIDIA/libcudacxx/blob/main/include/cuda/std/detail/libcxx/include/complex

  * looks simple enough, it might work with an older CUDA





