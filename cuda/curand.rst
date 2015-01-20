CUDA Random numbers
=====================

* https://developer.nvidia.com/search/google_cse_adv/curand
* https://developer.nvidia.com/curand
* http://docs.nvidia.com/cuda/curand/index.html

CURAND consists of two pieces: a library on the host (CPU) side and a device
(GPU) header file. The host-side library is treated like any other CPU library:
users include the header file, /include/curand.h, to get function declarations
and then link against the library. Random numbers can be generated on the
device or on the host CPU. For device generation, calls to the library happen
on the host, but the actual work of random number generation occurs on the
device. The resulting random numbers are stored in global memory on the device.
Users can then call their own kernels to use the random numbers, or they can
copy the random numbers back to the host for further processing. For host CPU
generation, all of the work is done on the host, and the random numbers are
stored in host memory.

The second piece of CURAND is the device header file, /include/curand_kernel.h.
This file defines device functions for setting up random number generator
states and generating sequences of random numbers. User code may include this
header file, and user-written kernels may then call the device functions
defined in the header file. This allows random numbers to be generated and
immediately consumed by user kernels without requiring the random numbers to be
written to and then read from global memory.

