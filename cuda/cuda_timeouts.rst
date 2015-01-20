CUDA Error Handling including timeouts
========================================


Handling / Reset ?
-------------------

* http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
* http://stackoverflow.com/questions/19632401/how-to-work-around-gpu-watchdog-timer-limitation-on-cuda-code-in-os-x
* http://stackoverflow.com/questions/9602312/gpu-card-resets-after-2-seconds


Compute and Graphics
---------------------

Using GPU for both, forces use of timeouts.

* https://devtalk.nvidia.com/default/topic/483643/cuda-the-launch-timed-out-and-was-terminated/
* https://devtalk.nvidia.com/search/more/sitecommentsearch/Launch%20timeout/


CUDA Driver API Errors
--------------------------


* http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES


`CUDA_ERROR_LAUNCH_FAILED = 700`
    An exception occurred on the device while executing a kernel. Common causes
    include dereferencing an invalid device pointer and accessing out of bounds
    shared memory. The context cannot be used, so it must be destroyed (and a new
    one should be created). All existing device memory allocations from this
    context are invalid and must be reconstructed if the program is to continue
    using CUDA.


`CUDA_ERROR_LAUNCH_TIMEOUT = 702`
    This indicates that the device kernel took too long to execute. This can only
    occur if timeouts are enabled - see the device attribute
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. 
    The context cannot be used (and must be destroyed similar to CUDA_ERROR_LAUNCH_FAILED). All
    existing device memory allocations from this context are invalid and must be
    reconstructed if the program is to continue using CUDA.



`CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17`
    Specifies whether there is a run time limit on kernels


deviceQuery
--------------

::

    delta:w blyth$ cuda-samples-bin-deviceQuery | grep limit 
      Run time limit on kernels:                     Yes

