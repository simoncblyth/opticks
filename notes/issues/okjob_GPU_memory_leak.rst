okjob_GPU_memory_leak
=======================

During 1000 event run monitor with::

    nvidia-smi -lms 500    # every half second 



starts flat at 941Mib::


    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     13888      G   /usr/bin/X                         24MiB |
    |    0   N/A  N/A     15789      G   /usr/bin/gnome-shell              112MiB |
    |    0   N/A  N/A     16775      G   /usr/bin/X                        129MiB |
    |    0   N/A  N/A     23246      C   python                            941MiB |
    |    0   N/A  N/A    352750      G   /usr/bin/gnome-shell               14MiB |
    +-----------------------------------------------------------------------------+

Jumps to 1283MiB::

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     13888      G   /usr/bin/X                         24MiB |
    |    0   N/A  N/A     15789      G   /usr/bin/gnome-shell              112MiB |
    |    0   N/A  N/A     16775      G   /usr/bin/X                        129MiB |
    |    0   N/A  N/A     23246      C   python                           1283MiB |
    |    0   N/A  N/A    352750      G   /usr/bin/gnome-shell               14MiB |
    +-----------------------------------------------------------------------------+

Then proceeds steadily upwards ending after 1000 launches at 1414MiB::

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     13888      G   /usr/bin/X                         24MiB |
    |    0   N/A  N/A     15789      G   /usr/bin/gnome-shell              112MiB |
    |    0   N/A  N/A     16775      G   /usr/bin/X                        129MiB |
    |    0   N/A  N/A     23246      C   python                           1414MiB |
    |    0   N/A  N/A    352750      G   /usr/bin/gnome-shell               15MiB |
    +-----------------------------------------------------------------------------+


* 1414-1283 

::

    In [2]: (1414-1283)/1000.
    Out[2]: 0.131


Leaking about 0.1 MB per launch 



Install pynvml with conda::

    N[blyth@localhost nvml_py]$ ./moni.py 
    devcount:2 
    handle:<pynvml.nvml.LP_struct_c_nvmlDevice_t object at 0x7fc05499d440>
    {'pid': 226283, 'usedGpuMemory': 986710016, 'gpuInstanceId': 4294967295, 'computeInstanceId': 4294967295}
    pid 226283 using 986710016 bytes of memory on device 0.
    handle:<pynvml.nvml.LP_struct_c_nvmlDevice_t object at 0x7fc05499cf80>


::

    N[blyth@localhost nvml_py]$ cat ~/nvml_py/moni.py 
    #!/usr/bin/env python

    import pynvml

    pynvml.nvmlInit()

    devcount = pynvml.nvmlDeviceGetCount()
    print("devcount:%d " % devcount )

    for dev_id in range(devcount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        print("handle:%s" % handle) 

        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):

            print(proc)
            print(
                "pid %d using %d bytes of memory on device %d."
                % (proc.pid, proc.usedGpuMemory, dev_id)
            )



    N[blyth@localhost nvml_py]$ 



