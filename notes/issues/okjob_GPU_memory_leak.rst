okjob_GPU_memory_leak
=======================


Overview
----------

* okjob.sh gun:1 is leaking steadily at 0.003 [GB/s] measured with smonitor.sh 
* cxs_min.sh torch running does not leak at a measureable level 


Most likely culprits, as more dynamic allocation handling are:

1. hits
2. gensteps 


cxs_min.sh : NOT LEAKING 
---------------------------

Workstation::

    ~/o/sysrap/smonitor.sh build
    ~/o/sysrap/smonitor.sh run

    TEST=large_scan ~/o/cxs_min.sh 

    CTRL-C the smonitor


::

    .
     [167.325  12.735]
     [168.327  12.735]
     [169.328  12.735]
     [170.332  12.735]
     [171.334  12.735]
     [172.336  12.735]
     [173.338  12.735]]
    dmem      0.002  (usedGpuMemory_GB[sel][-1]-usedGpuMemory_GB[sel][0]) 
    dt      153.299  (t[sel][-1]-t[sel][0]) 
    dmem/dt       0.000  
    smonitor.sh device 0 total_GB 25.8 pid 96770 
    line fit:  slope      0.001 [GB/s] intercept     12.702 



QEvent_Lifecycle_Test.sh : NOT LEAKING
------------------------------------------

::

    ~/o/qudarap/tests/QEvent_Lifecycle_Test.sh 


implemented smonitor monitoring
-----------------------------------

* tried changing to event mode Nothing : but thats too bit a change for comparable numbers 

::

    np.c_[t[sel], usedGpuMemory_GB[sel]]
    [[128.246   1.345]
     [129.247   1.35 ]
     [130.249   1.354]
     [131.25    1.356]
     [132.251   1.36 ]
     [133.252   1.363]
     [134.254   1.367]
     [135.255   1.372]
     [136.256   1.374]
     [137.259   1.376]
     [138.26    1.381]
     [139.264   1.385]
     [140.266   1.387]
     [141.268   1.391]
     [142.269   1.396]
     [143.271   1.399]
     [144.274   1.403]
     [145.277   1.405]
     [146.279   1.409]
     [147.28    1.413]
     [148.281   1.417]
     [149.285   1.421]
     [150.287   1.423]
     [151.288   1.427]
     [152.292   1.429]
     [153.293   1.434]
     [154.295   1.439]
     [155.298   1.441]
     [156.3     1.445]
     [157.302   1.447]
     [158.304   1.452]
     [159.305   1.454]
     [160.308   1.459]
     [161.31    1.463]
     [162.313   1.465]
     [163.315   1.47 ]
     [164.316   1.472]
     [165.319   1.476]
     [166.322   1.478]
     [167.324   1.481]
     [168.326   1.481]
     [169.328   1.483]
     [170.329   1.483]]
    dmem      0.137  (usedGpuMemory_GB[sel][-1]-usedGpuMemory_GB[sel][0]) 
    dt       42.083  (t[sel][-1]-t[sel][0]) 
    dmem/dt       0.003  
    smonitor.sh device 0 total_GB 25.8 pid 280674 
    line fit:  slope      0.003 [GB/s] intercept      0.907 



nvidia-smi monitoring
------------------------


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



pynvml
----------

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



