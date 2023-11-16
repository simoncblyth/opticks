hmm_running_scontext_test_on_non_gpu_machine_truncates_the_sdevice_record
============================================================================


::

    L7[blyth@lxslc708 .opticks]$ l scontext/
    total 12
    4 -rw-r--r-- 1 blyth dyw  304 Nov 16 10:48 sdevice.bin
    4 drwxr-xr-x 8 blyth dyw 4096 Nov  8 22:06 ..
    4 drwxr-xr-x 2 blyth dyw 4096 Nov  8 21:38 .

    L7[blyth@lxslc708 .opticks]$ which scontext_test
    ~/junotop/ExternalLibs/opticks/head/lib/scontext_test

    L7[blyth@lxslc708 .opticks]$ scontext_test
     CUDA_VISIBLE_DEVICES : [-]
    scontext::desc []
    all_devices
    []
    visible_devices
    []

    L7[blyth@lxslc708 .opticks]$ cd scontext/
    L7[blyth@lxslc708 scontext]$ l
    total 8
    0 -rw-r--r-- 1 blyth dyw    0 Nov 16 15:52 sdevice.bin
    4 drwxr-xr-x 8 blyth dyw 4096 Nov  8 22:06 ..
    4 drwxr-xr-x 2 blyth dyw 4096 Nov  8 21:38 .
    L7[blyth@lxslc708 scontext]$ 



