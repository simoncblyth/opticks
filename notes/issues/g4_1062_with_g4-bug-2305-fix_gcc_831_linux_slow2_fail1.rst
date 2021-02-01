g4_1062_with_g4-bug-2305-fix_gcc_831_linux_slow2_fail1
=========================================================

With Geant4 1062 setup a newer than standard gcc for CentOS7 with 
devtoolset-8 and CUDA 10.1 giving gcc 8.3.1.  Initial CUDA nvcc problems
were fixed by adding "--std=c++11" to the nvcc flags, in cmake/Modules/OpticksCUDAFlags.cmake.

gcc::

    [simon@localhost ~]$ gcc --version
    gcc (GCC) 8.3.1 20190311 (Red Hat 8.3.1-3)
    Copyright (C) 2018 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

After hitting the informative ABORT::

    geocache-create
    ...
    X4PhysicalVolume::convertMaterials@263:  num_materials 36 num_material_with_efficiency 1
    GMaterialLib::dumpSensitiveMaterials@1230: X4PhysicalVolume::convertMaterials num_sensitive_materials 1
     0 :                       Bialkali
    GSurfaceLib::createStandardSurface@599: ABORT : non-sensor surfaces must have a reflectivity
    GSurfaceLib::createStandardSurface@600: This ABORT may be caused by Geant4 bug 2305 https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2305 
    GSurfaceLib::createStandardSurface@601: which is present inGeant4  releases 1060,1061,1062,1063,1070 
    GSurfaceLib::createStandardSurface@602: See the bash function g4-;g4-bug-2305-fix to change Geant4 or use a different Geant4 release
    OKX4Test: /home/simon/opticks/ggeo/GSurfaceLib.cc:604: GPropertyMap<float>* GSurfaceLib::createStandardSurface(GPropertyMap<float>*): Assertion `_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "' failed.
    /home/simon/local/opticks/bin/o.sh: line 362: 249609 Aborted                 (core dumped) /home/simon/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /home/simon/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_e


Applied the g4-bug-2305-fix with::

    g4-
    g4-bug-2305-fix
    g4-build


Subseqently geocache-create completes. And opticks-t gives::

    SLOW: tests taking longer that 15 seconds
      24 /25  Test #24 : ExtG4Test.G4GDMLReadSolids_1062_mapOfMatPropVects_bug Passed                         19.12  
      8  /36  Test #8  : CFG4Test.CG4Test                              Passed                         18.08  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         22.15  


    FAILS:  1   / 442   :  Tue Feb  2 01:16:55 2021   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.56   
    [simon@localhost ~]$ 



IntegrationTests.tboolean.box fail ? FIXED
--------------------------------------------

Curious fail, /home/simon/local/opticks/build/integration/ctest.log::

     34 bash: o.sh: command not found
     35 tboolean-- RC 127

Huh PATH missing from ctest environment ?

* **Fixed by following the setup guidance in ~/.opticks_config** 

  * **opticks-setup** was omitted

* https://simoncblyth.bitbucket.io/opticks/docs/install.html#bash-shell-setup-with-opticks-config 

::

    [simon@localhost integration]$ which tboolean.sh 
    ~/local/opticks/bin/tboolean.sh

    [simon@localhost integration]$ which o.sh 
    ~/local/opticks/bin/o.sh



Slow ExtG4Test.G4GDMLReadSolids_1062_mapOfMatPropVects_bug ? FIXED by making xsd schema validation optional
-----------------------------------------------------------------------------------------------------------

Mystifying how ExtG4Test.G4GDMLReadSolids_1062_mapOfMatPropVects_bug can take 19s ?

Running under gdb, note that seems to take a long while to cleanup threads: possibly 
Geant4 1062 has switched on multithreading by default ? Causing a thread related delay
Nope its due to validation loading schema from network.
The setup for GDML reading appears very slow (when logged in as S P@simon, but not so bad from O P@blyth) ?::

    [simon@localhost ~]$ G4GDMLReadSolids_1062_mapOfMatPropVects_bug
    writing gdml to /tmp/mapOfMatPropVects_BUG.gdml
    parsing gdml from /tmp/mapOfMatPropVects_BUG.gdml
    G4VERSION_NUMBER 1062
    G4GDML: Reading '/tmp/mapOfMatPropVects_BUG.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/mapOfMatPropVects_BUG.gdml' done!
    Stripping off GDML names of materials, solids and volumes ...
     skinname skin0 pname EFFICIENCY :  mn     0.0000 mx     0.0000
     skinname skin1 pname EFFICIENCY :  mn     1.0000 mx     1.0000
    [simon@localhost ~]$ 

Interrupting it and looking at backtrace suggests the 
pause is network related delay to access the xsd schema::

    G4GDML: Reading '/tmp/mapOfMatPropVects_BUG.gdml'...
    [New Thread 0x7fffe4c36700 (LWP 404493)]
    ^C
    Program received signal SIGINT, Interrupt.
    [Switching to Thread 0x7fffe4c36700 (LWP 404493)]
    0x00007ffff7def57a in dl_open_worker () from /lib64/ld-linux-x86-64.so.2
    Missing separate debuginfos, use: debuginfo-install cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-57.el7.x86_64 libgcc-4.8.5-39.el7.x86_64 libidn-1.28-4.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libstdc++-4.8.5-39.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-21.el7_6.x86_64 openssl-libs-1.0.2k-19.el7.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff7def57a in dl_open_worker () from /lib64/ld-linux-x86-64.so.2
    #1  0x00007ffff7dea7c4 in _dl_catch_error () from /lib64/ld-linux-x86-64.so.2
    #2  0x00007ffff7deeb7b in _dl_open () from /lib64/ld-linux-x86-64.so.2
    #3  0x00007fffe8fa4722 in do_dlopen () from /lib64/libc.so.6
    #4  0x00007ffff7dea7c4 in _dl_catch_error () from /lib64/ld-linux-x86-64.so.2
    #5  0x00007fffe8fa47e2 in __libc_dlopen_mode () from /lib64/libc.so.6
    #6  0x00007fffe8f797b8 in __nss_lookup_function () from /lib64/libc.so.6
    #7  0x00007fffe8f4d11d in gaih_inet.constprop.8 () from /lib64/libc.so.6
    #8  0x00007fffe8f4e564 in getaddrinfo () from /lib64/libc.so.6
    #9  0x00007fffe786e367 in Curl_getaddrinfo_ex () from /lib64/libcurl.so.4
    #10 0x00007fffe7878414 in getaddrinfo_thread () from /lib64/libcurl.so.4
    #11 0x00007fffe7875ddb in curl_thread_create_thunk () from /lib64/libcurl.so.4
    #12 0x00007fffea580ea5 in start_thread () from /lib64/libpthread.so.0
    #13 0x00007fffe8f658dd in clone () from /lib64/libc.so.6
    (gdb) 



Following above fixes, down to zero FAILs
------------------------------------------

::

    SLOW: tests taking longer that 15 seconds
      8  /36  Test #8  : CFG4Test.CG4Test                              Passed                         17.85  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         22.20  


    FAILS:  0   / 442   :  Tue Feb  2 03:12:43 2021   
    [simon@localhost ~]$ 

