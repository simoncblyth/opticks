Two out of 22 libs CUDARap.so and libThrustRap.so are duplicated into lib and lib64 : why ?
============================================================================================

::

    blyth@localhost boostrap]$ ll /home/blyth/local/opticks/lib64/
    total 106524
    drwxrwxr-x. 12 blyth blyth      152 Apr  1 18:19 ..
    drwxrwxr-x. 31 blyth blyth     4096 Apr  9 15:42 cmake
    -rwxr-xr-x.  1 blyth blyth   662600 Apr  9 19:48 libUseCUDA.so
    -rwxr-xr-x.  1 blyth blyth    32368 Apr  9 21:16 libOKConf.so
    -rwxr-xr-x.  1 blyth blyth  1123400 Apr  9 21:16 libSysRap.so
    -rwxr-xr-x.  1 blyth blyth 11327648 Apr  9 21:25 libBoostRap.so
    -rwxr-xr-x.  1 blyth blyth 25732864 Apr  9 21:26 libNPY.so
    -rwxr-xr-x.  1 blyth blyth  2594560 Apr  9 21:26 libYoctoGLRap.so
    -rwxr-xr-x.  1 blyth blyth  9924632 Apr  9 21:26 libOpticksCore.so
    -rwxr-xr-x.  1 blyth blyth 10237960 Apr  9 21:26 libGGeo.so
    -rwxr-xr-x.  1 blyth blyth  1164464 Apr  9 21:26 libAssimpRap.so
    -rwxr-xr-x.  1 blyth blyth  2734968 Apr  9 21:26 libOpenMeshRap.so
    -rwxr-xr-x.  1 blyth blyth  1363136 Apr  9 21:26 libOpticksGeo.so
    -rwxr-xr-x.  1 blyth blyth  1256976 Apr  9 21:27 libCUDARap.so
    -rwxr-xr-x.  1 blyth blyth  3866064 Apr  9 21:27 libThrustRap.so
    -rwxr-xr-x.  1 blyth blyth  4791744 Apr  9 21:27 libOptiXRap.so
    -rwxr-xr-x.  1 blyth blyth  2507880 Apr  9 21:28 libOKOP.so
    -rwxr-xr-x.  1 blyth blyth  4776888 Apr  9 21:28 libOGLRap.so
    -rwxr-xr-x.  1 blyth blyth   737640 Apr  9 21:28 libOpticksGL.so
    -rwxr-xr-x.  1 blyth blyth   367376 Apr  9 21:28 libOK.so
    -rwxr-xr-x.  1 blyth blyth  4171728 Apr  9 21:28 libExtG4.so
    -rwxr-xr-x.  1 blyth blyth 17926032 Apr  9 21:28 libCFG4.so
    -rwxr-xr-x.  1 blyth blyth   293864 Apr  9 21:28 libOKG4.so
    -rwxr-xr-x.  1 blyth blyth   705928 Apr  9 21:28 libG4OK.so
    drwxrwxr-x.  4 blyth blyth     4096 Apr  9 21:28 .
    drwxrwxr-x.  2 blyth blyth     4096 Apr  9 21:28 pkgconfig
    [blyth@localhost boostrap]$ 
    [blyth@localhost boostrap]$ cd ..
    [blyth@localhost opticks]$ ll /home/blyth/local/opticks/lib/*.so
    -rwxr-xr-x. 1 blyth blyth 1256976 Apr  9 21:27 /home/blyth/local/opticks/lib/libCUDARap.so
    -rwxr-xr-x. 1 blyth blyth 3866064 Apr  9 21:27 /home/blyth/local/opticks/lib/libThrustRap.so


::

    [blyth@localhost opticks]$ diff lib/libCUDARap.so lib64/libCUDARap.so 
    [blyth@localhost opticks]$ diff lib/libThrustRap.so lib64/libThrustRap.so 


