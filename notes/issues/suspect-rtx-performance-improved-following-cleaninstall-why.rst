suspect-rtx-performance-improved-following-cleaninstall-why
================================================================


* :google:`CUDA compile cache`

* https://devblogs.nvidia.com/cuda-pro-tip-understand-fat-binaries-jit-caching/

Ingredients:

1. CUDA cache
2. OptiX cache
3. Opticks saved PTX for one compute capability 70
4. two GPUs with compute capabilites 70 and 75 
5. BUT OptiX parses the PTX and recompiles anyhow 



::

    CUDA_CACHE_DISABLE
          1 disables caching (no binary code is added to or retrieved from the cache).

    CUDA_CACHE_PATH
          directory location of compute cache files ~/.nv/ComputeCache/

    CUDA_CACHE_MAXSIZE 
          specifies the size of the compute cache in bytes; the default size is 256 MiB 
          binary codes whose size exceeds the cache size are not cached; 
          older binary codes are evicted from the cache to make room for newer binary codes if needed.



Hmm the cache is maxed out::

    [blyth@localhost opticks]$ du -hs ~/.nv/ComputeCache/
    256M    /home/blyth/.nv/ComputeCache/





    [blyth@localhost opticks]$ l  ~/.nv/ComputeCache/
    total 8
    -rw-rw-r--.  1 blyth blyth 5648 May 22 17:20 index
    drwx------. 15 blyth blyth  123 May 22 13:16 d
    drwx------. 14 blyth blyth  114 May 22 13:15 e
    drwx------. 16 blyth blyth  132 May 21 21:59 a
    drwx------. 17 blyth blyth  141 May 21 21:31 7
    drwx------. 17 blyth blyth  141 May 21 11:25 8
    drwx------. 16 blyth blyth  132 May 20 17:30 0
    drwx------. 14 blyth blyth  114 May 20 17:25 f
    drwx------. 15 blyth blyth  123 May 19 19:31 b
    drwx------. 16 blyth blyth  132 May 19 19:22 1
    drwx------. 15 blyth blyth  123 May 18 23:06 2
    drwx------. 15 blyth blyth  123 May 18 19:41 9
    drwx------. 17 blyth blyth  141 May 18 19:36 6
    drwx------. 16 blyth blyth  132 May 18 19:30 3
    drwx------. 18 blyth blyth  150 May 18 14:43 4
    drwx------. 17 blyth blyth  141 May  9 15:07 c
    drwx------. 16 blyth blyth  132 Apr 28 11:13 5
    [blyth@localhost opticks]$ 


::

    blyth@localhost opticks]$ ll ~/local/opticks/installcache/PTX/
    total 3040
    drwxrwxr-x. 5 blyth blyth     39 Jul  6  2018 ..
    -rw-r--r--. 1 blyth blyth 117064 Apr 10 17:17 OptiXRap_generated_intersect_analytic_test.cu.ptx
    -rw-r--r--. 1 blyth blyth  38844 Apr 10 21:38 UseOptiXProgram_generated_UseOptiXProgram_minimal.cu.ptx
    -rw-r--r--. 1 blyth blyth  38844 Apr 10 22:16 UseOptiXProgramCPP_generated_UseOptiXProgramCPP_minimal.cu.ptx
    -rw-r--r--. 1 blyth blyth  38844 Apr 10 22:59 UseOptiXProgramPP_generated_UseOptiXProgramPP_minimal.cu.ptx
    -rw-r--r--. 1 blyth blyth  41794 Apr 11 13:26 UseOptiXBufferPP_generated_basicTest.cu.ptx
    -rw-r--r--. 1 blyth blyth  42412 Apr 11 17:07 UseOptiXRapBufferPP_generated_bufferTest.cu.ptx
    -rw-r--r--. 1 blyth blyth  42412 Apr 11 21:08 UseOContextBufferPP_generated_bufferTest.cu.ptx
    -rw-r--r--. 1 blyth blyth  40994 Apr 11 23:13 UseOptiXBufferPP_generated_bufferTest.cu.ptx
    -rw-r--r--. 1 blyth blyth  39323 Apr 11 23:33 UseOptiXBuffer_generated_minimal.cu.ptx
    -rw-r--r--. 1 blyth blyth  39389 Apr 11 23:40 UseOptiXBuffer_generated_UseOptiXBuffer.cu.ptx
    -rw-r--r--. 1 blyth blyth  40404 Apr 12 21:48 OptiXRap_generated_intersect_analytic_test_dummy.cu.ptx
    -rw-r--r--. 1 blyth blyth  41794 Apr 23 09:34 UseOptiXGeometry_generated_basicTest.cu.ptx
    -rw-r--r--. 1 blyth blyth   9304 Apr 23 14:39 UseOptiXGeometryTriangles_generated_box.cu.ptx
    -rw-r--r--. 1 blyth blyth  51416 Apr 23 16:09 UseOptiXGeometryTriangles_generated_UseOptiXGeometryTriangles.cu.ptx
    -rw-r--r--. 1 blyth blyth  38844 Apr 28 14:09 UseOptiXProgram_generated_minimal.cu.ptx
    -rw-r--r--. 1 blyth blyth  41794 Apr 28 14:10 UseOptiXProgramPP_generated_basicTest.cu.ptx
    -rw-r--r--. 1 blyth blyth   9304 May 21 14:07 UseOptiXGeometry_generated_box.cu.ptx
    -rw-r--r--. 1 blyth blyth  48610 May 21 14:07 UseOptiXGeometry_generated_UseOptiXGeometry.cu.ptx
    -rw-r--r--. 1 blyth blyth  48610 May 21 14:33 UseOptiXGeometryStandalone_generated_UseOptiXGeometryStandalone.cu.ptx
    -rw-r--r--. 1 blyth blyth   9304 May 21 14:33 UseOptiXGeometryStandalone_generated_box.cu.ptx
    -rw-r--r--. 1 blyth blyth   6865 May 22 14:25 OptiXRap_generated_visit_instance.cu.ptx
    -rw-r--r--. 1 blyth blyth  12007 May 22 14:25 OptiXRap_generated_TriangleMesh.cu.ptx
    -rw-r--r--. 1 blyth blyth  42656 May 22 14:25 OptiXRap_generated_textureTest.cu.ptx
    -rw-r--r--. 1 blyth blyth  40376 May 22 14:25 OptiXRap_generated_texTest.cu.ptx
    -rw-r--r--. 1 blyth blyth  38913 May 22 14:25 OptiXRap_generated_tex0Test.cu.ptx
    -rw-r--r--. 1 blyth blyth   1987 May 22 14:25 OptiXRap_generated_solve_callable.cu.ptx
    -rw-r--r--. 1 blyth blyth  40727 May 22 14:25 OptiXRap_generated_seedTest.cu.ptx
    -rw-r--r--. 1 blyth blyth  71677 May 22 14:25 OptiXRap_generated_Roots3And4Test.cu.ptx
    -rw-r--r--. 1 blyth blyth  45978 May 22 14:25 OptiXRap_generated_OEventTest.cu.ptx
    -rw-r--r--. 1 blyth blyth  39121 May 22 14:25 OptiXRap_generated_minimalTest.cu.ptx
    -rw-r--r--. 1 blyth blyth   9126 May 22 14:25 OptiXRap_generated_material1_radiance.cu.ptx
    -rw-r--r--. 1 blyth blyth   7147 May 22 14:25 OptiXRap_generated_material1_propagate.cu.ptx


