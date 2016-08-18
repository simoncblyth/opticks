Linux Interop Buffer Overwrite
================================

Following changes for OptiX 400 Assumed Similar Issue Still Manifest
------------------------------------------------------------------------

Report from Tao::

    ihep@linux-h5h2:~/simon-dev-env/env-dev-2016aug1> op-taov2.sh --jpmt -c --tag 2
    300 -rwxr-xr-x 1 ihep users 304480 8月  17 18:26 /home/ihep/simon-dev-env/env-dev-2016aug1/local/opticks/lib/GGeoViewTest
    proceeding : /home/ihep/simon-dev-env/env-dev-2016aug1/local/opticks/lib/GGeoViewTest --size 1920,1080,1 --jpmt -c --tag 2
    2016-08-17 21:02:39.313 INFO  [31001] [Timer::operator@38] Opticks:: START
    ...
    ...
    2016-08-17 21:02:52.836 INFO  [31001] [OPropagator::initEvent@276] OPropagator::initEvent DONE
    2016-08-17 21:02:52.836 INFO  [31001] [OEngineImp::preparePropagator@208] OEngineImp::preparePropagator DONE 
    2016-08-17 21:02:52.836 INFO  [31001] [OpEngine::preparePropagator@94] OpEngine::preparePropagator DONE 
    2016-08-17 21:02:52.836 INFO  [31001] [OpSeeder::seedPhotonsFromGensteps@65] OpSeeder::seedPhotonsFromGensteps
    2016-08-17 21:02:52.836 INFO  [31001] [OpSeeder::seedPhotonsFromGenstepsViaOpenGL@79] OpSeeder::seedPhotonsFromGenstepsViaOpenGL
    CResource::mapGLToCUDA buffer_id 43 imp.bufsize 368640 sizeof(T) 4 size 92160 
    CResource::mapGLToCUDA buffer_id 45 imp.bufsize 62800512 sizeof(T) 4 size 15700128 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs : dev_ptr 0x203891000 size 23040 num_bytes 368640 hexdump 0 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox : dev_ptr 0x203914c00 size 3925032 num_bytes 62800512 hexdump 0 
    2016-08-17 21:02:52.841 INFO  [31001] [OpSeeder::seedPhotonsFromGenstepsImp@146] OpSeeder::seedPhotonsFromGenstepsImp gensteps 3840,6,4 num_genstep_values 92160
    TBufPair<T>::seedDestination (CBufSlice)src : dev_ptr 0x203891000 size 23040 num_bytes 368640 stride 24 begin 3 end 92160 hexdump 0 
    TBufPair<T>::seedDestination (CBufSlice)dst : dev_ptr 0x203914c00 size 3925032 num_bytes 62800512 stride 16 begin 0 end 15700128 hexdump 0 
    iexpand  counts_size 3840 output_size 981258
    2016-08-17 21:02:52.891 INFO  [31001] [OpZeroer::zeroRecords@61] OpZeroer::zeroRecords
    CResource::mapGLToCUDA buffer_id 46 imp.bufsize 157001280 sizeof(T) 2 size 78500640 
    OpZeroer::zeroRecordsViaOpenGL(CBufSpec)s_rec : dev_ptr 0x203899800 size 78500640 num_bytes 157001280 hexdump 0 
    2016-08-17 21:02:52.900 INFO  [31001] [OEngineImp::propagate@214] OEngineImp::propagate
    2016-08-17 21:02:52.900 INFO  [31001] [OPropagator::prelaunch@326] OPropagator::prelaunch count 0 size(981258,1)
    2016-08-17 21:02:52.900 INFO  [31001] [OContext::close@194] OContext::close numEntryPoint 2
    2016-08-17 21:02:53.090 INFO  [31001] [OConfig::dump@158] OContext::close m_raygen_index 2 m_exception_index 2
    OProg R 0 pinhole_camera.cu.ptx pinhole_camera 
    OProg E 0 pinhole_camera.cu.ptx exception 
    OProg M 1 constantbg.cu.ptx miss 
    OProg R 1 generate.cu.ptx generate 
    OProg E 1 generate.cu.ptx exception 
    2016-08-17 21:02:53.090 INFO  [31001] [OConfig::createProgram@40] OConfig::createProgram path /home/ihep/simon-dev-env/env-dev-2016aug1/local/opticks/installcache/PTX/OptiXRap_generated_pinhole_camera.cu.ptx
    2016-08-17 21:02:54.230 INFO  [31001] [OConfig::createProgram@40] OConfig::createProgram path /home/ihep/simon-dev-env/env-dev-2016aug1/local/opticks/installcache/PTX/OptiXRap_generated_pinhole_camera.cu.ptx
    2016-08-17 21:02:54.232 INFO  [31001] [OConfig::createProgram@40] OConfig::createProgram path /home/ihep/simon-dev-env/env-dev-2016aug1/local/opticks/installcache/PTX/OptiXRap_generated_constantbg.cu.ptx
    2016-08-17 21:02:54.236 INFO  [31001] [OConfig::createProgram@40] OConfig::createProgram path /home/ihep/simon-dev-env/env-dev-2016aug1/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
    2016-08-17 21:02:54.668 INFO  [31001] [OConfig::createProgram@40] OConfig::createProgram path /home/ihep/simon-dev-env/env-dev-2016aug1/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
    2016-08-17 21:02:54.689 INFO  [31001] [OContext::launch@220] OContext::launch entry 1 width 981258 height 1
    2016-08-17 21:05:59.941 INFO  [31001] [Timer::operator@38] OEngineImp:: prelaunch
    2016-08-17 21:05:59.941 INFO  [31001] [OContext::launch@220] OContext::launch entry 1 width 981258 height 1
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: driver().cuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address, file:/root/sw/wsapps/raytracing/rtsdk/rel4.0/src/CUDA/Memory.cpp, line: 134)
    /home/ihep/simon-dev-env/env-dev-2016aug1/opticks/bin/op-taov2.sh: 行 567: 31001 已放弃               /home/ihep/simon-dev-env/env-dev-2016aug1/local/opticks/lib/GGeoViewTest --size 1920,1080,1 --jpmt -c --tag 2
    /home/ihep/simon-dev-env/env-dev-2016aug1/opticks/bin/op-taov2.sh RC 134




Weihai Investigations
-----------------------

From compute mode run::

    2016-07-21 11:29:22.326 INFO  [9380] [OpEngine::preparePropagator@89] OpEngine::preparePropagator DONE 
    2016-07-21 11:29:22.326 INFO  [9380] [OpSeeder::seedPhotonsFromGensteps@65] OpSeeder::seedPhotonsFromGensteps
    OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)genstep name genstep size 6 multiplicity 4 sizeofatom 4 NumAtoms 24 NumBytes 96 
    OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_gs : dev_ptr 0xb07200000 size 6 num_bytes 96 
    OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)photon  name photon size 400000 multiplicity 4 sizeofatom 4 NumAtoms 1600000 NumBytes 6400000 
    OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_ox : dev_ptr 0xb07300000 size 400000 num_bytes 6400000 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs : dev_ptr 0xb07200000 size 6 num_bytes 96 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox : dev_ptr 0xb07300000 size 400000 num_bytes 6400000 
    2016-07-21 11:29:22.328 INFO  [9380] [OpSeeder::seedPhotonsFromGenstepsImp@141] OpSeeder::seedPhotonsFromGenstepsImp gensteps 1,6,4 num_genstep_values 24
    TBufPair<T>::seedDestination (CBufSlice)src : dev_ptr 0xb07200000 size 6 num_bytes 96 stride 24 begin 3 end 24 
    TBufPair<T>::seedDestination (CBufSlice)dst : dev_ptr 0xb07300000 size 400000 num_bytes 6400000 stride 16 begin 0 end 1600000 
    iexpand  counts_size 1 output_size 100000
    2016-07-21 11:29:22.332 INFO  [9380] [OpZeroer::zeroRecords@61] OpZeroer::zeroRecords


From interop mode run (OpSeeder buffer sizes are x4 ???)::

    2016-07-21 10:00:50.232 INFO  [881] [OpSeeder::seedPhotonsFromGenstepsViaOpenGL@79] OpSeeder::seedPhotonsFromGenstepsViaOpenGL
    CResource::mapGLToCUDA buffer_id 16 imp.bufsize 96      sizeof(T) 4 size 24 
    CResource::mapGLToCUDA buffer_id 18 imp.bufsize 6400000 sizeof(T) 4 size 1600000 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs : dev_ptr 0x20491ae00 size 24 num_bytes 96 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox : dev_ptr 0x20492f800 size 1600000 num_bytes 6400000 
    2016-07-21 10:00:50.239 INFO  [881] [OpSeeder::seedPhotonsFromGenstepsImp@134] OpSeeder::seedPhotonsFromGenstepsImp gensteps 1,6,4 num_genstep_values 24
    TBufPair<T>::seedDestination (CBufSlice)src : dev_ptr 0x20491ae00 size 24 num_bytes 96 stride 24 begin 3 end 24 
    TBufPair<T>::seedDestination (CBufSlice)dst : dev_ptr 0x20492f800 size 1600000 num_bytes 6400000 stride 16 begin 0 end 1600000 
    iexpand  counts_size 1 output_size 100000
    2016-07-21 10:00:50.263 INFO  [881] [OpZeroer::zeroRecords@61] OpZeroer::zeroRecords

Source of the unexpected x4 bufsize is CResource::mapGLToCUDA::

     53    void* mapGLToCUDA()
     54    {
     55        checkCudaErrors( cudaGraphicsMapResources(1, &resource, stream) );
     56        checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void ##)&dev_ptr, &bufsize, resource) );
     57        //printf("Resource::mapGLToCUDA bufsize %lu dev_ptr %p \n", bufsize, dev_ptr );
     58        return dev_ptr ;
     59    }


::

    In [3]: t = np.load("torchdbg.npy")

    In [9]: np.set_printoptions(suppress=True)

    In [10]: t
    Out[10]: 
    array([[[      0.      ,       0.      ,       0.      ,       0.      ],
            [ -18079.453125, -799699.4375  ,   -6605.      ,       0.1     ],
            [      0.      ,       0.      ,       1.      ,       1.      ],
            [      0.      ,       0.      ,       0.      ,     380.      ],
            [      0.      ,       1.      ,       0.      ,       1.      ],
            [      0.      ,       0.      ,       0.      ,       0.      ]]], dtype=float32)

    In [11]: t.view(np.int32)
    Out[11]: 
    array([[[      4096,          0,         95,     100000],
            [-963821848, -918340297, -976328704, 1036831949],
            [         0,          0, 1065353216, 1065353216],
            [         0,          0,          0, 1136525312],
            [         0, 1065353216,          0, 1065353216],
            [         0,          0,          0,          1]]], dtype=int32)

    In [5]: t.shape
    Out[5]: (1, 6, 4)

    In [6]: 6*4
    Out[6]: 24

    In [7]: 6*4*4
    Out[7]: 96      ## 96 bytes is correct








