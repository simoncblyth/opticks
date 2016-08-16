Linux Interop Buffer Overwrite
================================


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








