
optix 400 : seedPhotonsFromGenstepsImp FATAL : mismatch between CPU and GPU photon counts
=============================================================================================

Tao encountered crazy photon counts from the Thrust reduction of the gensteps::


    OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_ox : dev_ptr 0x700ca0000 size 400000 num_bytes 6400000 hexdump 0 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs : dev_ptr 0x700ba0000 size 6 num_bytes 96 hexdump 0 
    OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox : dev_ptr 0x700ca0000 size 400000 num_bytes 6400000 hexdump 0 
    2016-08-16 14:19:34.869 INFO  [3271157] [OpSeeder::seedPhotonsFromGenstepsImp@146] OpSeeder::seedPhotonsFromGenstepsImp gensteps 1,6,4 num_genstep_values 24
    2016-08-16 14:19:34.875 FATAL [3271157] [OpSeeder::seedPhotonsFromGenstepsImp@156] OpSeeder::seedPhotonsFromGenstepsImp num_photons 4294967295 x_num_photons 100000
    Assertion failed: (num_photons == x_num_photons && "FATAL : mismatch between CPU and GPU photon counts from the gensteps"), function seedPhotonsFromGenstepsImp, file /Users/blyth/opticks/opticksop/OpSeeder.cc, line 162.
    Abort trap: 6


Suggests the gensteps failed to be properly uploaded::

    132 void OpSeeder::seedPhotonsFromGenstepsImp(const CBufSpec& s_gs, const CBufSpec& s_ox)
    133 {
    134     s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs");
    135     s_ox.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox");
    136 
    137     TBuf tgs("tgs", s_gs );
    138     TBuf tox("tox", s_ox );
    139    
    140     //tgs.dump<unsigned int>("App::seedPhotonsFromGenstepsImp tgs", 6*4, 3, nv0 ); // stride, begin, end 
    141 
    142     NPY<float>* gensteps =  m_evt->getGenstepData() ;
    143 
    144     unsigned int num_genstep_values = gensteps->getNumValues(0) ;
    145 
    146     LOG(info) << "OpSeeder::seedPhotonsFromGenstepsImp"
    147                << " gensteps " << gensteps->getShapeString()
    148                << " num_genstep_values " << num_genstep_values
    149                ;
    150 
    151     unsigned int num_photons = tgs.reduce<unsigned int>(6*4, 3, num_genstep_values );  // adding photon counts for each genstep 
    152 
    153     unsigned int x_num_photons = m_evt->getNumPhotons() ;
    154 
    155     if(num_photons != x_num_photons)
    156           LOG(fatal)
    157           << "OpSeeder::seedPhotonsFromGenstepsImp"
    158           << " num_photons " << num_photons
    159           << " x_num_photons " << x_num_photons
    160           ;
    161 
    162     assert(num_photons == x_num_photons && "FATAL : mismatch between CPU and GPU photon counts from the gensteps") ;



Taos fix for the issue (in compute mode)

* https://bitbucket.org/simoncblyth/opticks/commits/2fd2a8fb3b2615a85d7bf0126d2ffe999ab7b609
* use RT_BUFFER_COPY_ON_DIRTY for OptiX buffers in compute mode
* use CUDA and not OptiX to do the gensteps upload

::

    +    // memcpy( buffer->map(), npy->getBytes(), numBytes );
    +    // buffer->unmap(); 
    +    void* d_ptr = NULL;
    +    rtBufferGetDevicePointer(buffer->get(), 0, &d_ptr);
    +    cudaMemcpy(d_ptr, npy->getBytes(), numBytes, cudaMemcpyHostToDevice);
    +    buffer->markDirty();

    buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_COPY_ON_DIRTY);







