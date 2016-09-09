
Interop CUDA Memory Error
============================


::
    OKTest 

    ...
    2016-09-08 21:03:56.200 INFO  [3537] [OContext::configureBuffer@432]   sequence          100000,1,2 USER size (ijk)     200000 elementsize 8
    2016-09-08 21:03:56.203 INFO  [3537] [SLog::operator@15] OEvent::OEvent DONE
    2016-09-08 21:03:56.203 INFO  [3537] [OpSeeder::seedPhotonsFromGensteps@61] OpSeeder::seedPhotonsFromGensteps
    2016-09-08 21:03:56.203 INFO  [3537] [OpSeeder::seedPhotonsFromGenstepsViaOpenGL@76] OpSeeder::seedPhotonsFromGenstepsViaOpenGL
    2016-09-08 21:03:56.240 INFO  [3537] [OpSeeder::seedPhotonsFromGenstepsImp@148] OpSeeder::seedPhotonsFromGenstepsImp gensteps 1,6,4 num_genstep_values 24
    iexpand  counts_size 1 output_size 100000
    2016-09-08 21:03:56.260 INFO  [3537] [OpZeroer::zeroRecords@54] OpZeroer::zeroRecords
    OpZeroer::zeroRecordsViaOpenGL(CBufSpec)s_rec : dev_ptr 0x711040000 size 8000000 num_bytes 16000000 hexdump 0 
    2016-09-08 21:03:56.274 INFO  [3537] [OContext::launch@221] OContext::launch entry 0 width 100000 height 1
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Memory allocation failed (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: driver().cuGraphicsMapResources( 1, &m_resource, hStream.get() ) returned (2): Out of memory, file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/CUDA/GraphicsResource.cpp, line: 73)
    Abort trap: 6
    simon:opticks blyth$ 
    simon:opticks blyth$ 

