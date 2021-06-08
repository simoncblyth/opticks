gracious_no_gpu_handling
==========================

::


    epsilon:cfg4 blyth$ CVD=- lldb__ CRandomEngineTest 
    /Applications/Xcode/Xcode_10_1.app/Contents/Developer/usr/bin/lldb -f CRandomEngineTest -o r --
    (lldb) target create "/usr/local/opticks/lib/CRandomEngineTest"
    Current executable set to '/usr/local/opticks/lib/CRandomEngineTest' (x86_64).
    (lldb) r
    2021-06-07 16:15:12.090 INFO  [100489] [main@103] /usr/local/opticks/lib/CRandomEngineTest
    2021-06-07 16:15:12.090 INFO  [100489] [main@109]  pindex0 0 pindex1 1 pstep 1
    2021-06-07 16:15:12.092 ERROR [100489] [Opticks::postconfigureCVD@3000]  --cvd [-] option internally sets CUDA_VISIBLE_DEVICES []
    libc++abi.dylib: terminating with uncaught exception of type thrust::system::system_error: get_max_shared_memory_per_block :failed to cudaGetDevice: no CUDA-capable device is detected

    Process 39272 launched: '/usr/local/opticks/lib/CRandomEngineTest' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff64438b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff64603080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff643941ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff6228ef8f libc++abi.dylib`abort_message + 245
        frame #4: 0x00007fff6228f113 libc++abi.dylib`default_terminate_handler() + 241
        frame #5: 0x00007fff636d0eab libobjc.A.dylib`_objc_terminate() + 105
        frame #6: 0x00007fff622aa7c9 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff622aa26f libc++abi.dylib`__cxa_throw + 121
        frame #8: 0x0000000100826fa6 libThrustRap.dylib`thrust::cuda_cub::core::get_max_shared_memory_per_block() + 326
        frame #9: 0x00000001008275d5 libThrustRap.dylib`void thrust::cuda_cub::parallel_for<thrust::cuda_cub::tag, thrust::cuda_cub::__uninitialized_fill::functor<thrust::device_ptr<double>, double>, unsigned long>(thrust::cuda_cub::execution_policy<thrust::cuda_cub::tag>&, thrust::cuda_cub::__uninitialized_fill::functor<thrust::device_ptr<double>, double>, unsigned long) + 117
        frame #10: 0x00000001008274ec libThrustRap.dylib`thrust::detail::vector_base<double, thrust::device_malloc_allocator<double> >::vector_base(unsigned long) + 108
        frame #11: 0x0000000100824dcf libThrustRap.dylib`TCURANDImp<double>::TCURANDImp(unsigned int, unsigned int, unsigned int) + 175
        frame #12: 0x0000000100823d8b libThrustRap.dylib`TCURAND<double>::TCURAND(this=0x000000010b7225c0, ni=100000, nj=16, nk=16) at TCURAND.cc:30
        frame #13: 0x0000000100823de7 libThrustRap.dylib`TCURAND<double>::TCURAND(this=0x000000010b7225c0, ni=100000, nj=16, nk=16) at TCURAND.cc:31
        frame #14: 0x0000000100138e1e libCFG4.dylib`CRandomEngine::CRandomEngine(this=<unavailable>, manager=<unavailable>) at CRandomEngine.cc:105 [opt]
        frame #15: 0x0000000100004c9d CRandomEngineTest`main [inlined] CRandomEngineTest::CRandomEngineTest(this=<unavailable>, manager=<unavailable>) at CRandomEngineTest.cc:53 [opt]
        frame #16: 0x0000000100004c7f CRandomEngineTest`main [inlined] CRandomEngineTest::CRandomEngineTest(this=<unavailable>, manager=<unavailable>) at CRandomEngineTest.cc:54 [opt]
        frame #17: 0x0000000100004c7f CRandomEngineTest`main(argc=<unavailable>, argv=<unavailable>) at CRandomEngineTest.cc:128 [opt]
        frame #18: 0x00007fff642e8015 libdyld.dylib`start + 1
    (lldb) 



    epsilon:cfg4 blyth$ CVD=- TCURANDImp=INFO lldb__ CRandomEngineTest 




