Attempt to use SolveCubic_callable failing
==============================================

* compilation took a long time, eventually failing in launch


::

    2017-08-11 14:28:22.213 INFO  [3111812] [OEvent::uploadGensteps@242] OEvent::uploadGensteps (INTEROP) SKIP OpenGL BufferId 13
    2017-08-11 14:28:22.213 INFO  [3111812] [OpSeeder::seedComputeSeedsFromInteropGensteps@64] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    2017-08-11 14:28:22.227 INFO  [3111812] [OContext::close@219] OContext::close numEntryPoint 2
    Process 1486 stopped
    * thread #1: tid = 0x2f7b84, 0x00000001179fb24b libgpgpucomp.dylib`CfgSets::NextFunc(NvirProgram*, NvirCallGraph*) + 155, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP
        frame #0: 0x00000001179fb24b libgpgpucomp.dylib`CfgSets::NextFunc(NvirProgram*, NvirCallGraph*) + 155
    libgpgpucomp.dylib`CfgSets::NextFunc(NvirProgram*, NvirCallGraph*) + 155:
    -> 0x1179fb24b:  testq  %rax, %rax
       0x1179fb24e:  jne    0x1179fb210               ; CfgSets::NextFunc(NvirProgram*, NvirCallGraph*) + 96
       0x1179fb250:  movl   0x298(%rsi), %r9d
       0x1179fb257:  xorl   %edi, %edi
    (lldb) bt
    * thread #1: tid = 0x2f7b84, 0x00000001179fb24b libgpgpucomp.dylib`CfgSets::NextFunc(NvirProgram*, NvirCallGraph*) + 155, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP
      * frame #0: 0x00000001179fb24b libgpgpucomp.dylib`CfgSets::NextFunc(NvirProgram*, NvirCallGraph*) + 155
        frame #1: 0x00000001179f8c71 libgpgpucomp.dylib`CfgSets::ComputeLiveInfo(NvirProgram*, NvirCfg*) + 465
        frame #2: 0x00000001179fd751 libgpgpucomp.dylib`NvirLive::GetLiveInfo(NvirProgram*) + 97
        frame #3: 0x00000001179fe66e libgpgpucomp.dylib`NvirLive::RemoveDeadInstructions(NvirProgram*, bool, bool) + 2942
        frame #4: 0x000000011796186c libgpgpucomp.dylib`NvirCopyProp::CopyProp(NvirProgram*, bool, bool) + 1500
        frame #5: 0x0000000117a9e9ed libgpgpucomp.dylib`TDriver::QueueRun() + 205
        frame #6: 0x0000000117af7a22 libgpgpucomp.dylib`ProfileData_Fermi::GenerateCodeThroughDriver(NvirProgram*) + 82
        frame #7: 0x0000000117af876a libgpgpucomp.dylib`ProfileData_Fermi::GenerateCode(NvirProgram*) + 202
        frame #8: 0x0000000117d07a7a libgpgpucomp.dylib`sm_30CodeGen + 74
        frame #9: 0x0000000117d0f07f libgpgpucomp.dylib`generateCode(ocgCompilationUnit*) + 2143
        frame #10: 0x0000000117d6161d libgpgpucomp.dylib`listTraverse + 45
        frame #11: 0x0000000117d0db85 libgpgpucomp.dylib`CompileProgram() + 4373
        frame #12: 0x0000000117d12bea libgpgpucomp.dylib`assemble(unsigned int, char**, char**) + 8442
        frame #13: 0x0000000117d109d1 libgpgpucomp.dylib`ptxAssemble + 209
        frame #14: 0x0000000117d0221f libgpgpucomp.dylib`fatBinaryCtl_Compile + 671
        frame #15: 0x0000000117518ce0 libcuda_310.40.45_mercury.dylib`cuiDeviceCodeObtainCubin + 1088
        frame #16: 0x0000000117517c4b libcuda_310.40.45_mercury.dylib`cuiModuleLoadDataExCommon + 459
        frame #17: 0x0000000117517a6d libcuda_310.40.45_mercury.dylib`cuiModuleLoadDataEx + 61
        frame #18: 0x0000000117474280 libcuda_310.40.45_mercury.dylib`cuapiModuleLoadDataEx + 272
        frame #19: 0x000000011744c556 libcuda_310.40.45_mercury.dylib`cuModuleLoadDataEx + 118
        frame #20: 0x000000010283a56d liboptix.1.dylib`___lldb_unnamed_function3218$$liboptix.1.dylib + 989
        frame #21: 0x000000010283ac58 liboptix.1.dylib`___lldb_unnamed_function3219$$liboptix.1.dylib + 280
        frame #22: 0x00000001027c6374 liboptix.1.dylib`___lldb_unnamed_function2422$$liboptix.1.dylib + 260
        frame #23: 0x00000001027b5c0e liboptix.1.dylib`___lldb_unnamed_function2287$$liboptix.1.dylib + 94
        frame #24: 0x0000000102854782 liboptix.1.dylib`___lldb_unnamed_function3522$$liboptix.1.dylib + 2402
        frame #25: 0x0000000102670bd6 liboptix.1.dylib`___lldb_unnamed_function950$$liboptix.1.dylib + 54
        frame #26: 0x00000001025c0a19 liboptix.1.dylib`rtContextCompile + 105
        frame #27: 0x00000001035408db libOptiXRap.dylib`optix::ContextObj::compile(this=0x00000001190f0720) + 43 at optixpp_namespace.h:2376
        frame #28: 0x000000010353fff4 libOptiXRap.dylib`OContext::launch(this=0x0000000119104f50, lmode=14, entry=0, width=0, height=0, times=0x000000010b96ed40) + 660 at OContext.cc:271
        frame #29: 0x0000000103555cb0 libOptiXRap.dylib`OPropagator::prelaunch(this=0x000000011ad26b40) + 592 at OPropagator.cc:144
        frame #30: 0x0000000103555eba libOptiXRap.dylib`OPropagator::launch(this=0x000000011ad26b40) + 58 at OPropagator.cc:154
        frame #31: 0x0000000103ae41da libOKOP.dylib`OpEngine::propagate(this=0x000000011290e6d0) + 58 at OpEngine.cc:100
        frame #32: 0x0000000103bd2e42 libOK.dylib`OKPropagator::propagate(this=0x000000011290e670) + 626 at OKPropagator.cc:71
        frame #33: 0x0000000103bd287d libOK.dylib`OKMgr::propagate(this=0x00007fff5fbfe5c8) + 285 at OKMgr.cc:96
        frame #34: 0x000000010000adb9 OKTest`main(argc=24, argv=0x00007fff5fbfe6a0) + 1385 at OKTest.cc:59
        frame #35: 0x00007fff8bf9a5fd libdyld.dylib`start + 1
    (lldb) c
    Process 1486 resuming
    Process 1486 stopped
    * thread #1: tid = 0x2f7b84, 0x000000011789621e libgpgpucomp.dylib`SparseBVLL::Copy(SparseBVLLAllocator*, SparseBVLL const*) + 254, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP
        frame #0: 0x000000011789621e libgpgpucomp.dylib`SparseBVLL::Copy(SparseBVLLAllocator*, SparseBVLL const*) + 254
    libgpgpucomp.dylib`SparseBVLL::Copy(SparseBVLLAllocator*, SparseBVLL const*) + 254:
    -> 0x11789621e:  movl   0x8(%rbx), %ecx
       0x117896221:  movl   %ecx, 0x8(%rax)
       0x117896224:  movl   0xc(%rbx), %ecx
       0x117896227:  movl   %ecx, 0xc(%rax)
    (lldb) bt
    * thread #1: tid = 0x2f7b84, 0x000000011789621e libgpgpucomp.dylib`SparseBVLL::Copy(SparseBVLLAllocator*, SparseBVLL const*) + 254, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP
      * frame #0: 0x000000011789621e libgpgpucomp.dylib`SparseBVLL::Copy(SparseBVLLAllocator*, SparseBVLL const*) + 254
        frame #1: 0x0000000117a55b11 libgpgpucomp.dylib`ForwardDataFlowSolver<SparseBV>::ComputeFunctionKill(int, SparseBV*, SparseBV*, SparseBV*, bool) + 689
        frame #2: 0x0000000117a54c0e libgpgpucomp.dylib`DataFlowSolver<SparseBV>::PropagateAcrossFunctions(bool, SparseBV*, SparseBV*) + 238
        frame #3: 0x0000000117a54fc3 libgpgpucomp.dylib`DataFlowSolver<SparseBV>::Solve(bool, bool, bool) + 483
        frame #4: 0x0000000117a33a95 libgpgpucomp.dylib`NvirRegAlloc::OptimizeInterblockRanges(NvirProgram*, int, int) + 1637
        frame #5: 0x0000000117a352f4 libgpgpucomp.dylib`NvirRegAlloc::AnalyzeInterblockColorRanges(NvirProgram*, int) + 1012
        frame #6: 0x0000000117a3b782 libgpgpucomp.dylib`NvirRegAlloc::ControlRegisterPressure(NvirProgram*, int, NvirCodeArray*, int) + 2338
        frame #7: 0x0000000117a445a1 libgpgpucomp.dylib`NvirRegAlloc::AllocateRegistersInClass(NvirProgram*, int, NvirCodeArray*, int, int*) + 1377
        frame #8: 0x0000000117a45bec libgpgpucomp.dylib`NvirRegAlloc::AssignRegsToColors(NvirProgram*) + 2124
        frame #9: 0x0000000117af7979 libgpgpucomp.dylib`ProfileData_Fermi::PHASE_AllocateRegisters(NvirProgram*) + 25
        frame #10: 0x0000000117b6d3b4 libgpgpucomp.dylib`NvirRegAlloc_Fermi::Run(NvirProgram*, TDriver::tStateVector*) + 20
        frame #11: 0x0000000117a9e9ed libgpgpucomp.dylib`TDriver::QueueRun() + 205
        frame #12: 0x0000000117af7a22 libgpgpucomp.dylib`ProfileData_Fermi::GenerateCodeThroughDriver(NvirProgram*) + 82
        frame #13: 0x0000000117af876a libgpgpucomp.dylib`ProfileData_Fermi::GenerateCode(NvirProgram*) + 202
        frame #14: 0x0000000117d07a7a libgpgpucomp.dylib`sm_30CodeGen + 74
        frame #15: 0x0000000117d0f07f libgpgpucomp.dylib`generateCode(ocgCompilationUnit*) + 2143
        frame #16: 0x0000000117d6161d libgpgpucomp.dylib`listTraverse + 45
        frame #17: 0x0000000117d0db85 libgpgpucomp.dylib`CompileProgram() + 4373
        frame #18: 0x0000000117d12bea libgpgpucomp.dylib`assemble(unsigned int, char**, char**) + 8442
        frame #19: 0x0000000117d109d1 libgpgpucomp.dylib`ptxAssemble + 209
        frame #20: 0x0000000117d0221f libgpgpucomp.dylib`fatBinaryCtl_Compile + 671
        frame #21: 0x0000000117518ce0 libcuda_310.40.45_mercury.dylib`cuiDeviceCodeObtainCubin + 1088
        frame #22: 0x0000000117517c4b libcuda_310.40.45_mercury.dylib`cuiModuleLoadDataExCommon + 459
        frame #23: 0x0000000117517a6d libcuda_310.40.45_mercury.dylib`cuiModuleLoadDataEx + 61
        frame #24: 0x0000000117474280 libcuda_310.40.45_mercury.dylib`cuapiModuleLoadDataEx + 272
        frame #25: 0x000000011744c556 libcuda_310.40.45_mercury.dylib`cuModuleLoadDataEx + 118
        frame #26: 0x000000010283a56d liboptix.1.dylib`___lldb_unnamed_function3218$$liboptix.1.dylib + 989
        frame #27: 0x000000010283ac58 liboptix.1.dylib`___lldb_unnamed_function3219$$liboptix.1.dylib + 280
        frame #28: 0x00000001027c6374 liboptix.1.dylib`___lldb_unnamed_function2422$$liboptix.1.dylib + 260
        frame #29: 0x00000001027b5c0e liboptix.1.dylib`___lldb_unnamed_function2287$$liboptix.1.dylib + 94
        frame #30: 0x0000000102854782 liboptix.1.dylib`___lldb_unnamed_function3522$$liboptix.1.dylib + 2402
        frame #31: 0x0000000102670bd6 liboptix.1.dylib`___lldb_unnamed_function950$$liboptix.1.dylib + 54
        frame #32: 0x00000001025c0a19 liboptix.1.dylib`rtContextCompile + 105
        frame #33: 0x00000001035408db libOptiXRap.dylib`optix::ContextObj::compile(this=0x00000001190f0720) + 43 at optixpp_namespace.h:2376
        frame #34: 0x000000010353fff4 libOptiXRap.dylib`OContext::launch(this=0x0000000119104f50, lmode=14, entry=0, width=0, height=0, times=0x000000010b96ed40) + 660 at OContext.cc:271
        frame #35: 0x0000000103555cb0 libOptiXRap.dylib`OPropagator::prelaunch(this=0x000000011ad26b40) + 592 at OPropagator.cc:144
        frame #36: 0x0000000103555eba libOptiXRap.dylib`OPropagator::launch(this=0x000000011ad26b40) + 58 at OPropagator.cc:154
        frame #37: 0x0000000103ae41da libOKOP.dylib`OpEngine::propagate(this=0x000000011290e6d0) + 58 at OpEngine.cc:100
        frame #38: 0x0000000103bd2e42 libOK.dylib`OKPropagator::propagate(this=0x000000011290e670) + 626 at OKPropagator.cc:71
        frame #39: 0x0000000103bd287d libOK.dylib`OKMgr::propagate(this=0x00007fff5fbfe5c8) + 285 at OKMgr.cc:96
        frame #40: 0x000000010000adb9 OKTest`main(argc=24, argv=0x00007fff5fbfe6a0) + 1385 at OKTest.cc:59
        frame #41: 0x00007fff8bf9a5fd libdyld.dylib`start + 1
    (lldb) c
    Process 1486 resuming
## intersect_analytic.cu:bounds pts:   2 pln:   0 trs:   6 
##csg_bounds_prim primIdx   0 partOffset   0 numParts   1 height  0 numNodes  1 tranBuffer_size   6 
##csg_bounds_prim primIdx   1 partOffset   1 numParts   1 height  0 numNodes  1 tranBuffer_size   6 
##csg_bounds_prim primIdx   0 nodeIdx  1 depth  0 elev  0 typecode 23 tranOffset  0 gtransformIdx  1 complement 0 
##csg_bounds_prim primIdx   1 nodeIdx  1 depth  0 elev  0 typecode  6 tranOffset  1 gtransformIdx  1 complement 0 

       1.000    0.000    0.000    0.000   (trIdx:  0)[vt]
       0.000    1.000    0.000    0.000

       1.000    0.000    0.000    0.000   (trIdx:  3)[vt]
       0.000    1.000    0.000    0.000

       0.000    0.000    1.000    0.000   (trIdx:  0)[vt]
       0.000    0.000    0.000    1.000

       0.000    0.000    1.000    0.000   (trIdx:  3)[vt]
       0.000    0.000    0.000    1.000
##intersect_analytic.cu:bounds primIdx 0 primFlag 101 min  -150.0000  -150.0000   -50.0000 max   150.0000   150.0000    50.0000 
##intersect_analytic.cu:bounds primIdx 1 primFlag 101 min  -400.0000  -400.0000  -400.0000 max   400.0000   400.0000   400.0000 
    2017-08-11 14:29:53.371 INFO  [3111812] [OPropagator::prelaunch@149] 1 : (0;10000,1) prelaunch_times vali,comp,prel,lnch  0.000090.1678 0.8556 0.0000
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: Kernel launch returned (719): Launch failed, [6619200])
    Process 1486 stopped
    * thread #1: tid = 0x2f7b84, 0x00007fff90b27866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff90b27866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff90b27866:  jae    0x7fff90b27870            ; __pthread_kill + 20
       0x7fff90b27868:  movq   %rax, %rdi
       0x7fff90b2786b:  jmp    0x7fff90b24175            ; cerror_nocancel
       0x7fff90b27870:  retq   
    (lldb) 
    (lldb) bt
    * thread #1: tid = 0x2f7b84, 0x00007fff90b27866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff90b27866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff881c435c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8ef14b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8e7d4f31 libc++abi.dylib`abort_message + 257
        frame #4: 0x00007fff8e7fa93a libc++abi.dylib`default_terminate_handler() + 240
        frame #5: 0x00007fff8eb32322 libobjc.A.dylib`_objc_terminate() + 124
        frame #6: 0x00007fff8e7f81d1 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff8e7f7c5b libc++abi.dylib`__cxa_throw + 124
        frame #8: 0x000000010352d4d9 libOptiXRap.dylib`optix::ContextObj::checkError(this=0x00000001190f0720, code=RT_ERROR_UNKNOWN) const + 121 at optixpp_namespace.h:1840
        frame #9: 0x0000000103540940 libOptiXRap.dylib`optix::ContextObj::launch(this=0x00000001190f0720, entry_point_index=0, image_width=10000, image_height=1) + 80 at optixpp_namespace.h:2386
        frame #10: 0x000000010354010d libOptiXRap.dylib`OContext::launch(this=0x0000000119104f50, lmode=16, entry=0, width=10000, height=1, times=0x000000010b9723f0) + 941 at OContext.cc:295
        frame #11: 0x0000000103556019 libOptiXRap.dylib`OPropagator::launch(this=0x000000011ad26b40) + 409 at OPropagator.cc:166
        frame #12: 0x0000000103ae41da libOKOP.dylib`OpEngine::propagate(this=0x000000011290e6d0) + 58 at OpEngine.cc:100
        frame #13: 0x0000000103bd2e42 libOK.dylib`OKPropagator::propagate(this=0x000000011290e670) + 626 at OKPropagator.cc:71
        frame #14: 0x0000000103bd287d libOK.dylib`OKMgr::propagate(this=0x00007fff5fbfe5c8) + 285 at OKMgr.cc:96
        frame #15: 0x000000010000adb9 OKTest`main(argc=24, argv=0x00007fff5fbfe6a0) + 1385 at OKTest.cc:59
        frame #16: 0x00007fff8bf9a5fd libdyld.dylib`start + 1
    (lldb) 

