OKG4Test_prelaunch_FPE_causing_fail FIXED with C4FPEDetection
=================================================================

* have not seen this problem on macOS, only Linux
* root cause is using a debug Geant4 build which switches on the FPE detection 
* bites me in ckm- CerenkovMinimal too 

* need a higher level way to switch off the FPE detection in G4Opticks, 
  is that finniky abut when it gets done ... would guess not, just try doing it 
  when G4Opticks is first invoked ?

  * hmm should G4Opticks be exposing the underlying Opticks to users or not ?



It appears the prelaunch width,height of 0,0 that have been using forever actually causes an FPE inside OptiX 
that shows up only in OKG4Test and not OKTest for example : because G4 debug build has G4FPE_DEBUG defined ?? 

Not quite, changing to width,height to 1,1 did not avoid the issue.  BUT disabling FPE detection did. 

* https://bitbucket.org/simoncblyth/opticks/commits/7acf764e69522bab694fef47efa6001af0f2b74e
* disabling the FPEDetection from G4 using C4FPEDetection succeeds to get OKG4Test to prelaunch normally and complete



OKG4Test prelaunch fail::

    2018-08-29 21:20:07.052 INFO  [32311] [OContext::upload@406] UPLOAD_WITH_CUDA markDirty DONE (1,6,4)  NumBytes(0) 96 NumBytes(1) 96 NumValues(0) 24 NumValues(1) 24{}
    TBuf::TBuf.m_spec : dev_ptr 0x7f4d91e00000 size 6 num_bytes 96 hexdump 0 
    TBuf::TBuf.m_spec : dev_ptr 0x7f4da8a00000 size 10000 num_bytes 40000 hexdump 0 
    2018-08-29 21:20:07.064 INFO  [32311] [OContext::close@244] OContext::close numEntryPoint 1
    2018-08-29 21:20:07.064 INFO  [32311] [OContext::close@248] OContext::close setEntryPointCount done.
    2018-08-29 21:20:07.086 INFO  [32311] [OContext::close@254] OContext::close m_cfg->apply() done.
    2018-08-29 21:20:07.086 INFO  [32311] [OContext::launch@273] OContext::launch entry 0 width 0 height 0
    2018-08-29 21:20:07.088 INFO  [32311] [OContext::launch@284] OContext::launch VALIDATE time: 0.002238
    2018-08-29 21:20:07.088 INFO  [32311] [OContext::launch@291] OContext::launch COMPILE time: 9e-06
    ERROR: 8 - Floating point invalid operation.

    Call Stack:
    27: /home/blyth/local/opticks/externals/lib64/libG4run.so
    26: /home/blyth/local/opticks/externals/lib64/libG4run.so
    25: /usr/lib64/libpthread.so.0
    24: /usr/local/OptiX_510/lib64/liboptix.so.51
    23: /usr/local/OptiX_510/lib64/liboptix.so.51
    22: /usr/local/OptiX_510/lib64/liboptix.so.51
    21: /usr/local/OptiX_510/lib64/liboptix.so.51
    20: /usr/local/OptiX_510/lib64/liboptix.so.51
    19: /usr/local/OptiX_510/lib64/liboptix.so.51
    18: /usr/local/OptiX_510/lib64/liboptix.so.51
    17: /usr/local/OptiX_510/lib64/liboptix.so.51
    16: /usr/local/OptiX_510/lib64/liboptix.so.51
    15: /usr/local/OptiX_510/lib64/liboptix.so.51
    14: /usr/local/OptiX_510/lib64/liboptix.so.51
    13: /usr/local/OptiX_510/lib64/liboptix.so.51
    12: /usr/local/OptiX_510/lib64/liboptix.so.51
    11: /home/blyth/local/opticks/lib64/libOptiXRap.so
    10: /home/blyth/local/opticks/lib64/libOptiXRap.so : OContext::launch_(unsigned int, unsigned int, unsigned int)
    9: /home/blyth/local/opticks/lib64/libOptiXRap.so : OContext::launch(unsigned int, unsigned int, unsigned int, unsigned int, STimes*)
    8: /home/blyth/local/opticks/lib64/libOptiXRap.so : OPropagator::prelaunch()
    7: /home/blyth/local/opticks/lib64/libOptiXRap.so : OPropagator::launch()
    6: /home/blyth/local/opticks/lib64/libOKOP.so : OpEngine::propagate()
    5: /home/blyth/local/opticks/lib64/libOK.so : OKPropagator::propagate()
    4: /home/blyth/local/opticks/lib64/libOKG4.so : OKG4Mgr::propagate_()
    3: /home/blyth/local/opticks/lib64/libOKG4.so : OKG4Mgr::propagate()
    2: OKG4Test
    1: /usr/lib64/libc.so.6
    0: OKG4Test
    Aborted (core dumped)
    [blyth@localhost opticks]$ o


OKG4Test prior to Geant4 banner get "WARNING - FPE detection is activated"::


    2018-08-29 21:19:50.548 FATAL [32311] [SLog::SLog@12] CG4::CG4 

            ############################################
            !!! WARNING - FPE detection is activated !!!
            ############################################

    **************************************************************
     Geant4 version Name: geant4-10-04-patch-02    (25-May-2018)
                           Copyright : Geant4 Collaboration
                          References : NIM A 506 (2003), 250-303
                                     : IEEE-TNS 53 (2006), 270-278
                                     : NIM A 835 (2016), 186-225
                                 WWW : http://geant4.org/
    **************************************************************



Chase that::

   epsilon:optixrap blyth$ g4-cc G4FPEDetection
   /usr/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManagerKernel.cc:  #include "G4FPEDetection.hh"

      67 #ifdef G4FPE_DEBUG
      68   #include "G4FPEDetection.hh"
      69 #endif

::

      81 G4RunManagerKernel::G4RunManagerKernel()
      82 : physicsList(0),currentWorld(0),
      83  geometryInitialized(false),physicsInitialized(false),
      84  geometryToBeOptimized(true),
      85  physicsNeedsToBeReBuilt(true),verboseLevel(0),
      86  numberOfParallelWorld(0),geometryNeedsToBeClosed(true),
      87  numberOfStaticAllocators(0)
      88 {
      89 #ifdef G4FPE_DEBUG
      90   InvalidOperationDetection();
      91 #endif



g4-cls G4FPEDetection::

    141   static void InvalidOperationDetection()
    142   {
    143     std::cout << std::endl
    144               << "        "
    145               << "############################################" << std::endl
    146               << "        "
    147               << "!!! WARNING - FPE detection is activated !!!" << std::endl
    148               << "        "
    149               << "############################################" << std::endl;
    150 
    151     (void) feenableexcept( FE_DIVBYZERO );
    152     (void) feenableexcept( FE_INVALID );
    153     //(void) feenableexcept( FE_OVERFLOW );
    154     //(void) feenableexcept( FE_UNDERFLOW );
    155 
    156     sigfillset(&termaction.sa_mask);
    157     sigdelset(&termaction.sa_mask,SIGFPE);
    158     termaction.sa_sigaction=TerminationSignalHandler;
    159     termaction.sa_flags=SA_SIGINFO;
    160     sigaction(SIGFPE, &termaction, &oldaction);
    161   }



