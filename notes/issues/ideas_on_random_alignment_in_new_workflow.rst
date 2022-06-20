ideas_on_random_alignment_in_new_workflow
===========================================

* from :doc:`U4RecorderTest_cf_CXRaindropTest`

Aligning the simulations requires:

1. same random streams 
2. same randoms get used for the same purposes in the two simulations. 

How to do that.

1. devise simtags enumeration for random consumptions that can be derived eg from the backtrace (or from GPU logging) 
   collect these names into simstream arrays 

   * whilst developing could also collect the random values to check are getting them all 
   * advantage of doing this from backtraces is that it can be automated, so can do for millions of photons

2. some simple code to read the two simstreams and present them together, 
   so can see where the "zippers" are not aligned  

3. study G4 and Opticks code to find "burn" random consumptions that are actually not being used, insert corresponding 
   burns (with instrumentation enumeration tags) and possibly reorder curand calls into GPU code 
   to get the consumption to line up 


GPU side simstream
---------------------

* doing this from GPU logfile parsing is inherently limited to small stats

* would be good to run the GPU code on the CPU, so could use same SBacktrace machinery 

  * BUT that is a lot of work to setup, requiring prd and state captures or mocking texture lookups CPU side 

* GPU side are in control of all the code doing the consumption so can devise an enumeration for all 
  the curand_uniform callsite and write those enumerations into GPU side callsite/simstream array 

::

    epsilon:qudarap blyth$ grep curand_uniform qsim.h | wc -l 
          23

* if the number of active callsite were less than 16 it would be convenient for nibble packing 
* this enumeration should be reusable CPU side : it can have GPU side natural names eg::

     to_boundary_SI_burn 
     to_boundary_AB
     to_boundary_SC 



CPU side simstream : many consumptions from G4 internals : so have to use SBacktrace for a complete picture
----------------------------------------------------------------------------------------------------------------

Review the start of the consumption deciding on the winning process for a step (~5 consumptions)

* :doc:`G4SteppingManager_DefinePhysicalStepLength`



On the CPU side SBacktrace.hh provides an automated way to collect backtraces, eg::

   U4Random_select=-1,0,-1,1 U4Random_select_action=backtrace ./U4RecorderTest.sh run
       ##  dump the backtrace for the first and second random consumption "cursor 0 and 1" of all photons pidx:"-1" 

::

    2022-06-20 09:43:30.460 INFO  [27161425] [U4Random::flat@416]  m_seq_index    0 m_seq_nv  256 cursor    0 idx    0 d    0.74022
    2022-06-20 09:43:30.460 INFO  [27161425] [U4Random::flat@430] U4Random_select -1,0,-1,1 m_select->size 4 (-1,0) YES  (-1,1) NO 
    SBacktrace::Dump addrlen 17
    SFrames..
    0   libSysRap.dylib                     0x0000000111bf7c7b SBacktrace::Dump(std::__1::basic_ostream<char, std::__1::char_traits<char> >&)                       + 107      
    1   libSysRap.dylib                     0x0000000111bf7bfb SBacktrace::Dump()                                                                                   + 27       
    2   libU4.dylib                         0x000000010c18b53c U4Random::flat()                                                                                     + 2348     
    3   libG4processes.dylib                0x000000010f6a96da G4VProcess::ResetNumberOfInteractionLengthLeft()                                                     + 42       
    4   libG4processes.dylib                0x000000010f6abd0b G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*) + 91       
    5   libG4tracking.dylib                 0x000000010deffff0 G4VProcess::PostStepGPIL(G4Track const&, double, G4ForceCondition*)                                  + 80       
    6   libG4tracking.dylib                 0x000000010deffa1a G4SteppingManager::DefinePhysicalStepLength()                                                        + 298      
    7   libG4tracking.dylib                 0x000000010defcc3a G4SteppingManager::Stepping()                                                                        + 394      
    8   libG4tracking.dylib                 0x000000010df1386f G4TrackingManager::ProcessOneTrack(G4Track*)                                                         + 1679     
    9   libG4event.dylib                    0x000000010ddd871a G4EventManager::DoProcessing(G4Event*)                                                               + 3306     
    10  libG4event.dylib                    0x000000010ddd9c2f G4EventManager::ProcessOneEvent(G4Event*)                                                            + 47       
    11  libG4run.dylib                      0x000000010dce59e5 G4RunManager::ProcessOneEvent(int)                                                                   + 69       
    12  libG4run.dylib                      0x000000010dce5815 G4RunManager::DoEventLoop(int, char const*, int)                                                     + 101      
    13  libG4run.dylib                      0x000000010dce3cd1 G4RunManager::BeamOn(int, char const*, int)                                                          + 193      
    14  U4RecorderTest                      0x000000010c05a04a main + 1402
    15  libdyld.dylib                       0x00007fff72c44015 start + 1
    16  ???                                 0x0000000000000001 0x0 + 1
    2022-06-20 09:43:30.460 INFO  [27161425] [U4Random::flat@416]  m_seq_index    0 m_seq_nv  256 cursor    1 idx    1 d    0.43845


Problem with the backtrace. 

* no easy to automate way to see which process is doing this consumption (in debugger can find this by looking at fCurrentProcess in "f 4") 
* TODO: look at cfg4/CProcessManager probably can query Geant4 to get the relevant processes and their order when U4Random::flat gets called 
* could be unecessary sledgehammer as not many processes and probably the ordering can be discerned manually : so long as its consistent


