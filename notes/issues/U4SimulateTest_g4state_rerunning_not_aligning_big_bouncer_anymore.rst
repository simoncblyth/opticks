U4SimulateTest_g4state_rerunning_not_aligning_big_bouncer_anymore
====================================================================

From jfs : j/PMTFastSim/junoPMTOpticalModel_vs_CustomBoundaryART_propagation_time_discrepancy.rst



Try N=1 rerun::

    U4Recorder::PreUserTrackingAction_Optical@169:  setting rerun_id 726
    U4Recorder::PreUserTrackingAction_Optical@177:  labelling photon : track 0x7fc76a275730 label 0x7fc76a276620 label.desc spho (gs:ix:id:gn   0 726  726[  0,  0,  0,  0])
    U4Recorder::saveOrLoadStates@282:  cannot U4Engine::RestoreState with null g4state 
    Assertion failed: (g4state), function saveOrLoadStates, file /Users/blyth/opticks/u4/U4Recorder.cc, line 283.
    ./U4SimulateTest.sh: line 150: 38837 Abort trap: 6           $bin
    ./U4SimulateTest.sh run error
    epsilon:tests blyth$ 

    (lldb) f 4
    frame #4: 0x00000001002427d9 libU4.dylib`U4Recorder::saveOrLoadStates(this=0x0000000107bf5940, id=726) at U4Recorder.cc:283
       280      {
       281          const NP* g4state = sev->getG4State(); 
       282          LOG_IF( fatal, g4state == nullptr ) << " cannot U4Engine::RestoreState with null g4state " ; 
    -> 283          assert( g4state ); 
       284  


The SEvt load is looking in ALL not ALL0 or ALL1::

    U::DirList path /tmp/blyth/opticks/GEOM/hamaLogicalPMT/U4SimulateTest/ALL ext - NO ENTRIES FOUND 
    main@73:  reldir SEL0 rerun 1

    epsilon:tests blyth$ l /tmp/blyth/opticks/GEOM/hamaLogicalPMT/U4SimulateTest/
    total 0
    0 drwxr-xr-x  22 blyth  wheel  704 Dec 13 14:14 ALL0


Rearrange rerun load/save to load from the alldir and save to the seldir. 


Looks like running off different g4state::

    vimdiff U4SimulateTest_0.log U4SimulateTest_1.log


Comparing g4state between ALL0 and ALL1::

    ./U4SimulateTest.sh af 

    In [17]: np.all( a.g4state[-102:]  == b.g4state[-102:]  )
    Out[17]: True

Note lots of pid are not the same, but lots are.

Try reworking the SEvt handling to get N=0 and N=1 starting 
from the same random states::

     70     int g4state_rerun_id = SEventConfig::G4StateRerun();
     71     bool rerun = g4state_rerun_id > -1 ;
     72     const char* seldir = U::FormName( "SEL", VERSION, nullptr );
     73     const char* alldir = U::FormName( "ALL", VERSION, nullptr );
     74     const char* alldir0 = U::FormName( "ALL", 0, nullptr );      
     75     
     76     LOG(info) 
     77         << " g4state_rerun_id " << g4state_rerun_id
     78         << " alldir " << alldir 
     79         << " alldir0 " << alldir0
     80         << " seldir " << seldir
     81         << " rerun " << rerun
     82         ;
     83     
     84     SEvt* evt = nullptr ;
     85     if(rerun == false)
     86     {   
     87         evt = SEvt::Create();  
     88         evt->setReldir(alldir);
     89     }  
     90     else
     91     {   
     92         evt = SEvt::Load(alldir0) ;
     93         evt->clear_partial("g4state");  // clear loaded evt but keep g4state 
     94         evt->setReldir(seldir);
     95         // when rerunning have to load states from alldir0 and then change reldir to save into seldir
     96     }
     97     // HMM: note how reldir at object rather then static level is a bit problematic for loading 
     98     



This incantation succeeds to rerun the big bouncer in N=1::

    vi U4SimulateTest.sh      ## switch to running_mode=SRM_G4STATE_SAVE

    N=0 ./U4SimulateTest.sh   ## saves g4state into /tmp/blyth/opticks/GEOM/hamaLogicalPMT/U4SimulateTest/ALL0

    vi U4SimulateTest.sh      ## switch to running_mode=SRM_G4STATE_RERUN for PID 726 
    
    N=1 ./U4SimulateTest.sh   ## loads g4state from ALL0, saves into SEL1



Plotting that with U4SimtraceTest.sh  ?
------------------------------------------

::


    N=0 APID=726 BPID=-1 AOPT=nrm FOCUS=0,10,185 ./U4SimtraceTest.sh ana
    ## nrm (coming from aux) not being set for FastSim points 

    N=1 APID=-1 BPID=726 BOPT=nrm FOCUS=0,10,185 ./U4SimtraceTest.sh ana
    



