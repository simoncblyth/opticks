G4CXTest_raindrop_CPU_with_EVENT_MODE_DebugLite_writing_only_sframe_genstep_domain
====================================================================================


This was pilot error. The G4CXTest_raindrop.sh script using the old SEvent_MakeGenstep_num_ph
when needs to ue OPTICKS_NUM_PHOTON. 



::

    ~/opticks/g4cx/tests/G4CXTest_raindrop_CPU.sh


    2024-03-28 19:42:53.919 INFO  [8929676] [SEvt::add_array@3608]  k U4R.npy a (1, )
    2024-03-28 19:42:53.920 INFO  [8929676] [SEvt::gather@3594] SEvt::id ECPU (0)  GSV YES SEvt__gather
    2024-03-28 19:42:53.920 INFO  [8929676] [SEvt::gather_components@3505]  num_comp 7 from provider SEvt
    2024-03-28 19:42:53.920 INFO  [8929676] [SEvt::gather_components@3516]  k         genstep a  <f4(1, 6, 4, ) null_component NO 
    2024-03-28 19:42:53.920 INFO  [8929676] [SEvt::gather_components@3516]  k          photon a - null_component YES
    2024-03-28 19:42:53.920 INFO  [8929676] [SEvt::gather_components@3516]  k          record a - null_component YES
    2024-03-28 19:42:53.920 INFO  [8929676] [SEvt::gather_components@3516]  k             seq a - null_component YES
    2024-03-28 19:42:53.920 INFO  [8929676] [SEvt::gather_components@3516]  k             hit a - null_component YES
    2024-03-28 19:42:53.920 INFO  [8929676] [SEvt::gather_components@3516]  k          domain a  <f4(2, 4, 4, ) null_component NO 
    2024-03-28 19:42:53.920 INFO  [8929676] [SEvt::gather_components@3516]  k        inphoton a - null_component YES
    2024-03-28 19:42:53.920 INFO  [8929676] [SEvt::gather_components@3546]  num_comp 7 num_genstep 1 num_photon -1 num_hit -1 gather_total 1 genstep_tota    

HUH sevent not populated?::

    3089 NP* SEvt::gatherPhoton() const
    3090 {
    3091     if( evt->photon == nullptr ) return nullptr ;
    3092     NP* p = makePhoton();
    3093     p->read2( (float*)evt->photon );
    3094     return p ;
    3095 }
    3096 
    3097 NP* SEvt::gatherRecord() const
    3098 {
    3099     if( evt->record == nullptr ) return nullptr ;
    3100     NP* r = makeRecord();
    3101     r->read2( (float*)evt->record );
    3102     return r ;
    3103 }


::

    BP=SEvt::addGentep ~/opticks/g4cx/tests/G4CXTest_raindrop_CPU.sh run

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000109f17980 libSysRap.dylib`SEvt::addGenstep(NP const*)
        frame #1: 0x0000000109f178ae libSysRap.dylib`SEvt::addInputGenstep() + 2798
        frame #2: 0x0000000109f21fdd libSysRap.dylib`SEvt::beginOfEvent(int) + 845
        frame #3: 0x00000001004ab2c2 libU4.dylib`U4Recorder::BeginOfEventAction_(this=0x000000010add1bf0, eventID_=0) at U4Recorder.cc:294
        frame #4: 0x00000001000557c4 G4CXTest`G4CXApp::BeginOfEventAction(this=0x000000010add1b60, event=0x000000010afb16b0) at G4CXApp.h:267


Dumping the input genstep quad6 shows numphoton 0::


    .   2042	
       2043	    int gidx = int(gs.size())  ;  // 0-based genstep label index
       2044	    bool enabled = GIDX == -1 || GIDX == gidx ; 
       2045	
       2046	    quad6& q = const_cast<quad6&>(q_);  
       2047	    if(!enabled) q.set_numphoton(0);   
    (lldb) p q_
    (const quad6) $3 = {
      q0 = {
        f = (x = 0.00000000000000000000000000000000000000000000840779078, y = 0, z = 0, w = 0)
        i = (x = 6, y = 0, z = 0, w = 0)
        u = (x = 6, y = 0, z = 0, w = 0)
      }
      q1 = {
        f = (x = -80, y = 0, z = 0, w = 0)
        i = (x = -1029701632, y = 0, z = 0, w = 0)
        u = (x = 3265265664, y = 0, z = 0, w = 0)
      }

::

    059 
     60 if [ "$OPTICKS_RUNNING_MODE" == "SRM_TORCH" ]; then
     61     export SEvent_MakeGenstep_num_ph=$NUM
     62     #src="rectangle"
     63     src="disc"
     64 
     65     ## TODO: shoot all angles from just inside drop to check after TIR speed
     66 
     67     if [ "$src" == "rectangle" ]; then
     68         export storch_FillGenstep_pos=0,0,0
     69         export storch_FillGenstep_type=rectangle
     70         export storch_FillGenstep_zenith=-20,20
     71         export storch_FillGenstep_azimuth=-20,20
     72     elif [ "$src" == "disc" ]; then
     73         export storch_FillGenstep_type=disc
     74         export storch_FillGenstep_radius=50        # radius
     75         export storch_FillGenstep_zenith=0,1       # radial range scale
     76         export storch_FillGenstep_azimuth=0,1      # phi segment twopi fraction 
     77         export storch_FillGenstep_mom=1,0,0
     78         export storch_FillGenstep_pos=-80,0,0
     79     fi
     80 fi


    156 NP* SEvent::MakeGenstep( int gentype, int index_arg )
    157 {
    158     bool with_index = index_arg != -1 ;
    159     if(with_index) assert( index_arg >= 0 );  // index_arg is 0-based 
    160     int num_ph = with_index ? SEventConfig::NumPhoton(index_arg) : ssys::getenvint("SEvent_MakeGenstep_num_ph", 100 ) ;
    161     bool dump = ssys::getenvbool("SEvent_MakeGenstep_dump");
    162     unsigned num_gs = 1 ;
    163     const int M = 1000000 ;
    164 
    165     LOG(LEVEL)
    166         << " gentype " << gentype
    167         << " index_arg " << index_arg
    168         << " with_index " << ( with_index ? "YES" : "NO " )
    169         << " num_ph " << num_ph
    170         << " num_ph/M " << num_ph/M
    171         << " dump " << dump
    172         ;
    173 
    174     NP* gs = NP::Make<float>(num_gs, 6, 4 );
    175     gs->set_meta<std::string>("creator", "SEvent::MakeGenstep" );
    176     gs->set_meta<int>("num_ph", num_ph );
    177     gs->set_meta<int>("index_arg",  index_arg );
    178 
    179     switch(gentype)
    180     {
    181         case  OpticksGenstep_TORCH:         FillGenstep<storch>(   gs, num_ph, dump) ; break ;
    182         case  OpticksGenstep_CERENKOV:      FillGenstep<scerenkov>(gs, num_ph, dump) ; break ;
    183         case  OpticksGenstep_SCINTILLATION: FillGenstep<sscint>(   gs, num_ph, dump) ; break ;
    184         case  OpticksGenstep_CARRIER:       FillGenstep<scarrier>( gs, num_ph, dump) ; break ;
    185     }
    186     return gs ;
    187 }


::

    2024-03-28 20:18:02.116 INFO  [9062804] [G4CXApp::GeneratePrimaries@223] [ SEventConfig::RunningModeLabel SRM_TORCH eventID 0
    2024-03-28 20:18:02.116 INFO  [9062804] [*SEvent::MakeGenstep@165]  gentype 6 index_arg 0 with_index YES num_ph 0 num_ph/M 0 dump 0
    SGenerate::GeneratePhotons SGenerate__GeneratePhotons_RNG_PRECOOKED : NO 
    U4VPrimaryGenerator::GeneratePrimaries_From_Photons ph (0, 4, 4, )
    2024-03-28 20:18:02.116 INFO  [9062804] [SEvent::SetGENSTEP@43]  GENSTEP (1, 6, 4, )

::

    EVENT_DEBUG=1 BP=SEvent::MakeGenstep ~/opticks/g4cx/tests/G4CXTest_raindrop_CPU.sh dbg 


    (lldb) f 1
    frame #1: 0x0000000109ec572a libSysRap.dylib`SEvent::MakeTorchGenstep(idx_arg=0) at SEvent.cc:140
       137 	
       138 	**/
       139 	
    -> 140 	NP* SEvent::MakeTorchGenstep(   int idx_arg){    return MakeGenstep( OpticksGenstep_TORCH, idx_arg ) ; }
       141 	NP* SEvent::MakeCerenkovGenstep(int idx_arg){ return MakeGenstep( OpticksGenstep_CERENKOV, idx_arg ) ; }
       142 	NP* SEvent::MakeScintGenstep(   int idx_arg){    return MakeGenstep( OpticksGenstep_SCINTILLATION, idx_arg ) ; }
       143 	NP* SEvent::MakeCarrierGenstep( int idx_arg){  return MakeGenstep( OpticksGenstep_CARRIER, idx_arg ) ; }
    (lldb) 

    (lldb) f 2
    frame #2: 0x0000000100054b6e G4CXTest`G4CXApp::GeneratePrimaries(this=0x000000010ae9cf40, event=0x000000010aeb3d20) at G4CXApp.h:236
       233 	    else if(SEventConfig::IsRunningModeTorch())
       234 	    {
       235 	        int idx_arg = eventID ; 
    -> 236 	        NP* gs = SEvent::MakeTorchGenstep(idx_arg) ;        
       237 	        NP* ph = SGenerate::GeneratePhotons(gs); 
       238 	        U4VPrimaryGenerator::GeneratePrimaries_From_Photons(event, ph);
       239 	        delete ph ; 
    (lldb) 






G4CXTest_raindrop_CPU_flag_zero_assert
------------------------------------------


::

    2024-03-28 20:24:10.241 ERROR [9098247] [U4StepPoint::Flag@169]  U4OpBoundaryProcess::GetStatus<T>() : Undefined 
     U4OpBoundaryProcess::Get<T>() NO 
     U4Physics::Switches() 
    U4Physics::Switches
    WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:WITH_CUSTOM4_AND_WITH_PMTSIM
    WITH_CUSTOM4_AND_NOT_WITH_PMTSIM
    DEBUG_TAG

    2024-03-28 20:24:10.242 ERROR [9098247] [U4StepPoint::Flag@181]  UNEXPECTED BoundaryFlag ZERO  
     flag 0 OpticksPhoton::Flag(flag) .
     bstat 0 U4OpBoundaryProcessStatus::Name(bstat) Undefined
    2024-03-28 20:24:10.241 ERROR [9098247] [U4Recorder::UserSteppingAction_Optical@986]  ERR flag zero : post 
    U4StepPoint::DescPositionTime(post)
    U4StepPoint::DescPositionTime (    -43.831    -21.761     10.261      0.121)
    U4StepPoint::Desc<T>(post)
    U4StepPoint::Desc
     proc 2 procName Transportation procNameRaw Transportation
     status 1 statusName fGeomBoundary
     bstat 0 bstatName Undefined
     flag 0 flagName .
    Assertion failed: (flag > 0), function UserSteppingAction_Optical, file /Users/blyth/opticks/u4/U4Recorder.cc, line 998.
    Process 36143 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff50698b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff50698b66 <+10>: jae    0x7fff50698b70            ; <+20>
        0x7fff50698b68 <+12>: movq   %rax, %rdi
        0x7fff50698b6b <+15>: jmp    0x7fff5068fae9            ; cerror_nocancel
        0x7fff50698b70 <+20>: retq   
    Target 0: (G4CXTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff50698b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff50863080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff505f41ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff505bc1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001004ae1f0 libU4.dylib`void U4Recorder::UserSteppingAction_Optical<C4OpBoundaryProcess>(this=0x000000010a954a90, step=0x000000010ad17dc0) at U4Recorder.cc:998
        frame #5: 0x00000001004ad156 libU4.dylib`U4Recorder::UserSteppingAction(this=0x000000010a954a90, step=0x000000010ad17dc0) at U4Recorder.cc:342
        frame #6: 0x00000001000559c1 G4CXTest`G4CXApp::UserSteppingAction(this=0x000000010a954a00, step=0x000000010ad17dc0) at G4CXApp.h:309
        frame #7: 0x00000001000559fc G4CXTest`non-virtual thunk to G4CXApp::UserSteppingAction(this=0x000000010a954a00, step=0x000000010ad17dc0) at G4CXApp.h:0
        frame #8: 0x0000000102311f06 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010ad17c30) at G4SteppingManager.cc:243
        frame #9: 0x000000010232886f libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010ad17bf0, apValueG4Track=0x000000010cd5e140) at G4TrackingManager.cc:126
        frame #10: 0x00000001021ee71a libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010ad17b60, anEvent=0x000000010aa60790) at G4EventManager.cc:185
        frame #11: 0x00000001021efc2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x000000010ad17b60, anEvent=0x000000010aa60790) at G4EventManager.cc:338
        frame #12: 0x00000001020fb9e5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010ad17980, i_event=0) at G4RunManager.cc:399
        frame #13: 0x00000001020fb815 libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010ad17980, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #14: 0x00000001020f9cd1 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010ad17980, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #15: 0x000000010005615f G4CXTest`G4CXApp::BeamOn(this=0x000000010a954a00) at G4CXApp.h:344
        frame #16: 0x000000010005628a G4CXTest`G4CXApp::Main() at G4CXApp.h:351
        frame #17: 0x00000001000564bc G4CXTest`main(argc=1, argv=0x00007ffeefbfe568) at G4CXTest.cc:13
        frame #18: 0x00007fff50548015 libdyld.dylib`start + 1
        frame #19: 0x00007fff50548015 libdyld.dylib`start + 1
    (lldb) 


AHAH: this is process mismatch because not finding the SPMT info in the raindrop geom caused to 
switch to the InstrumentedBoundaryProcess.::

    (lldb) f 6
    frame #6: 0x00000001000559c1 G4CXTest`G4CXApp::UserSteppingAction(this=0x000000010a954a00, step=0x000000010ad17dc0) at G4CXApp.h:309
       306 	
       307 	void G4CXApp::PreUserTrackingAction(const G4Track* trk){  fRecorder->PreUserTrackingAction(trk); }
       308 	void G4CXApp::PostUserTrackingAction(const G4Track* trk){ fRecorder->PostUserTrackingAction(trk); }
    -> 309 	void G4CXApp::UserSteppingAction(const G4Step* step){     fRecorder->UserSteppingAction(step) ; }
       310 	
       311 	void G4CXApp::OpenGeometry(){  G4GeometryManager::GetInstance()->OpenGeometry(); } // static
       312 	G4CXApp::~G4CXApp(){ OpenGeometry(); }
    (lldb) f 5
    frame #5: 0x00000001004ad156 libU4.dylib`U4Recorder::UserSteppingAction(this=0x000000010a954a90, step=0x000000010ad17dc0) at U4Recorder.cc:342
       339 	    if(!U4Track::IsOptical(step->GetTrack())) return ; 
       340 	
       341 	#if defined(WITH_CUSTOM4)
    -> 342 	     UserSteppingAction_Optical<C4OpBoundaryProcess>(step); 
       343 	#elif defined(WITH_PMTSIM)
       344 	     UserSteppingAction_Optical<CustomG4OpBoundaryProcess>(step); 
       345 	#else
    (lldb) 




Real solution is to make C4OpBoundaryProcess work without PMT info, 
it can assert just when PMT info actually needed.::

     118 
     119 C4OpBoundaryProcess::C4OpBoundaryProcess(
     120                                                const C4IPMTAccessor* accessor,
     121                                                const G4String& processName,
     122                                                G4ProcessType type)
     123              :
     124              G4VDiscreteProcess(processName, type),
     125              m_custom_status('U'),
     126              m_custom_art(new C4CustomART(
     127                                         accessor,
     128                                         theAbsorption,
     129                                         theReflectivity,
     130                                         theTransmittance,
     131                                         theEfficiency,
     132                                         theGlobalPoint,
     133                                         OldMomentum,
     134                                         OldPolarization,
     135                                         theRecoveredNormal,
     136                                         thePhotonMomentum
     137                                        ))
     138 {


HMM seems to work already.



Change raindrop to box shaped "drop" and change to circle_inwards_hemi 
directed at single point on drop surface

::

    np.c_[np.unique(b.q, return_counts=True)] 
    [[b'TO BR BR BR BR BR BR BR BT SA                                                                   ' b'2']
     [b'TO BR BR BR BR BR BR BT SA                                                                      ' b'12']
     [b'TO BR BR BR BR BR BT SA                                                                         ' b'45']
     [b'TO BR BR BR BR BT SA                                                                            ' b'109']
     [b'TO BR BR BR BT SA                                                                               ' b'880']
     [b'TO BR BR BT SA                                                                                  ' b'2465']
     [b'TO BR BT SA                                                                                     ' b'46578']
     [b'TO BT SA                                                                                        ' b'49909']]

    PICK=B MODE=3 SELECT="TO BR BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BR : 224.901 224.901 
    speed min/max for : 1 -> 2 : BR -> BR : 224.901 224.901 
    speed min/max for : 2 -> 3 : BR -> BT : 224.892 224.912 
    speed min/max for : 3 -> 4 : BT -> SA : 299.792 299.793 
    _pos.shape (2465, 3) 
    _beg.shape (2465, 3) 


When disable UseGivenVelocity the wrong velocity appears::

    export U4Recorder__PreUserTrackingAction_Optical_DISABLE_UseGivenVelocity=1 

    PICK=B MODE=3 SELECT="TO BR BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BR : 224.901 224.901 
    speed min/max for : 1 -> 2 : BR -> BR : 299.792 299.793 
    speed min/max for : 2 -> 3 : BR -> BT : 299.723 299.806 
    speed min/max for : 3 -> 4 : BT -> SA : 224.900 224.901 
    _pos.shape (2465, 3) 

    PICK=B MODE=3 SELECT="TO BR BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BR : 224.901 224.901 
    speed min/max for : 1 -> 2 : BR -> BR : 224.901 224.901 
    speed min/max for : 2 -> 3 : BR -> BT : 224.892 224.912 
    speed min/max for : 3 -> 4 : BT -> SA : 299.792 299.793 
    _pos.shape (2465, 3) 
    _beg.shape (2465, 3) 





::

    ~/o/examples/UseGeometryShader/run.sh 

