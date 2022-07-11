U4Stack_Linux_Darwin_difference
==================================


* Next :doc:`input_photons_in_MOI_target_frame`


Overview : forced to switch to manual consumption tagging 
---------------------------------------------------------------

Auto tagging approach that works on Darwin cannot work eaily on Linux as the Linux SBacktrace::Summary 
is missing crucial line with "DsG4Scintillation::ResetNumberOfInteractionLengthLeft"
But scintillation is the only G4VRestDiscreteProcess in play so do not need to Shim.
But same problem with other processes. 

The deficient SBacktrace::Summary might arise from the inline imps.
Changing to .cc rather than .h imps made no difference to SBacktrace::Summary.
So was forced to switch to manual tagging. While more code is needed for 
manual tagging it has significant speed advantage. 


Processes
------------

::

    class DsG4Scintillation   : public G4VRestDiscreteProcess, public G4UImessenger

    class G4VRestDiscreteProcess : public G4VProcess 



    class G4OpAbsorption      : public G4VDiscreteProcess

    class G4OpRayleigh        : public G4VDiscreteProcess

    class G4OpBoundaryProcess : public G4VDiscreteProcess

    class G4VDiscreteProcess     : public G4VProcess




More systematic way to find untagged consumption
---------------------------------------------------

::

    474 #ifdef DEBUG_TAG
    475 /**
    476 U4Random::check_cursor_vs_tagslot
    477 ----------------------------------
    478 
    479 This is called by setSequenceIndex with index -1 signalling the end 
    480 of the index. A comparison between the below counts is made:
    481 
    482 * number of randoms provided by U4Random::flat for the last m_seq_index as indicated by the cursor 
    483 * random consumption tags added with SEvt::AddTag
    484 
    485 **/
    486 
    487 void U4Random::check_cursor_vs_tagslot()
    488 {
    489     assert(m_seq_index > -1) ;  // must not call when disabled, use G4UniformRand to use standard engine
    490     int cursor = *(m_cur_values + m_seq_index) ;  // get the cursor value to use for this generation, starting from 0 
    491     int slot = SEvt::GetTagSlot();
    492     bool cursor_slot_match = cursor == slot ;
    493 
    494     //LOG(info) << " m_seq_index " << m_seq_index << " cursor " << cursor << " slot " << slot << " cursor_slot_match " << cursor_slot_match ; 
    495 
    496     if(!cursor_slot_match)
    497     {
    498         m_problem_idx.push_back(m_seq_index);
    499         LOG(error)
    500             << " m_seq_index " << m_seq_index
    501             << " cursor " << cursor
    502             << " slot " << slot
    503             << " cursor_slot_match " << cursor_slot_match
    504             << std::endl
    505             << " PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS "
    506             ;
    507     }
    508 }
    509 #endif



Dump the idx with untagged consumptions::

    2022-07-11 17:41:35.674 INFO  [54066] [U4Recorder::EndOfRunAction@79] 
    2022-07-11 17:41:35.674 INFO  [54066] [main@199] /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/J000/ALL
    U4Random::saveProblemIdx m_problem_idx.size 493 (9993 9986 9973 9922 9904 9897 9867 9850 9824 9802 9799 9785 9775 9755 9751 9726 9724 9707 9696 9679 9659 9654 9584 9492 9461 9422 9419 9413 9401 9367 9321 9311 9298 9281 9272 9264 9255 9254 9225 9196 9189 9187 9175 9162 9150 9123 9084 9007 8998 8989 8987 8924 8863 8855 8813 8808 8802 8773 8766 8765 8750 8746 8735 8699 8695 8683 8675 8669 8629 8608 8589 8585 8567 8564 8543 8541 8531 8503 8451 8438 8433 8427 8413 8394 8378 8363 8344 8318 8297 8229 8222 8171 8136 8100 8092 8050 8036 8006 8002 7992 7985 7961 7948 7931 7926 7923 7917 7901 7896 7884 7868 7838 7790 7769 7762 7754 7752 7733 7719 7683 7635 7627 7623 7609 7579 7571 7568 7534 7518 7510 7505 7489 7480 7431 7378 7370 7350 7347 7322 7311 7302 7291 7279 7219 7191 7168 7069 7049 7020 6957 6907 6836 6776 6761 6755 6751 6704 6696 6648 6589 6586 6562 6550 6542 6518 6514 6512 6508 6493 6467 6426 6420 6390 6384 6369 6356 6338 6302 6266 6242 6241 6152 6150 6100 6088 6083 6021 6015 6008 5974 5958 5916 5914 5907 5868 5863 5825 5777 5773 5753 5751 5745 5708 5691 5688 5662 5649 5523 5441 5413 5408 5356 5352 5219 5126 5116 5103 5097 5081 5077 5055 5046 5036 5020 5011 4986 4944 4925 4883 4873 4798 4759 4755 4753 4747 4736 4649 4646 4623 4605 4597 4595 4547 4534 4529 4521 4519 4507 4473 4444 4415 4397 4377 4375 4368 4350 4341 4337 4310 4287 4260 4247 4163 4158 4129 4066 4040 3983 3973 3967 3966 3952 3941 3935 3932 3899 3882 3852 3824 3803 3784 3778 3766 3741 3723 3718 3713 3707 3684 3675 3667 3614 3604 3595 3594 3543 3514 3473 3390 3366 3337 3306 3268 3263 3248 3243 3240 3237 3224 3223 3214 3205 3162 3157 3043 3024 3020 3011 3008 2959 2953 2950 2944 2943 2928 2920 2910 2901 2897 2876 2848 2844 2804 2783 2782 2761 2753 2681 2669 2614 2597 2590 2482 2480 2450 2421 2404 2389 2375 2362 2303 2281 2269 2239 2235 2211 2181 2162 2138 2108 2086 2073 2069 2062 2032 2026 2025 2012 2008 1996 1993 1989 1988 1983 1980 1925 1904 1892 1878 1868 1864 1846 1833 1826 1816 1789 1784 1781 1780 1763 1739 1736 1718 1697 1682 1680 1665 1658 1644 1616 1615 1604 1592 1572 1512 1502 1420 1410 1403 1375 1358 1357 1338 1320 1308 1293 1273 1265 1244 1229 1225 1216 1205 1178 1087 1073 1061 1059 1034 1015 1011 1003 999 994 965 914 904 902 880 878 827 819 792 774 757 756 737 732 729 725 719 702 637 630 609 604 601 598 582 561 524 496 466 429 392 391 389 374 346 345 307 277 271 270 257 220 208 191 189 168 150 145 143 122 97 86 83 74 66 53 52 46 37 )


PIDX running dumps all the SBacktrace::Summary so can look for unexpected bt to find the untagged consumption::

    PIDX=9993 ./U4RecorderTest.sh run 


* reemission will be a big source of untagged consumption, but is it the only one ?

::

    2022-07-11 17:47:27.997 INFO  [54262] [SEvt::addTag@804]  idx 9993 PIDX 9993 tag 5 flat 0.259223 evt.tag 0x10ac3820 tagr.slot 2
    ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength PIDX 9993 currentInteractionLength 44837.1652990 theNumberOfInteractionLengthLeft  1.3500648 value 60533.0742188
    2022-07-11 17:47:27.997 INFO  [54262] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor    3 idx 2558211 d    0.50091
    2022-07-11 17:47:27.997 INFO  [54262] [U4Random::flat@436] 
    SBacktrace::Summary
    U4Random::flat
    G4SteppingManager::DefinePhysicalStepLength
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 17:47:27.997 INFO  [54262] [SEvt::addTag@804]  idx 9993 PIDX 9993 tag 6 flat 0.500906 evt.tag 0x10ac3820 tagr.slot 3
    ShimG4OpAbsorption::PostStepGetPhysicalInteractionLength PIDX 9993 currentInteractionLength 117102.8434534 theNumberOfInteractionLengthLeft  0.6913363 value 80957.4531250
    2022-07-11 17:47:27.997 INFO  [54262] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor    4 idx 2558212 d    0.76245
    2022-07-11 17:47:27.997 INFO  [54262] [U4Random::flat@436] 
    SBacktrace::Summary



SRandom.h protocol base to U4Random allows SEvt::addTag to notice untagged consumption at the next SEvt::addTag
------------------------------------------------------------------------------------------------------------------

* this avoids having to look thru large numbers of stack traces to find unexpected ones as will 
  now assert at the addTag following untagged consumption 



DiMe : ChooseReflection DoReflection
----------------------------------------

::

    2022-07-11 18:24:30.668 INFO  [59788] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor   41 idx 2558249 d    0.34018
    2022-07-11 18:24:30.668 INFO  [59788] [U4Random::flat@436] 
    SBacktrace::Summary
    U4Random::flat
    G4SteppingManager::DefinePhysicalStepLength
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 18:24:30.668 INFO  [59788] [SEvt::addTag@805]  idx 9993 PIDX 9993 tag 6 flat 0.340178 evt.tag 0x10b7a820 tagr.slot 41
    ShimG4OpAbsorption::PostStepGetPhysicalInteractionLength PIDX 9993 currentInteractionLength 38562.9650658 theNumberOfInteractionLengthLeft  1.0782876 value 41581.9687500
    2022-07-11 18:24:30.668 INFO  [59788] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor   42 idx 2558250 d    0.39386
    2022-07-11 18:24:30.668 INFO  [59788] [U4Random::flat@436] 
    SBacktrace::Summary
    U4Random::flat
    InstrumentedG4OpBoundaryProcess::DielectricMetal
    InstrumentedG4OpBoundaryProcess::PostStepDoIt
    G4SteppingManager::InvokePSDIP
    G4SteppingManager::InvokePostStepDoItProcs
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 18:24:30.669 INFO  [59788] [SEvt::addTag@805]  idx 9993 PIDX 9993 tag 11 flat 0.393856 evt.tag 0x10b7a820 tagr.slot 42
    2022-07-11 18:24:30.669 INFO  [59788] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor   43 idx 2558251 d    0.73080
    2022-07-11 18:24:30.669 INFO  [59788] [U4Random::flat@436] 
    SBacktrace::Summary
    U4Random::flat
    InstrumentedG4OpBoundaryProcess::ChooseReflection
    InstrumentedG4OpBoundaryProcess::DielectricMetal
    InstrumentedG4OpBoundaryProcess::PostStepDoIt
    G4SteppingManager::InvokePSDIP
    G4SteppingManager::InvokePostStepDoItProcs
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 18:24:30.669 INFO  [59788] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor   44 idx 2558252 d    0.86766
    2022-07-11 18:24:30.669 INFO  [59788] [U4Random::flat@436] 
    SBacktrace::Summary
    U4Random::flat
    InstrumentedG4OpBoundaryProcess::DoReflection
    InstrumentedG4OpBoundaryProcess::DielectricMetal
    InstrumentedG4OpBoundaryProcess::PostStepDoIt
    G4SteppingManager::InvokePSDIP
    G4SteppingManager::InvokePostStepDoItProcs
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 18:24:30.669 INFO  [59788] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor   45 idx 2558253 d    0.84256
    2022-07-11 18:24:30.669 INFO  [59788] [U4Random::flat@436] 
    SBacktrace::Summary
    U4Random::flat
    InstrumentedG4OpBoundaryProcess::DoReflection
    InstrumentedG4OpBoundaryProcess::DielectricMetal
    InstrumentedG4OpBoundaryProcess::PostStepDoIt
    G4SteppingManager::InvokePSDIP
    G4SteppingManager::InvokePostStepDoItProcs
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 18:24:30.669 INFO  [59788] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor   46 idx 2558254 d    0.63358
    2022-07-11 18:24:30.669 INFO  [59788] [U4Random::flat@436] 
    SBacktrace::Summary
    U4Random::flat
    InstrumentedG4OpBoundaryProcess::DoReflection
    InstrumentedG4OpBoundaryProcess::DielectricMetal
    InstrumentedG4OpBoundaryProcess::PostStepDoIt
    G4SteppingManager::InvokePSDIP
    G4SteppingManager::InvokePostStepDoItProcs
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 18:24:30.669 INFO  [59788] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor   47 idx 2558255 d    0.45532
    2022-07-11 18:24:30.670 INFO  [59788] [U4Random::flat@436] 
    SBacktrace::Summary
    U4Random::flat
    InstrumentedG4OpBoundaryProcess::DoReflection
    InstrumentedG4OpBoundaryProcess::DielectricMetal
    InstrumentedG4OpBoundaryProcess::PostStepDoIt
    G4SteppingManager::InvokePSDIP
    G4SteppingManager::InvokePostStepDoItProcs
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 18:24:30.670 INFO  [59788] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor   48 idx 2558256 d    0.36513
    2022-07-11 18:24:30.670 INFO  [59788] [U4Random::flat@436] 
    SBacktrace::Summary
    U4Random::flat
    InstrumentedG4OpBoundaryProcess::DoReflection
    InstrumentedG4OpBoundaryProcess::DielectricMetal
    InstrumentedG4OpBoundaryProcess::PostStepDoIt
    G4SteppingManager::InvokePSDIP
    G4SteppingManager::InvokePostStepDoItProcs
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 18:24:30.670 INFO  [59788] [U4Random::flat@425]  SEvt::PIDX 9993 m_seq_index 9993 m_seq_nv  256 cursor   49 idx 2558257 d    0.70390
    2022-07-11 18:24:30.670 INFO  [59788] [U4Random::flat@436] 
    SBacktrace::Summary
    U4Random::flat
    G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength
    G4SteppingManager::DefinePhysicalStepLength
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 18:24:30.670 INFO  [59788] [SEvt::addTag@805]  idx 9993 PIDX 9993 tag 3 flat 0.703896 evt.tag 0x10b7a820 tagr.slot 43
    2022-07-11 18:24:30.670 ERROR [59788] [SEvt::addTag@825]  idx 9993 cursor_slot_match 0 flat 0.703896 tagr.slot 44 ( from SRandom  flat_prior 0.703896 flat_cursor 50  ) 
     MISMATCH MEANS ONE OR MORE PRIOR CONSUMPTIONS WERE NOT TAGGED 
    U4RecorderTest: /data/blyth/junotop/opticks/sysrap/SEvt.cc:839: void SEvt::addTag(unsigned int, float): Assertion cursor_slot_match


::

    u4
    BP=InstrumentedG4OpBoundaryProcess::DoReflection  PIDX=9993 ./uxs.sh dbg


Auto BP is lldb only (huh there is some gdb script somewhere too?)::

    (gdb) b InstrumentedG4OpBoundaryProcess::DoReflection
    Function "InstrumentedG4OpBoundaryProcess::DoReflection" not defined.
    Make breakpoint pending on future shared library load? (y or [n]) y
    Breakpoint 1 (InstrumentedG4OpBoundaryProcess::DoReflection) pending.
    (gdb) r

::

    epsilon:issues blyth$ t gdb_
    gdb_ () 
    { 
        : prepares and invokes gdb - sets up breakpoints based on BP envvar containing space delimited symbols;
        if [ -z "$BP" ]; then
            H="";
            B="";
            T="-ex r";
        else
            H="-ex \"set breakpoint pending on\"";
            B="";
            for bp in $BP;
            do
                B="$B -ex \"break $bp\" ";
            done;
            T="-ex \"info break\" -ex r";
        fi;
        local runline="gdb $H $B $T --args $* ";
        echo $runline;
        date;
        eval $runline;
        date
    }




::

     59 inline G4ThreeVector G4LambertianRand(const G4ThreeVector& normal)
     60 {
     61   G4ThreeVector vect;
     62   G4double ndotv;
     63   G4int count=0;
     64   const G4int max_trials = 1024;
     65 
     66   do
     67   {
     68     ++count;
     69     vect = G4RandomDirection();
     70     ndotv = normal * vect;
     71 
     72     if (ndotv < 0.0)
     73     {
     74       vect = -vect;
     75       ndotv = -ndotv;
     76     }
     77 
     78   } while (!(G4UniformRand() < ndotv) && (count < max_trials));
     79 
     80   return vect;
     81 }





Manual Tagging will take some effort : but its faster than auto tag and deficient backtrace means have to go manual anyhow
--------------------------------------------------------------------------------------------------------------------------------




::

    2022-07-10 22:05:00.704 INFO  [454628] [U4RecorderTest::GeneratePrimaries@134] ]
    2022-07-10 22:05:00.704 INFO  [454628] [U4Recorder::BeginOfEventAction@77] 
    2022-07-10 22:05:00.718 ERROR [454628] [U4Random::flat@431] 
    SBacktrace::Summary
    U4Random::flat
    G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength
    G4SteppingManager::DefinePhysicalStepLength
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-10 22:05:00.718 INFO  [454628] [U4Random::flat@438] U4Random_select - m_select->size 0

    Program received signal SIGINT, Interrupt.
    0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff7ba9090 in U4Random::flat (this=0x7fffffff5900) at /data/blyth/junotop/opticks/u4/U4Random.cc:441
    #2  0x00000000004267b2 in DsG4Scintillation::ResetNumberOfInteractionLengthLeft (this=0x1c80150) at /data/blyth/junotop/opticks/u4/tests/DsG4Scintillation.cc:114
    #3  0x00007ffff37e8554 in G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*) ()
       from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4processes.so
    #4  0x00007ffff44ae599 in G4SteppingManager::DefinePhysicalStepLength() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #5  0x00007ffff44acb48 in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #6  0x00007ffff44b8472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #7  0x00007ffff46ef389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #8  0x00007ffff498aa6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x00007ffff498853e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #10 0x0000000000413a97 in main (argc=1, argv=0x7fffffff6308) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:196
    (gdb) 



::

    (gdb) c
    Continuing.
    2022-07-10 22:21:45.729 ERROR [454628] [U4Random::flat@431] 
    SBacktrace::Summary
    U4Random::flat
    G4SteppingManager::DefinePhysicalStepLength
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-10 22:21:45.729 INFO  [454628] [U4Random::flat@438] U4Random_select - m_select->size 0

    Program received signal SIGINT, Interrupt.
    0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    (gdb) bt
    #0  0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff7ba9090 in U4Random::flat (this=0x7fffffff5900) at /data/blyth/junotop/opticks/u4/U4Random.cc:441
    #2  0x000000000041c536 in ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft (this=0x1cdbd00) at /data/blyth/junotop/opticks/u4/ShimG4OpRayleigh.h:48
    #3  0x000000000041c656 in ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength (this=0x1cdbd00, track=..., previousStepSize=0, condition=0xa4fef8)
        at /data/blyth/junotop/opticks/u4/ShimG4OpRayleigh.h:73
    #4  0x00007ffff44ae599 in G4SteppingManager::DefinePhysicalStepLength() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #5  0x00007ffff44acb48 in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #6  0x00007ffff44b8472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #7  0x00007ffff46ef389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #8  0x00007ffff498aa6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x00007ffff498853e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #10 0x0000000000413a97 in main (argc=1, argv=0x7fffffff6308) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:196
    (gdb) 








::

    022-07-10 23:23:15.135 INFO  [456353] [U4Recorder::BeginOfEventAction@77] 
    2022-07-10 23:23:15.149 ERROR [456353] [U4Random::flat@431] 
    SBacktrace::Summary
    U4Random::flat
    G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength
    G4SteppingManager::DefinePhysicalStepLength
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-10 23:23:15.150 INFO  [456353] [U4Random::flat@438] U4Random_select - m_select->size 0

    Program received signal SIGINT, Interrupt.
    0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff7b9fa0c in U4Random::flat (this=0x7fffffff57c0) at /data/blyth/junotop/opticks/u4/U4Random.cc:441
    #2  0x0000000000423a36 in DsG4Scintillation::ResetNumberOfInteractionLengthLeft (this=0x1c7ec00) at /data/blyth/junotop/opticks/u4/tests/DsG4Scintillation.cc:114
    #3  0x00007ffff37e8554 in G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*) ()
       from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4processes.so
    #4  0x00007ffff44ae599 in G4SteppingManager::DefinePhysicalStepLength() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #5  0x00007ffff44acb48 in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #6  0x00007ffff44b8472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #7  0x00007ffff46ef389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #8  0x00007ffff498aa6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x00007ffff498853e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #10 0x0000000000413951 in main (argc=1, argv=0x7fffffff61c8) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:200
    (gdb) 




Switch to manual tagging : occasionally missing a consumption
----------------------------------------------------------------------

::

    pre  U4StepPoint::DescPositionTime (     12.745     -7.456   -990.000      0.000)
     post U4StepPoint::DescPositionTime (     12.745     -7.456   1629.730     13.437)
    2022-07-11 01:16:39.641 ERROR [12213] [U4Random::check_cursor_vs_tagslot@487]  m_seq_index 9956 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9955 type 5 U4Step::Name UNEXPECTED cosThetaSign 0 spec LS///LS boundary 4294967295 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (    -10.187    -10.697   -990.000      0.000)
     post U4StepPoint::DescPositionTime (    -10.187    -10.697  17824.000     96.501)
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9955 type 1 U4Step::Name NOT_AT_BOUNDARY cosThetaSign 0 spec  boundary 0 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (    -10.187    -10.697  17824.00


    post U4StepPoint::DescPositionTime (      0.700    -38.031  48749.974    209.862)
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9920 type 1 U4Step::Name NOT_AT_BOUNDARY cosThetaSign 0 spec  boundary 0 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (      8.095     -1.973   -990.000      0.000)
     post U4StepPoint::DescPositionTime (      8.095     -1.973   2913.424     20.021)
    2022-07-11 01:16:39.699 ERROR [12213] [U4Random::check_cursor_vs_tagslot@487]  m_seq_index 9920 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9919 type 1 U4Step::Name NOT_AT_BOUNDARY cosThetaSign 0 spec  boundary 0 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (      1.424    -25.395   -990.000      0.000)
     post U4StepPoint::DescPositionTime (      1.424    -25.395  14759.101     80.780)
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9918 type 1 U4Step::Name NOT_AT_BOUNDARY cosThetaSign 0 spec  boundary 0 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (    -30.193     -8.788   -990.000      0.000)
     post U4StepPoint::DescPositionTime (    -30.193     -8.788  13908.053     76.415)
    2022-07-11 01:16:39.700 ERROR [12213] [U4Random::check_cursor_vs_tagslot@487]  m_seq_index 9918 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9917 type 5 U4Step::Name UNEXPECTED cosThetaSign 0 spec LS///LS boundary 4294967295 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (     11.630     12.447   -990.000      0.000)
     post U4StepPoint::DescPositionTime (     11.630     12.447  17824.000     96.501)
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9917 type 5 U4Step::Name UNEXPECTED cosThetaSign 0 spec LS///LS boundary 4294967295 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (     11.630     12.447  17824.000     96.501)
     post U4StepPoint::DescPositionTime (     11.


::

    PIDX=9920 ./U4RecorderTest.sh run
    PIDX=9918 ./U4RecorderTest.sh run
    PIDX=9956 ./U4RecorderTest.sh run




::


    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff7b96a70 in G4VPhysicalVolume::GetLogicalVolume (this=0x0) at /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/include/Geant4/G4VPhysicalVolume.icc:49
    49	  return flogical;
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff7b96a70 in G4VPhysicalVolume::GetLogicalVolume (this=0x0) at /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/include/Geant4/G4VPhysicalVolume.icc:49
    #1  0x00007ffff7b9ad40 in U4Step::Solid (point=0xa55040) at /data/blyth/junotop/opticks/u4/U4Step.h:436
    #2  0x00007ffff7b9a641 in U4Step::KludgePrimIdx (step=0xa54f10, type=1, idx=8374) at /data/blyth/junotop/opticks/u4/U4Step.h:192
    #3  0x00007ffff7b9a2c8 in U4Step::MockOpticksBoundaryIdentity (current_photon=..., step=0xa54f10, idx=8374) at /data/blyth/junotop/opticks/u4/U4Step.h:104
    #4  0x00007ffff7b9cf61 in U4Recorder::UserSteppingAction_Optical<InstrumentedG4OpBoundaryProcess> (this=0xabafe0, step=0xa54f10) at /data/blyth/junotop/opticks/u4/U4Recorder.cc:257
    #5  0x00007ffff7b9cbed in U4Recorder::UserSteppingAction<InstrumentedG4OpBoundaryProcess> (this=0xabafe0, step=0xa54f10) at /data/blyth/junotop/opticks/u4/U4Recorder.cc:83
    #6  0x0000000000413625 in U4RecorderTest::UserSteppingAction (this=0x7fffffff54b0, step=0xa54f10) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:148
    #7  0x00007ffff44ace1d in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #8  0x00007ffff44b8472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #9  0x00007ffff46ef389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #10 0x00007ffff498aa6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #11 0x00007ffff498853e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #12 0x0000000000413b21 in main (argc=1, argv=0x7fffffff6438) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:200
    (gdb) 




One untagged consumption looks to be at the end of the history : was DiMe
----------------------------------------------------------------------------

::

    u4t ; PIDX=9920 ./U4RecorderTest.sh run 


    2022-07-11 02:52:24.585 INFO  [26174] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   48 idx 2539568 d    0.69924
    2022-07-11 02:52:24.585 INFO  [26174] [SEvt::addTag@804]  idx 9920 PIDX 9920 tag 4 flat 0.69924 evt.tag 0x12265820 tagr.slot 48
    2022-07-11 02:52:24.585 INFO  [26174] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   49 idx 2539569 d    0.49888
    2022-07-11 02:52:24.585 INFO  [26174] [SEvt::addTag@804]  idx 9920 PIDX 9920 tag 5 flat 0.498883 evt.tag 0x12265820 tagr.slot 49
    ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength PIDX 9920 currentInteractionLength 1000000.0000000 theNumberOfInteractionLengthLeft  0.6953840 value 695383.9375000
    2022-07-11 02:52:24.585 INFO  [26174] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   50 idx 2539570 d    0.95529
    2022-07-11 02:52:24.585 INFO  [26174] [SEvt::addTag@804]  idx 9920 PIDX 9920 tag 6 flat 0.955289 evt.tag 0x12265820 tagr.slot 50
    ShimG4OpAbsorption::PostStepGetPhysicalInteractionLength PIDX 9920 currentInteractionLength 1987.1562558 theNumberOfInteractionLengthLeft  0.0457416 value 90.8957291
    2022-07-11 02:52:24.585 INFO  [26174] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   51 idx 2539571 d    0.51655
    2022-07-11 02:52:24.585 INFO  [26174] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   52 idx 2539572 d    0.98802
    2022-07-11 02:52:24.585 INFO  [26174] [SEvt::addTag@804]  idx 9920 PIDX 9920 tag 9 flat 0.988018 evt.tag 0x12265820 tagr.slot 51
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9920 type 2 U4Step::Name MOTHER_TO_CHILD cosThetaSign -1 spec Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum boundary 32 kludge_prim_idx -1 kludge_prim_idx_ 65535
     pre  U4StepPoint::DescPositionTime (  -9339.653   7475.979  15211.020    107.270)
     post U4StepPoint::DescPositionTime (  -9343.542   7480.095  15216.312    107.309)
    2022-07-11 02:52:24.585 ERROR [26174] [U4Random::check_cursor_vs_tagslot@489]  m_seq_index 9920 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    2022-07-11 02:52:24.585 INFO  [26174] [SEvt::beginPhoton@535]  idx 9919


     post U4StepPoint::DescPositionTime (  -3694.903  -2705.972  18808.381    109.800)
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9918 type 2 U4Step::Name MOTHER_TO_CHILD cosThetaSign -1 spec Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum boundary 35 kludge_prim_idx -1 kludge_prim_idx_ 65535
     pre  U4StepPoint::DescPositionTime (  -3694.903  -2705.972  18808.381    109.800)
     post U4StepPoint::DescPositionTime (  -3699.417  -2711.157  18816.101    109.852)
    2022-07-11 02:52:24.587 ERROR [26174] [U4Random::check_cursor_vs_tagslot@489]  m_seq_index 9918 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    2022-07-11 02:52:24.587 INFO  [26174] [SEvt::beginPhoton@535]  idx 9917


::

    2022-07-11 03:07:52.416 INFO  [26866] [SEvt::addTag@804]  idx 9920 PIDX 9920 tag 6 flat 0.955289 evt.tag 0x12052820 tagr.slot 50
    ShimG4OpAbsorption::PostStepGetPhysicalInteractionLength PIDX 9920 currentInteractionLength 1987.1562558 theNumberOfInteractionLengthLeft  0.0457416 value 90.8957291
    2022-07-11 03:07:52.416 INFO  [26866] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   51 idx 2539571 d    0.51655
    2022-07-11 03:07:52.416 INFO  [26866] [U4Random::flat@434] 
    SBacktrace::Summary
    U4Random::flat
    InstrumentedG4OpBoundaryProcess::DielectricMetal
    InstrumentedG4OpBoundaryProcess::PostStepDoIt
    G4SteppingManager::InvokePSDIP
    G4SteppingManager::InvokePostStepDoItProcs
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 03:07:52.416 INFO  [26866] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   52 idx 2539572 d    0.98802
    2022-07-11 03:07:52.416 INFO  [26866] [U4Random::flat@434] 
    SBacktrace::Summary
    U4Random::flat
    InstrumentedG4OpBoundaryProcess::G4BooleanRand_theEfficiency
    InstrumentedG4OpBoundaryProcess::DoAbsorption
    InstrumentedG4OpBoundaryProcess::DielectricMetal
    InstrumentedG4OpBoundaryProcess::PostStepDoIt
    G4SteppingManager::InvokePSDIP
    G4SteppingManager::InvokePostStepDoItProcs
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-11 03:07:52.416 INFO  [26866] [SEvt::addTag@804]  idx 9920 PIDX 9920 tag 9 flat 0.988018 evt.tag 0x12052820 tagr.slot 51
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9920 type 2 U4Step::Name MOTHER_TO_CHILD cosThetaSign -1 spec Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum boundary 32 kludge_prim_idx -1 kludge_prim_idx_ 65535
     pre  U4StepPoint::DescPositionTime (  -9339.653   7475.979  15211.020    107.270)
     post U4StepPoint::DescPositionTime (  -9343.542   7480.095  15216.312    107.309)
    2022-07-11 03:07:52.417 ERROR [26866] [U4Random::check_cursor_vs_tagslot@494]  m_seq_index 9920 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    2022-07-11 03:07:52.417 INFO  [26866] [SEvt::beginPhoton@535]  idx 9919
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9919 type 1 U4Step::Name NOT_AT_BOUNDARY cosThetaSign 0 s





Huh fixing that one appears to get all consumption tagged : thats unbelievable : expecting raft of reemission issues ?
--------------------------------------------------------------------------------------------------------------------------

* unless reemission is disabled ? it isnt : suspect material lacks necessary props

::

    In [6]: wq = np.where( a.seq[:,0] != b.seq[:,0] )[0]
    In [7]: len(wq)
    Out[7]: 4988



As are using input photons all DsG4Scintillation::PostStepDoIt will be reemission::

    BP=DsG4Scintillation::PostStepDoIt ./u4s.sh dbg


::

    (gdb) p flagReemission
    $5 = false
    (gdb) p aTrack.GetTrackStatus()
    $6 = fAlive


::

     282     G4String pname="";
     283     G4ThreeVector vertpos;
     284     //G4double vertenergy=0.0;
     285     //G4double reem_d=0.0;
     286     G4bool flagReemission= false;
     287     //DsPhotonTrackInfo* reemittedTI=0;
     288     if (aTrack.GetDefinition() == G4OpticalPhoton::OpticalPhoton()) {
     289         G4Track *track=aStep.GetTrack();
     290         //G4CompositeTrackInfo* composite=dynamic_cast<G4CompositeTrackInfo*>(track->GetUserInformation());
     291         //reemittedTI = composite?dynamic_cast<DsPhotonTrackInfo*>( composite->GetPhotonTrackInfo() ):0;
     292 
     293         const G4VProcess* process = track->GetCreatorProcess();
     294         if(process) pname = process->GetProcessName();
     295 
     296         if (verboseLevel > 0) {
     297           G4cout<<"Optical photon. Process name is " << pname<<G4endl;
     298         }
     299         if(doBothProcess) {
     300             flagReemission= doReemission
     301                 && aTrack.GetTrackStatus() == fStopAndKill
     302                 && aStep.GetPostStepPoint()->GetStepStatus() != fGeomBoundary;
     303         }
     304         else{
     305             flagReemission= doReemission
     306                 && aTrack.GetTrackStatus() == fStopAndKill
     307                 && aStep.GetPostStepPoint()->GetStepStatus() != fGeomBoundary
     308                 && pname=="Cerenkov";
     309         }
     310         if(verboseLevel > 0) {
     311             G4cout<<"flag of Reemission is "<<flagReemission<<"!!"<<G4endl;
     312         }
     313         if (!flagReemission) {
     314             return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
     315         }
     316     }


::

    (gdb) c
    Continuing.

    Breakpoint 9, DsG4Scintillation::PostStepDoIt (this=0x1c80c00, aTrack=..., aStep=...) at /data/blyth/junotop/opticks/u4/tests/DsG4Scintillation.cc:360
    360	    if (!Fast_Intensity && !Slow_Intensity )
    (gdb) p Fast_Intensity
    $15 = (const G4MaterialPropertyVector *) 0x0
    (gdb) p Slow_Intensity
    $16 = (const G4MaterialPropertyVector *) 0x0
    (gdb) p Reemission_Prob
    $17 = (const G4MaterialPropertyVector *) 0x733c80
    (gdb) 


Never getting to 370 because lack FASTCOMPONENT SLOWCOMPONENT::

     350     const G4MaterialPropertyVector* Fast_Intensity =
     351         aMaterialPropertiesTable->GetProperty("FASTCOMPONENT");
     352     const G4MaterialPropertyVector* Slow_Intensity =
     353         aMaterialPropertiesTable->GetProperty("SLOWCOMPONENT");
     354     const G4MaterialPropertyVector* Reemission_Prob =
     355         aMaterialPropertiesTable->GetProperty("REEMISSIONPROB");
     356     if (verboseLevel > 0 ) {
     357       G4cout << " MaterialPropertyVectors: Fast_Intensity " << Fast_Intensity
     358              << " Slow_Intensity " << Slow_Intensity << " Reemission_Prob " << Reemission_Prob << G4endl;
     359     }
     360     if (!Fast_Intensity && !Slow_Intensity )
     361         return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
     362 
     363     //-------------find the type of particle------------------------------//
     364     /*
     365         Find the particle type and register the scintillation time constant corresponding.
     366         We save the yield ratio and time constant in the form of G4PhysicVector. In this kind of G4PhysicVector, we interprete Energy as scintillation time and interprete Value as the yield ratio.
     367 
     368     */
     369     G4MaterialPropertyVector* Ratio_timeconstant = 0 ;
     370     if (aParticleName == "opticalphoton") {
     371       Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("OpticalCONSTANT");
     372     }


Comment U4Material::LoadBnd to use original materials, as the props grabbed from the bnd lack FASTCOMPONENT SLOWCOMPONENT::

    176 int main(int argc, char** argv)
    177 {
    178     OPTICKS_LOG(argc, argv);
    179 
    180     //U4Material::LoadOri();  // currently needs  "source ./IDPath_override.sh" to find _ori materials
    181     //U4Material::LoadBnd();   // "back" creation of G4 material properties from the Opticks bnd.npy obtained from SSim::Load 
    182 



YEP : after using original materials from GDML get the expected assert
-------------------------------------------------------------------------

     

::

    u4
    ./u4s.sh 


    DsG4Scintillation::DsG4Scintillation level 0 verboseLevel 0
    2022-07-11 22:01:58.369 INFO  [74874] [U4Recorder::BeginOfRunAction@79] 
    2022-07-11 22:01:58.370 INFO  [74874] [U4RecorderTest::GeneratePrimaries@129] [ fPrimaryMode I
    2022-07-11 22:01:58.373 INFO  [74874] [U4RecorderTest::GeneratePrimaries@137] ]
    2022-07-11 22:01:58.373 INFO  [74874] [U4Recorder::BeginOfEventAction@81] 
    2022-07-11 22:01:58.390 ERROR [74874] [U4Random::check_cursor_vs_tagslot@503]  m_seq_index 9979 cursor 9 slot 8 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    2022-07-11 22:01:58.390 ERROR [74874] [U4Random::check_cursor_vs_tagslot@503]  m_seq_index 9978 cursor 11 slot 4 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    2022-07-11 22:01:58.390 ERROR [74874] [SEvt::addTag@825]  idx 9978 cursor_slot_match 0 flat 0.0625658 tagr.slot 5 ( from SRandom  flat_prior 0.0625658 flat_cursor 12  ) 
     MISMATCH MEANS ONE OR MORE PRIOR CONSUMPTIONS WERE NOT TAGGED 
    U4RecorderTest: /data/blyth/junotop/opticks/sysrap/SEvt.cc:839: void SEvt::addTag(unsigned int, float): Assertion `cursor_slot_match' failed.
    ./u4s.sh: line 161: 74874 Aborted                 (core dumped) $bin
    === ./u4s.sh : run error
    N[blyth@localhost u4]$ 



Avoiding IDPath_override allows BoxOfScintillator to give the error on laptop : used this to add the reemission tagging
--------------------------------------------------------------------------------------------------------------------------


Simple way to check all consumption and manually look at the stacks in debugger, so can add the consumption tagging::

    PIDX=9999 BP=U4Random::flat ./u4s.sh dbg

    BP=G4VParticleChange::SetNumberOfSecondaries 


::

    (lldb) f 1
    frame #1: 0x00000001000402c7 U4RecorderTest`DsG4Scintillation::PostStepDoIt(this=0x00000001090e7170, aTrack=0x00000001105c3a30, aStep=0x000000010906c9f0) at DsG4Scintillation.cc:404
       401 	            return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
       402 	        G4double p_reemission=
       403 	            Reemission_Prob->Value(aTrack.GetKineticEnergy());
    -> 404 	        if (G4UniformRand() >= p_reemission)
       405 	            return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
       406 	        NumTracks= 1;
       407 	        weight= aTrack.GetWeight();
    (lldb) 




Now looks like all consumption is tagged in GDML geometry
-------------------------------------------------------------


::

    u4
    ./u4s.sh 

    G4GDML: Reading '/home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/origin_CGDMLKludge.gdml' done!
    DsG4Scintillation::DsG4Scintillation level 0 verboseLevel 0
    2022-07-11 22:47:36.523 INFO  [76254] [U4Recorder::BeginOfRunAction@79] 
    2022-07-11 22:47:36.524 INFO  [76254] [U4RecorderTest::GeneratePrimaries@129] [ fPrimaryMode I
    2022-07-11 22:47:36.527 INFO  [76254] [U4RecorderTest::GeneratePrimaries@137] ]
    2022-07-11 22:47:36.527 INFO  [76254] [U4Recorder::BeginOfEventAction@81] 
    2022-07-11 22:47:36.607 INFO  [76254] [U4Recorder::PreUserTrackingAction_Optical@152]  label.id 9000
    2022-07-11 22:47:36.677 INFO  [76254] [U4Recorder::PreUserTrackingAction_Optical@152]  label.id 8000
    2022-07-11 22:47:36.736 INFO  [76254] [U4Recorder::PreUserTrackingAction_Optical@152]  label.id 7000
    2022-07-11 22:47:36.794 INFO  [76254] [U4Recorder::PreUserTrackingAction_Optical@152]  label.id 6000
    2022-07-11 22:47:36.851 INFO  [76254] [U4Recorder::PreUserTrackingAction_Optical@152]  label.id 5000
    2022-07-11 22:47:36.907 INFO  [76254] [U4Recorder::PreUserTrackingAction_Optical@152]  label.id 4000
    2022-07-11 22:47:36.962 INFO  [76254] [U4Recorder::PreUserTrackingAction_Optical@152]  label.id 3000
    2022-07-11 22:47:37.016 INFO  [76254] [U4Recorder::PreUserTrackingAction_Optical@152]  label.id 2000
    2022-07-11 22:47:37.073 INFO  [76254] [U4Recorder::PreUserTrackingAction_Optical@152]  label.id 1000
    2022-07-11 22:47:37.129 INFO  [76254] [U4Recorder::PreUserTrackingAction_Optical@152]  label.id 0
    2022-07-11 22:47:37.129 INFO  [76254] [U4Recorder::EndOfEventAction@82] 
    2022-07-11 22:47:37.129 INFO  [76254] [U4Recorder::EndOfRunAction@80] 
    2022-07-11 22:47:37.129 INFO  [76254] [main@202] /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/J000/ALL
    U4Random::saveProblemIdx m_problem_idx.size 0 ()
    2022-07-11 22:47:37.187 INFO  [76254] [main@207] /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/J000/ALL
    === ./u4s.sh : logdir /tmp/blyth/opticks/U4RecorderTest



