fix_unexpected_empty_SEvt_dir_creation
=======================================

ISSUE : Unexpected empty directory created
--------------------------------------------------

While running as gitlab-runner user noticed creation of empty SEvt dirs
while array saving was not enabled for gun running.


::

    A[gitlab-runner@localhost gitlab-runner]$ find log
    log
    log/oj
    log/oj/gun
    log/oj/gun/gun
    log/oj/gun/gun/2026-04-09-1810
    log/oj/gun/gun/2026-04-09-1810/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1
    log/oj/gun/gun/2026-04-09-1810/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1_run.log
    log/oj/gun/gun/2026-04-09-1810/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1_run.env
    log/oj/gun/gun/2026-04-09-1810/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1/user_output.root
    log/oj/gun/gun/2026-04-09-1810/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1/python3.11.log
    log/oj/gun/gun/2026-04-09-1810/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1/SProf.txt
    log/oj/gun/gun/2026-04-09-1810/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1/run.npy
    log/oj/gun/gun/2026-04-09-1810/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1/run_meta.txt
    log/oj/gun/gun/2026-04-09-1810/OJ_LOCAL_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV_2026Apr07_ok1_hitlitemerged_seed42_evtmax1/output.root

    A[gitlab-runner@localhost gitlab-runner]$ find opticks    ## NOT EXPECTING THIS EMPTY DIRECTORY TO BE CREATED
    opticks
    opticks/GEOM
    opticks/GEOM/J26_1_1_opticks_Debug
    opticks/GEOM/J26_1_1_opticks_Debug/python3.11
    opticks/GEOM/J26_1_1_opticks_Debug/python3.11/ALL0_no_opticks_event_name
    A[gitlab-runner@localhost gitlab-runner]$ 



* So there is an inadvertant directory creation issue.


::

    A[gitlab-runner@localhost gitlab-runner]$ pwd
    /tmp/gitlab-runner

    A[gitlab-runner@localhost gitlab-runner]$ l opticks/GEOM/J26_1_1_opticks_Debug/python3.11/ALL0_no_opticks_event_name/
    total 0
    0 drwxr-xr-x. 2 gitlab-runner gitlab-runner  6 Apr  9 16:08 .
    0 drwxr-xr-x. 3 gitlab-runner gitlab-runner 40 Apr  9 16:08 ..



is slic count wrong ? NO, added debug - it really is zero
---------------------------------------------------------

::

    4669     // 7. count *slic* items within save_fold, when more than zero proceed to save to standard dir
    4670 
    4671     int slic = save_fold->_save_local_item_count();
    4672     if( slic > 0 )
    4673     {
    4674         const char* dir = getDir(dir_);   // THIS CREATES DIRECTORY
    4675         LOG_IF(info, MINIMAL||SIMTRACE) << dir << " [" << save_comp << "]"  ;
    4676         LOG(LEVEL) << descSaveDir(dir_) ;
    4677 
    4678         LOG(LEVEL) << "[ save_fold.save " << dir ;
    4679         save_fold->save(dir);
    4680         LOG(LEVEL) << "] save_fold.save " << dir ;
    4681 
    4682         int num_save_comp = SEventConfig::NumSaveComp();
    4683         if(num_save_comp > 0 ) saveFrame(dir);
    4684         // could add frame to the fold ?
    4685         // for now just restrict to saving frame when other components are saved
    4686     }
    4687     else
    4688     {
    4689         LOG(LEVEL) << "SKIP SAVE AS NPFold::_save_local_item_count zero " ;
    4690     }
    4691 


Seems not. The creation not happening there::

    2026-04-09 17:39:00.686 INFO  [3834983] [SEvt::save@4673]  slic 0 slicd [NPFold::_save_local_item_count_desc
     slic 0 kk.size 0 ff.size 0]NPFold::_save_local_item_count_desc

    2026-04-09 17:39:00.686 INFO  [3834983] [SEvt::save@4693] SKIP SAVE AS NPFold::_save_local_item_count zero 

Try catching mkdir::

    A[gitlab-runner@localhost oj]$ export TESTSCRIPT=gun/gun.sh TEST=hitlitemerged ENVSET_NAME=OJ_LOCAL ENVSET_VERBOSE=1 SEvt__MINIMAL=1 SEvt=INFO CATCH_MKDIR=1 ; $TESTSCRIPT dbg

cuInit::

    2026-04-09 17:48:55.185 INFO  [3841610] [SEvt::CreateOrReuse@1505]  integrationMode 1
    [New Thread 0x7fffbcdff640 (LWP 3841739)]

    Thread 1 "python" hit Catchpoint 1 (call to syscall mkdir), 0x00007ffff74fdc3b in mkdir () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install nvidia-driver-cuda-libs-580.82.07-1.el9.x86_64
    (gdb) bt
    #0  0x00007ffff74fdc3b in mkdir () from /lib64/libc.so.6
    #1  0x00007fffb501c376 in ?? () from /lib64/libcuda.so.1
    #2  0x00007fffb5d55115 in ?? () from /lib64/libcuda.so.1
    #3  0x00007fffb50d4fbb in ?? () from /lib64/libcuda.so.1
    #4  0x00007fffb5114152 in ?? () from /lib64/libcuda.so.1
    #5  0x00007fffb5105930 in cuInit () from /lib64/libcuda.so.1
    #6  0x00007fffbe2bef91 in libcudart_static_aa4a6bcb5fce58be20d542d9b467101e0a9360a5 () from /data1/blyth/local/opticks_Debug/lib64/libSysRap.so
    #7  0x00007fffbe2bf0c0 in libcudart_static_0bf7336e71b5df655f7fe4ef2dea52179e6fcf82 () from /data1/blyth/local/opticks_Debug/lib64/libSysRap.so
    #8  0x00007ffff748f3a8 in __pthread_once_slow () from /lib64/libc.so.6
    #9  0x00007fffbe311809 in libcudart_static_5887a27cefafb4cd438bdc166b0a6f874b079d4b () from /data1/blyth/local/opticks_Debug/lib64/libSysRap.so
    #10 0x00007fffbe2b287f in libcudart_static_418eebf4e9b7463362b8385a31d08da131d0ea88 () from /data1/blyth/local/opticks_Debug/lib64/libSysRap.so
    #11 0x00007fffbe2dcdda in cudaGetDeviceCount () from /data1/blyth/local/opticks_Debug/lib64/libSysRap.so
    #12 0x00007fffbe15a507 in sdevice::DeviceCount () at /home/blyth/opticks/sysrap/sdevice.h:154
    #13 0x00007fffbe15a52b in sdevice::Collect (devices=..., ordinal_from_index=true) at /home/blyth/opticks/sysrap/sdevice.h:180
    #14 0x00007fffbe15aa29 in sdevice::Visible (visible=...) at /home/blyth/opticks/sysrap/sdevice.h:316
    #15 0x00007fffbe15b73b in scontext::initPersist (this=0xc426320) at /home/blyth/opticks/sysrap/scontext.h:105
    #16 0x00007fffbe15b6de in scontext::init (this=0xc426320) at /home/blyth/opticks/sysrap/scontext.h:70
    #17 0x00007fffbe15b693 in scontext::scontext (this=0xc426320) at /home/blyth/opticks/sysrap/scontext.h:66
    #18 0x00007fffbe155aca in SEventConfig::Initialize_Meta () at /home/blyth/opticks/sysrap/SEventConfig.cc:1203
    #19 0x00007fffbe155c57 in SEventConfig::Initialize () at /home/blyth/opticks/sysrap/SEventConfig.cc:1265
    #20 0x00007fffbe166a54 in SEvt::SEvt (this=0xc4299f0) at /home/blyth/opticks/sysrap/SEvt.cc:181
    #21 0x00007fffbe16b6ca in SEvt::Create (ins=0) at /home/blyth/opticks/sysrap/SEvt.cc:1411
    #22 0x00007fffbe16ba35 in SEvt::CreateOrReuse (idx=0) at /home/blyth/opticks/sysrap/SEvt.cc:1469
    #23 0x00007fffbe16bcc3 in SEvt::CreateOrReuse () at /home/blyth/opticks/sysrap/SEvt.cc:1513
    #24 0x00007fffc0c7cc31 in G4CXOpticks::SetGeometry_JUNO (world=0xb1173b0, sd=0xa734870, jpmt=0xc426830, jlut=0xc4286b0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:109



After wading quite a few mkdir from python, cuda, optix find my ONE::

    2026-04-09 17:56:10.645 INFO  [3841610] [SEvt::save@4693] SKIP SAVE AS NPFold::_save_local_item_count zero
    2026-04-09 17:56:10.645 INFO  [3841610] [SEvt::clear_output@2225] ]
    2026-04-09 17:56:10.646 INFO  [3841610] [SEvt::setNumPhoton@2629]  evt->num_photon 0 evt->num_tag 0 evt->num_flat 0

    Thread 1 "python" hit Catchpoint 1 (call to syscall mkdir), 0x00007ffff74fdc3b in mkdir () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff74fdc3b in mkdir () from /lib64/libc.so.6
    #1  0x00007fffbe0a8f32 in sdirectory::MakeDirs (dirpath_=0xcd42f5b0 "/tmp/gitlab-runner/opticks/GEOM/J26_1_1_opticks_Debug/python3.11/ALL0_no_opticks_event_name", mode_=0) at /home/blyth/opticks/sysrap/sdirectory.h:62
    #2  0x00007fffbe178802 in SEvt::RunDir (base_=0x0) at /home/blyth/opticks/sysrap/SEvt.cc:4301
    #3  0x00007fffbe16cae8 in SEvt::SaveRunMeta (base=0x0) at /home/blyth/opticks/sysrap/SEvt.cc:1813
    #4  0x00007fffbe16d6d1 in SEvt::endOfEvent (this=0xc4299f0, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1998
    #5  0x00007fffbea7563f in QSim::reset (this=0x311a0000, eventID=0) at /home/blyth/opticks/qudarap/QSim.cc:840
    #6  0x00007fffc08e8027 in CSGOptiX::reset (this=0x311b6810, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:814
    #7  0x00007fffc0c7faa3 in G4CXOpticks::reset (this=0xd0f5630, eventID=0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:505
    #8  0x00007fffbcf80f0c in junoSD_PMT_v2_Opticks::EndOfEvent_reset (this=0xb12c4a0, eventID=0) at /home/blyth/junosw/Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc:666
    #9  0x00007fffbcf6eba2 in junoSD_PMT_v2::EndOfEvent (this=0xa734870, HCE=0xbb3e5da0) at /home/blyth/junosw/Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc:1173
    #10 0x00007fffc63410fa in G4SDStructure::Terminate(G4HCofThisEvent*) () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4digits_hits.so
    #11 0x00007fffc69d5a80 in G4EventManager::DoProcessing(G4Event*) () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #12 0x00007fffe314468e in G4SvcRunManager::SimulateEvent (this=0x65f73b0, i_event=0) at /home/blyth/junosw/Simulation/DetSimV2/G4Svc/src/G4SvcRunManager.cc:29
    #13 0x00007fffcfde5d2e in DetSimAlg::execute (this=0x65fb300) at /home/blyth/junosw/Simulation/DetSimV2/DetSimAlg/src/DetSimAlg.cc:112
    #14 0x00007fffd4b4d4b1 in Task::execute() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/sniper/InstallArea/lib64/libSniperKernel.so
    #15 0x00007fffd4b51bbd in TaskWatchDog::run() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/sniper/InstallArea/lib64/libSniperKernel.so
    #16 0x00007fffd4b4d054 in Task::run() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/sniper/InstallArea/lib64/libSniperKernel.so
    #17 0x00007fffcc8d38d3 in boost::python::objects::caller_py_function_impl<boost::python::detail::caller<bool (Task::*)(), boost::python::default_call_policies, boost::mpl::vector2<bool, Task&> > >::operator()(_object*, _object*) ()



BINGO : RunDir is always being called
--------------------------------------

::


    1811 void SEvt::SaveRunMeta(const char* base)
    1812 {
    1813     const char* dir = RunDir(base);
    1814     const char* name = "run.npy" ;
    1815 
    1816     bool is_save_nothing = IsSaveNothing();
    1817 
    1818     LOG_IF(info, RUNMETA)
    1819         << " [" << SEvt__RUNMETA << "]"
    1820         << " is_save_nothing " << ( is_save_nothing ? "YES" : "NO " )
    1821         << " base " << ( base ? base : "-" )
    1822         << " dir " << ( dir ? dir : "-" )
    1823         << " name " << name
    1824         << " SAVE_RUNDIR " << ( SAVE_RUNDIR ? "YES" : "NO " )
    1825         ;
    1826 
    1827     if(is_save_nothing) return ;
    1828 
    1829     if(SAVE_RUNDIR)
    1830     {
    1831         RUN_META->save(dir, name) ;
    1832     }
    1833     else
    1834     {
    1835         RUN_META->save(name) ;
    1836     }
    1837 }


    4286 /**
    4287 SEvt::RunDir
    4288 --------------
    4289 
    4290 Directory without event index, used for run level metadata.
    4291 
    4292 **/
    4293 
    4294 const char* SEvt::RunDir( const char* base_ )  // static
    4295 {
    4296     const char* base = DefaultBase(base_);
    4297     const char* reldir = SEventConfig::EventReldir() ;
    4298     const char* dir = spath::Resolve(base, reldir );
    4299 
    4300     bool is_save_nothing = IsSaveNothing();
    4301     if(!is_save_nothing) sdirectory::MakeDirs(dir,0);
    4302     return dir ;
    4303 }

    1769 /**
    1770 SEvt::IsSaveNothing
    1771 --------------------
    1772 
    1773 When using OPTICKS_EVENT_MODE of Minimal or Nothing
    1774 the saving of run metadata and directory creation
    1775 can be disabled with::
    1776 
    1777     export SEvt__SAVE_NOTHING=1
    1778 
    1779 **/
    1780 
    1781 bool SEvt::IsSaveNothing() // static
    1782 {
    1783     return SEventConfig::IsMinimalOrNothing() && SAVE_NOTHING ;
    1784 }
    1785 


::

    (gdb) p SEventConfig::IsMinimalOrNothing()
    $1 = true

    (gdb) p SEvt::IsSaveNothing()
    $3 = false

    (gdb) p SAVE_NOTHING
    $4 = false


Even without IsSaveNothing the wrong directory is being created.


