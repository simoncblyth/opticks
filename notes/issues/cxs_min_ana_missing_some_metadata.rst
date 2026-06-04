cxs_min_ana_missing_some_metadata
===================================

::

    ~/o/cxs_min.sh
    ~/o/cxs_min.sh ana



FIXED::

    AttributeError: No attribute SEvt__BeginOfRun 
    > /home/blyth/opticks/ana/npmeta.py(147)__getattr__()
        145     def __getattr__(self, k):
        146         if not k in self.d:
    --> 147              raise AttributeError("No attribute %s " % k)
        148         return self.find(k)
        149 

    ipdb> u
    > /home/blyth/opticks/sysrap/sevt.py(149)init_run()
        147         prof2stamp_ = lambda prof:prof.split(",")[0]
        148 
    --> 149         rr_ = [prof2stamp_(run_meta.SEvt__BeginOfRun),
        150                prof2stamp_(run_meta.SEvt__EndOfRun  )] if has_run_meta else [0,0]
        151 

    ipdb> p run_meta
    _init_stamp:1780479427198020
    _init_stamp_Fmt:2026-06-03T17:37:07..198
    source:SEvt__Init_RUN_META
    creator:CSGOptiXSMTest
    uname:Linux localhost.localdomain 5.14.0-570.37.1.el9_6.x86_64 #1 SMP PREEMPT_DYNAMIC Tue Aug 26 10:33:12 EDT 2025 x86_64 x86_64 x86_64 GNU/Linux
    CUDA_VISIBLE_DEVICES:0
    HOME:/home/blyth
    USER:blyth
    SCRIPT:cxs_min.sh
    PWD:/data1/blyth/tmp/GEOM/J26_1_1_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan
    TEST:medium_scan
    VERSION:1
    GEOM:J26_1_1_opticks_Debug
    ...



::

    1634 void SEvt::BeginOfRun()
    1635 {
    1636     SProf::Add("SEvt__BeginOfRun");
    1637     SProf::Write();
    1638 }
    1639 
    1640 
    1641 
    1642 
    1643 void SEvt::EndOfRun()
    1644 {
    1645     SProf::Add("SEvt__EndOfRun");
    1646     SProf::Write();
    1647 }
    1648 

    1852 void SEvt::beginOfEvent(int eventID)
    1853 {
    1854     if(isFirstEvtInstance() && eventID == 0) BeginOfRun() ;
    1855     if(eventID == 0) SProf::Add( isEGPU() ? "SEvt__beginOfEvent_FIRST_EGPU" : "SEvt__beginOfEvent_FIRST_ECPU" ) ;
    1856 
    1857     setStage(SEvt__beginOfEvent);
    1858     sprof::Stamp(p_SEvt__beginOfEvent_0);
    1859 



Is it being scrubbed or not called ?::

    TEST=debug BP=SEvt::BeginOfRun ~/o/cxs_min.sh


Looks like being scrubbed::

    hread 1 "CSGOptiXSMTest" hit Breakpoint 1, SEvt::BeginOfRun () at /home/blyth/opticks/sysrap/SEvt.cc:1636
    1636	    SProf::Add("SEvt__BeginOfRun");
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64 nvidia-driver-common-610.43.02-1.el9.x86_64 nvidia-driver-cuda-libs-610.43.02-1.el9.x86_64 nvidia-driver-libs-610.43.02-1.el9.x86_64 openssl-libs-3.5.1-7.el9_7.x86_64
    (gdb) bt
    #0  SEvt::BeginOfRun () at /home/blyth/opticks/sysrap/SEvt.cc:1636
    #1  0x00007ffff597d60f in SEvt::beginOfEvent (this=0x470d40, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1854
    #2  0x00007ffff6283416 in QSim::simulate (this=0x17f52b50, eventID=0, reset_=true) at /home/blyth/opticks/qudarap/QSim.cc:453
    #3  0x00007ffff7e5d78a in CSGOptiX::simulate (this=0x17f669d0, eventID=0, reset=true) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:732
    #4  0x00007ffff7e5a2f0 in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:191
    #5  0x0000000000404b15 in main (argc=1, argv=0x7fffffffb388) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) f 1
    #1  0x00007ffff597d60f in SEvt::beginOfEvent (this=0x470d40, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1854
    1854	    if(isFirstEvtInstance() && eventID == 0) BeginOfRun() ;
    (gdb) p isFirstEvtInstance()
    $1 = true
    (gdb) 

    (gdb) p SProf::NAME
    $2 = std::vector of length 7, capacity 8 = {"SEvt__Init_RUN_META", "CSGOptiX__SimulateMain_HEAD", "CSGFoundry__Load_HEAD", "CSGFoundry__Load_TAIL", "CSGOptiX__Create_HEAD", "CSGOptiX__Create_TAIL", "A000_QSim__simulate_HEAD"}
    (gdb) p SProf::META
    $3 = std::vector of length 7, capacity 8 = {"", "", "", "", "", "", ""}
    (gdb) p SProf::PROF
    $4 = std::vector of length 7, capacity 8 = {{st = 1780480291765766, vm = 49312, rs = 8512}, {st = 1780480291771189, vm = 49444, rs = 12544}, {st = 1780480291771217, vm = 49444, rs = 12544}, {st = 1780480292809241, vm = 5427332, rs = 875288}, {
        st = 1780480292809269, vm = 5427332, rs = 875288}, {st = 1780480293171176, vm = 8129524, rs = 1204440}, {st = 1780480293171216, vm = 8129524, rs = 1204440}}
    (gdb) 


    (gdb) bt
    #0  SEvt::BeginOfRun () at /home/blyth/opticks/sysrap/SEvt.cc:1636
    #1  0x00007ffff597d60f in SEvt::beginOfEvent (this=0x470d40, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1854
    #2  0x00007ffff6283416 in QSim::simulate (this=0x17f52b50, eventID=0, reset_=true) at /home/blyth/opticks/qudarap/QSim.cc:453
    #3  0x00007ffff7e5d78a in CSGOptiX::simulate (this=0x17f669d0, eventID=0, reset=true) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:732
    #4  0x00007ffff7e5a2f0 in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:191
    #5  0x0000000000404b15 in main (argc=1, argv=0x7fffffffb388) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) f 1
    #1  0x00007ffff597d60f in SEvt::beginOfEvent (this=0x470d40, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1854
    1854	    if(isFirstEvtInstance() && eventID == 0) BeginOfRun() ;
    (gdb) list
    1849	**/
    1850	
    1851	
    1852	void SEvt::beginOfEvent(int eventID)
    1853	{
    1854	    if(isFirstEvtInstance() && eventID == 0) BeginOfRun() ;
    1855	    if(eventID == 0) SProf::Add( isEGPU() ? "SEvt__beginOfEvent_FIRST_EGPU" : "SEvt__beginOfEvent_FIRST_ECPU" ) ;
    1856	
    1857	    setStage(SEvt__beginOfEvent);
    1858	    sprof::Stamp(p_SEvt__beginOfEvent_0);
    (gdb) b 1857
    Breakpoint 2 at 0x7ffff597d649: file /home/blyth/opticks/sysrap/SEvt.cc, line 1857.
    (gdb) c
    Continuing.

    Thread 1 "CSGOptiXSMTest" hit Breakpoint 2, SEvt::beginOfEvent (this=0x470d40, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1857
    1857	    setStage(SEvt__beginOfEvent);
    (gdb) p SProf::NAME
    $5 = std::vector of length 9, capacity 16 = {"SEvt__Init_RUN_META", "CSGOptiX__SimulateMain_HEAD", "CSGFoundry__Load_HEAD", "CSGFoundry__Load_TAIL", "CSGOptiX__Create_HEAD", "CSGOptiX__Create_TAIL", "A000_QSim__simulate_HEAD", 
      "A000_SEvt__BeginOfRun", "A000_SEvt__beginOfEvent_FIRST_EGPU"}
    (gdb) p SProf::META
    $6 = std::vector of length 9, capacity 16 = {"", "", "", "", "", "", "", "", ""}
    (gdb)  p SProf::PROF
    $7 = std::vector of length 9, capacity 16 = {{st = 1780480291765766, vm = 49312, rs = 8512}, {st = 1780480291771189, vm = 49444, rs = 12544}, {st = 1780480291771217, vm = 49444, rs = 12544}, {st = 1780480292809241, vm = 5427332, rs = 875288}, {
        st = 1780480292809269, vm = 5427332, rs = 875288}, {st = 1780480293171176, vm = 8129524, rs = 1204440}, {st = 1780480293171216, vm = 8129524, rs = 1204440}, {st = 1780480525338873, vm = 8129524, rs = 1204440}, {st = 1780480525339039, 
        vm = 8129524, rs = 1204440}}
    (gdb) 




AHHA is has prefix : "A000_SEvt__BeginOfRun" after SProf::SetTag called::

    125     static char TAG[N] ;
    126     static int  SetTag(int idx, const char* fmt=FMT ); // used to distinguish profiles from multiple events
    127     static bool HasTag();


DONE :  SProf::SetTag SProf::UnsetTag moved to SEvt level - not QSim and CSGOptiX
------------------------------------------------------------------------------------

::

    [lo] A[blyth@localhost sysrap]$ opticks-f SProf::SetTag
    ./qudarap/QSim.cc:    SProf::SetTag(eventID, "A%0.3d_" ) ;
    ./sysrap/SProf.hh:SProf::SetTag
    ./sysrap/SProf.hh:inline int SProf::SetTag(int idx, const char* fmt)
    ./sysrap/tests/SProfTest.cc:        SProf::SetTag(i, "A%0.3d_" ) ;
    ./sysrap/tests/SProfTest.cc:        SProf::SetTag(i, "A%0.3d_" );
    ./u4/tests/U4HitTest.cc:    SProf::SetTag(hidx);
    ./u4/tests/U4HitTest.cc:    SProf::SetTag(hidx);
    ./u4/tests/U4HitTest.cc:    SProf::SetTag(hidx);


    [lo] A[blyth@localhost qudarap]$ opticks-f SProf::UnsetTag
    ./CSGOptiX/CSGOptiX.cc:    SProf::UnsetTag();
    ./sysrap/SProf.hh:inline void SProf::UnsetTag()
    ./sysrap/tests/SProfTest.cc:    SProf::UnsetTag();
    ./sysrap/tests/SProfTest.cc:        if(i % 10 == 0 ) SProf::UnsetTag();
    ./sysrap/tests/SProfTest.cc:    SProf::UnsetTag();
    [lo] A[blyth@localhost opticks]$ 


FIXED : avoiding the A000 prefix did not solve issue, problem is that moved profiles to SProf from run_meta ? 
--------------------------------------------------------------------------------------------------------------


Fixed by moving rr collection into new method sevt.py:init_prof

::


    [lo] A[blyth@localhost CSGOptiX]$ ~/o/cxs_min.sh pdb
    Python 3.13.2 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:02) [GCC 11.2.0]
    Type 'copyright', 'credits' or 'license' for more information
    IPython 9.1.0 -- An enhanced Interactive Python. Type '?' for help.
    Tip: You can use LaTeX or Unicode completion, `\alpha<tab>` will insert the α symbol.
    [from opticks.sysrap.sevt import SEvt, SAB
    pvplt MODE:2 
    [from opticks.ana.p import * 
    [ana/p.py:from opticks.CSG.CSGFoundry import CSGFoundry 
    ]ana/p.py:from opticks.CSG.CSGFoundry import CSGFoundry 
    [ana/p.py:cf = CSGFoundry.Load()
    CSGFoundry.CFBase returning [/home/blyth/junosw/InstallArea/blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV/.opticks/GEOM/J26_1_1_opticks_Debug], note:[via GEOM,J26_1_1_opticks_Debug_CFBaseFromGEOM] 
    ]ana/p.py:cf = CSGFoundry.Load()
    ]from opticks.ana.p import * 
    qcf.py[ with_argmax 0 with_f2py_qcf_ab 0 ]
    ]from opticks.sysrap.sevt import SEvt, SAB
    before pvplt import MODE:3 
    GLOBAL:0 MODE:2 SEL:0
    INFO:opticks.ana.pvplt:SEvt.Load NEVT:0 
    INFO:opticks.ana.fold:Fold.Load args ('$AFOLD',) quiet:1
    sevt.init W2M
     None
    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    File ~/opticks/CSGOptiX/cxs_min.py:38
         35 logging.basicConfig(level=logging.INFO)
         36 print("GLOBAL:%d MODE:%d SEL:%d" % (GLOBAL,MODE, SEL))
    ---> 38 a = SEvt.Load("$AFOLD", symbol="a")
         39 print(repr(a))
         41 if "BFOLD" in os.environ:

    File ~/opticks/sysrap/sevt.py:59, in SEvt.Load(cls, *args, **kwa)
         57     f = Fold.Load(*args, **kwa )
         58 pass
    ---> 59 return None if f is None else cls(f, **kwa)

    File ~/opticks/sysrap/sevt.py:89, in SEvt.__init__(self, f, **kwa)
         86 self.W2M = kwa.get("W2M", None)
         87 print("sevt.init W2M\n", self.W2M )
    ---> 89 self.init_run(f)
         90 self.init_meta(f)
         91 self.init_fold_meta_timestamp(f)

    File ~/opticks/sysrap/sevt.py:149, in SEvt.init_run(self, f)
        145 has_run_meta = not run_meta is None
        147 prof2stamp_ = lambda prof:prof.split(",")[0]
    --> 149 rr_ = [prof2stamp_(run_meta.SEvt__BeginOfRun),
        150        prof2stamp_(run_meta.SEvt__EndOfRun  )] if has_run_meta else [0,0]
        152 rr = np.array(rr_, dtype=np.uint64 )
        154 self.rr = rr

    File ~/opticks/ana/npmeta.py:147, in NPMeta.__getattr__(self, k)
        145 def __getattr__(self, k):
        146     if not k in self.d:
    --> 147          raise AttributeError("No attribute %s " % k) 
        148     return self.find(k)

    AttributeError: No attribute SEvt__BeginOfRun 
    > /home/blyth/opticks/ana/npmeta.py(147)__getattr__()
        145     def __getattr__(self, k):
        146         if not k in self.d:
    --> 147              raise AttributeError("No attribute %s " % k)
        148         return self.find(k)
        149 

    ipdb>




FIXED : cxs_min.py Outdated use of sframe
-------------------------------------------

::

    qtab
    [[1 0 None]]
    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    File ~/opticks/CSGOptiX/cxs_min.py:81
         79 gpos = np.ones( [len(pp), 4 ] )
         80 gpos[:,:3] = pp
    ---> 81 lpos = np.dot( gpos, e.f.sframe.w2m )   ## sframe OUTDATED : sfr NOW ?
         82 upos = gpos if GLOBAL else lpos
         84 if MODE in [0,1]:

    AttributeError: 'Fold' object has no attribute 'sframe'
    > /home/blyth/opticks/CSGOptiX/cxs_min.py(81)<module>()
         79     gpos = np.ones( [len(pp), 4 ] )
         80     gpos[:,:3] = pp
    ---> 81     lpos = np.dot( gpos, e.f.sframe.w2m )   ## sframe OUTDATED : sfr NOW ?
         82     upos = gpos if GLOBAL else lpos
         83 

    ipdb>



NEXT : get cxs_min.sh running and analysis to work from release without using source tree (on Workstation, then Server)
-------------------------------------------------------------------------------------------------------------------------


* :doc:`source_tree_install_tree_isolation`

