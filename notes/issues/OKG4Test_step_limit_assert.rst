OKG4Test_step_limit_assert
=============================


::


    N[blyth@localhost opticks]$ gdb OKG4Test
    GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-114.el7
    ...
    2022-06-06 18:07:14.318 INFO  [66155] [NPY<T>::MakeFloat@2042]  nv 1024
    2022-06-06 18:07:14.319 INFO  [66155] [NPY<T>::MakeFloat@2042]  nv 12288
    2022-06-06 18:07:14.321 INFO  [66155] [OGeo::init@227]  is_gparts_transform_offset 0
    2022-06-06 18:07:14.321 INFO  [66155] [OGeo::init@254] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
    2022-06-06 18:07:14.329 INFO  [66155] [OGeo::convert@312] 
    OGeo::convert GGeoLib numMergedMesh 10 ptr 0x13a9240
    mm index   0 geocode   A                  numVolumes       3089 numFaces      182820 numITransforms           1 numITransforms*numVolumes        3089 GParts Y GPts Y
    mm index   1 geocode   A                  numVolumes          5 numFaces        1584 numITransforms       25600 numITransforms*numVolumes      128000 GParts Y GPts Y
    mm index   2 geocode   A                  numVolumes          7 numFaces        4320 numITransforms       12615 numITransforms*numVolumes       88305 GParts Y GPts Y
    mm index   3 geocode   A                  numVolumes          7 numFaces        5426 numITransforms        4997 numITransforms*numVolumes       34979 GParts Y GPts Y
    mm index   4 geocode   A                  numVolumes          6 numFaces        3284 numITransforms        2400 numITransforms*numVolumes       14400 GParts Y GPts Y
    mm index   5 geocode   A                  numVolumes          1 numFaces         528 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   6 geocode   A                  numVolumes          1 numFaces         960 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   7 geocode   A                  numVolumes          1 numFaces         384 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   8 geocode   A                  numVolumes          1 numFaces         240 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   9 geocode   A                  numVolumes        130 numFaces        1560 numITransforms         504 numITransforms*numVolumes       65520 GParts Y GPts Y
     num_remainder_volumes 3089 num_instanced_volumes 333564 num_remainder_volumes + num_instanced_volumes 336653 num_total_faces 201106 num_total_faces_woi 132257662 (woi:without instancing) 
       0 pts Y  GPts.NumPt  3089 lvIdx ( 138 17 2 1 0 16 15 3 7 4 ... 103 103 103 103 103 126 123 124 125) 0:1 1:1 2:1 3:1 4:1 5:1 6:1 7:1 12:126 13:63 14:1 15:1 16:1 17:1 18:10 19:30 20:30 21:30 22:30 23:30 24:30 25:30 26:30 27:30 28:30 29:30 30:30 31:30 32:30 33:30 34:30 35:30 36:30 37:30 38:30 39:10 40:30 41:30 42:30 43:30 44:30 45:30 46:30 47:30 48:30 49:30 50:30 51:30 52:30 53:30 54:30 55:30 56:30 57:30 58:30 59:30 60:30 61:30 62:30 63:30 64:30 65:30 66:30 67:30 68:30 69:30 70:30 71:30 72:30 73:30 74:30 75:30 76:30 77:30 78:30 79:30 80:30 81:30 82:30 83:30 84:30 85:30 86:30 87:30 88:30 89:30 90:2 91:36 92:8 93:8 94:1 95:1 96:370 97:220 102:56 103:56 123:1 124:1 125:1 126:1 127:1 128:1 135:1 136:1 137:1 138:1
       1 pts Y  GPts.NumPt     5 lvIdx ( 122 120 118 119 121) 118 119 120 121 122 all_same_count 1
       2 pts Y  GPts.NumPt     7 lvIdx ( 117 111 112 116 115 113 114) 111 112 113 114 115 116 117 all_same_count 1
       3 pts Y  GPts.NumPt     7 lvIdx ( 110 104 105 109 108 106 107) 104 105 106 107 108 109 110 all_same_count 1
       4 pts Y  GPts.NumPt     6 lvIdx ( 134 129 133 132 130 131) 129 130 131 132 133 134 all_same_count 1
       5 pts Y  GPts.NumPt     1 lvIdx ( 98) 98 all_same_count 1
       6 pts Y  GPts.NumPt     1 lvIdx ( 99) 99 all_same_count 1
       7 pts Y  GPts.NumPt     1 lvIdx ( 100) 100 all_same_count 1
       8 pts Y  GPts.NumPt     1 lvIdx ( 101) 101 all_same_count 1
       9 pts Y  GPts.NumPt   130 lvIdx ( 11 10 9 8 9 8 9 8 9 8 ... 8 9 8 9 8 9 8 9 8) 8:64 9:64 10:1 11:1

    2022-06-06 18:07:14.329 INFO  [66155] [OGeo::convert@316] [ nmm 10
    2022-06-06 18:07:15.625 INFO  [66155] [OGeo::convert@335] ] nmm 10
    2022-06-06 18:07:15.634 INFO  [66155] [NPY<T>::MakeFloat@2042]  nv 1071488
    2022-06-06 18:07:15.726 ERROR [66155] [cuRANDWrapper::setItems@154] CAUTION : are resizing the launch sequence 
    2022-06-06 18:07:16.505 FATAL [66155] [ORng::setSkipAhead@155] skipahead 0
    2022-06-06 18:07:16.591 FATAL [66155] [CWriter::expand@153]  gs_photons 100 Cannot expand as CWriter::initEvent has not been called   check CManager logging, perhaps --save not enabled   m_ok->isSave() 0 OR BeginOfGenstep notifications not received  m_BeginOfGenstep_count 1
    CWriter  m_enabled 1 m_evt 0 m_ni 0 m_BeginOfGenstep_count 1 m_records_buffer 0 m_deluxe_buffer 0 m_photons_buffer 0 m_history_buffer 0
    2022-06-06 18:08:54.174 INFO  [66155] [CScint::Check@16]  pmanager 0xc099630 proc 0
    2022-06-06 18:08:54.174 INFO  [66155] [CScint::Check@21] CProMgr n:[4] (0) name Transportation left -1 (1) name OpAbsorption left -1 (2) name OpRayleigh left -1 (3) name OpBoundary left -1
    2022-06-06 18:08:54.174 INFO  [66155] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : TORCH
    OKG4Test: /data/blyth/junotop/opticks/cfg4/CCtx.cc:104: unsigned int CCtx::step_limit() const: Assertion `_ok_event_init' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe4f26387 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe4f26387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe4f27a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe4f1f1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe4f1f252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff4ab0017 in CCtx::step_limit (this=0x1dd53ef0) at /data/blyth/junotop/opticks/cfg4/CCtx.cc:104
    #5  0x00007ffff4a45226 in CRec::add (this=0x1dd4f660, boundary_status=FresnelRefraction) at /data/blyth/junotop/opticks/cfg4/CRec.cc:286
    #6  0x00007ffff4a8ba25 in CRecorder::Record (this=0x1dd4f560, boundary_status=FresnelRefraction) at /data/blyth/junotop/opticks/cfg4/CRecorder.cc:357
    #7  0x00007ffff4ab77f8 in CManager::setStep (this=0x1dd53e90, step=0xaebae50) at /data/blyth/junotop/opticks/cfg4/CManager.cc:490
    #8  0x00007ffff4ab7440 in CManager::UserSteppingAction (this=0x1dd53e90, step=0xaebae50) at /data/blyth/junotop/opticks/cfg4/CManager.cc:415
    #9  0x00007ffff4aaf220 in CSteppingAction::UserSteppingAction (this=0xc92acd0, step=0xaebae50) at /data/blyth/junotop/opticks/cfg4/CSteppingAction.cc:41
    #10 0x00007ffff157de1d in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #11 0x00007ffff1589472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #12 0x00007ffff17c0389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #13 0x00007ffff1a5ba6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #14 0x00007ffff1a5953e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #15 0x00007ffff4ab4846 in CG4::propagate (this=0xad30520) at /data/blyth/junotop/opticks/cfg4/CG4.cc:438
    #16 0x00007ffff7ba8fbf in OKG4Mgr::propagate_ (this=0x7fffffff6da0) at /data/blyth/junotop/opticks/okg4/OKG4Mgr.cc:269
    #17 0x00007ffff7ba884c in OKG4Mgr::propagate (this=0x7fffffff6da0) at /data/blyth/junotop/opticks/okg4/OKG4Mgr.cc:162
    #18 0x00000000004056f9 in main (argc=1, argv=0x7fffffff7128) at /data/blyth/junotop/opticks/okg4/tests/OKG4Test.cc:29
    (gdb) 


