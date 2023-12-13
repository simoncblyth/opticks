getFrameComponents called twice
------------------------------------

::

    (gdb) bt
    #0  CSGTarget::getFrameComponents (this=0x4727a30, ce=..., midx=123, mord=0, iidxg=0, m2w=0x7fffffff0360, w2m=0x7fffffff03a0) at /home/blyth/junotop/opticks/CSG/CSGTarget.cc:209
    #1  0x00007ffff7cb8bf3 in CSGTarget::getFrame (this=0x4727a30, fr=..., midx=123, mord=0, iidxg=0) at /home/blyth/junotop/opticks/CSG/CSGTarget.cc:138
    #2  0x00007ffff7c241c7 in CSGFoundry::getFrame (this=0x685eed0, fr=..., midx=123, mord=0, iidxg=0) at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3361
    #3  0x00007ffff7c23f7a in CSGFoundry::getFrame (this=0x685eed0, fr=..., frs=0x7fffffffaa33 "sChimneyAcrylic:0:0") at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3329
    #4  0x00007ffff7c23d49 in CSGFoundry::getFrame (this=0x685eed0, frs=0x7fffffffaa33 "sChimneyAcrylic:0:0") at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3284
    #5  0x00007ffff7c23ca8 in CSGFoundry::getFrame (this=0x685eed0) at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3279
    #6  0x00007ffff7c244a8 in CSGFoundry::getFrameE (this=0x685eed0) at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3408
    #7  0x00007ffff7c246db in CSGFoundry::AfterLoadOrCreate () at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3444
    #8  0x00007ffff7c21f98 in CSGFoundry::Load () at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:2873
    #9  0x00007ffff7e594e8 in CSGOptiX::SimtraceMain () at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:168
    #10 0x0000000000405b15 in main (argc=1, argv=0x7fffffff1598) at /home/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 

    (gdb) c
    Continuing.
    2023-12-13 16:23:43.746 INFO  [372293] [CSGTarget::getFrameComponents@209]  (midx mord iidxg) (123 0 0) 
    2023-12-13 16:23:43.749 INFO  [372293] [CSGTarget::getInstanceTransform@439]  (midx mord iidx) (123 0 0)  lpr 0x80b1800 repeatIdx 0 primIdx 2920 local_ce ( 0.000, 0.000,18124.000,524.000) 
    2023-12-13 16:23:43.751 INFO  [372293] [CSGTarget::getGlobalCenterExtent@337] 
    t:[   1.000    0.000    0.000    0.000 ][   0.000    1.000    0.000    0.000 ][   0.000    0.000    1.000    0.000 ][   0.000    0.000    0.000     -nan ] ( i/g/si/sx       0  0      0   -1 )
    v:[   1.000   -0.000    0.000    0.000 ][  -0.000    1.000   -0.000    0.000 ][   0.000   -0.000    1.000    0.000 ][  -0.000    0.000   -0.000     -nan ] ( i/g/si/sx       0  0      0   -1 )
    2023-12-13 16:23:43.751 INFO  [372293] [CSGTarget::getGlobalCenterExtent@373] 
     q ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
     ins_idx 0 gas_idx 0 sensor_identifier 0 sensor_index -1
    2023-12-13 16:23:43.751 INFO  [372293] [CSGTarget::getGlobalCenterExtent@388]  gpr CSGPrim numNode/node/tran/plan   3 15662 7171    0 sbtOffset/meshIdx/repeatIdx/primIdx 2920  123    0 2920 mn (-524.000,-524.000,17824.000)  mx (524.000,524.000,18424.000)  gce ( 0.000, 0.000,18124.000,524.000) 
    2023-12-13 16:23:43.751 INFO  [372293] [CSGTarget::getFrame@139]  midx 123 mord 0 iidxg 0 rc 0
    [Detaching after fork from child process 376099]
    [New Thread 0x7fffe0ba1700 (LWP 376106)]
    [New Thread 0x7fffcd0dc700 (LWP 376117)]

    Thread 1 "CSGOptiXTMTest" hit Breakpoint 1, CSGTarget::getFrameComponents (this=0x4727a30, ce=..., midx=123, mord=0, iidxg=0, m2w=0x7ffffffefcb0, w2m=0x7ffffffefcf0) at /home/blyth/junotop/opticks/CSG/CSGTarget.cc:209
    209	    LOG(LEVEL) << " (midx mord iidxg) " << "(" << midx << " " << mord << " " << iidxg << ") " ;  
    (gdb) bt
    #0  CSGTarget::getFrameComponents (this=0x4727a30, ce=..., midx=123, mord=0, iidxg=0, m2w=0x7ffffffefcb0, w2m=0x7ffffffefcf0) at /home/blyth/junotop/opticks/CSG/CSGTarget.cc:209
    #1  0x00007ffff7cb8bf3 in CSGTarget::getFrame (this=0x4727a30, fr=..., midx=123, mord=0, iidxg=0) at /home/blyth/junotop/opticks/CSG/CSGTarget.cc:138
    #2  0x00007ffff7c241c7 in CSGFoundry::getFrame (this=0x685eed0, fr=..., midx=123, mord=0, iidxg=0) at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3361
    #3  0x00007ffff7c23f7a in CSGFoundry::getFrame (this=0x685eed0, fr=..., frs=0x7fffffffaa33 "sChimneyAcrylic:0:0") at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3329
    #4  0x00007ffff7c23d49 in CSGFoundry::getFrame (this=0x685eed0, frs=0x7fffffffaa33 "sChimneyAcrylic:0:0") at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3284
    #5  0x00007ffff7c23ca8 in CSGFoundry::getFrame (this=0x685eed0) at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3279
    #6  0x00007ffff7c244a8 in CSGFoundry::getFrameE (this=0x685eed0) at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:3408
    #7  0x00007ffff7e5c318 in CSGOptiX::initFrame (this=0xacd1780) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:632
    #8  0x00007ffff7e5ac61 in CSGOptiX::init (this=0xacd1780) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:455
    #9  0x00007ffff7e5a752 in CSGOptiX::CSGOptiX (this=0xacd1780, foundry_=0x685eed0) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:424
    #10 0x00007ffff7e5a24c in CSGOptiX::Create (fd=0x685eed0) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:359
    #11 0x00007ffff7e594f8 in CSGOptiX::SimtraceMain () at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:169
    #12 0x0000000000405b15 in main (argc=1, argv=0x7fffffff1598) at /home/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 

    (gdb) c
    Continuing.
    2023-12-13 16:24:36.135 INFO  [372293] [CSGTarget::getFrameComponents@209]  (midx mord iidxg) (123 0 0) 
    2023-12-13 16:24:36.135 INFO  [372293] [CSGTarget::getInstanceTransform@439]  (midx mord iidx) (123 0 0)  lpr 0x80b1800 repeatIdx 0 primIdx 2920 local_ce ( 0.000, 0.000,18124.000,524.000) 
    2023-12-13 16:24:36.137 INFO  [372293] [CSGTarget::getGlobalCenterExtent@337] 
    t:[   1.000    0.000    0.000    0.000 ][   0.000    1.000    0.000    0.000 ][   0.000    0.000    1.000    0.000 ][   0.000    0.000    0.000     -nan ] ( i/g/si/sx       0  0      0   -1 )
    v:[   1.000   -0.000    0.000    0.000 ][  -0.000    1.000   -0.000    0.000 ][   0.000   -0.000    1.000    0.000 ][  -0.000    0.000   -0.000     -nan ] ( i/g/si/sx       0  0      0   -1 )
    2023-12-13 16:24:36.137 INFO  [372293] [CSGTarget::getGlobalCenterExtent@373] 
     q ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) ( 0.000, 0.000, 0.000, 1.000) 
     ins_idx 0 gas_idx 0 sensor_identifier 0 sensor_index -1
    2023-12-13 16:24:36.137 INFO  [372293] [CSGTarget::getGlobalCenterExtent@388]  gpr CSGPrim numNode/node/tran/plan   3 15662 7171    0 sbtOffset/meshIdx/repeatIdx/primIdx 2920  123    0 2920 mn (-524.000,-524.000,17824.000)  mx (524.000,524.000,18424.000)  gce ( 0.000, 0.000,18124.000,524.000) 
    2023-12-13 16:24:36.137 INFO  [372293] [CSGTarget::getFrame@139]  midx 123 mord 0 iidxg 0 rc 0
    //CSGOptiX7.cu : simtrace idx 0 genstep_id 0 evt->num_simtrace 1254000 
    2023-12-13 16:24:36.413 INFO  [372293] [SEvt::save@3942] /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXTMTest/sChimneyAcrylic:0:0/A000 genstep,simtrace
    [Thread 0x7fffcd0dc700 (LWP 376117) exited]
    [Thread 0x7fffe0ba1700 (LWP 376106) exited]
    [Thread 0x7ffff44f7000 (LWP 372293) exited]
    [Thread 0x7fffecfb9700 (LWP 372378) exited]
    [New process 372293]
    [Inferior 1 (process 372293) exited normally]
    (gdb) 



