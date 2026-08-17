cxr_min_black_screen_init_fail_with_raindrop
==============================================


Issue
------

cxr_min.sh works fine with JUNO geom but failing with the very simple
raindrop geometry ? FIXED initial issue - but further problems remain.


Overview
---------

* disabling AFOLD, BFOLD in EVT.sh for RaindropRockAirWater enables geometry render to work,
  by switching off the attempted upload of 20G record.npy from AFOLD
* then config slicing gets animation to work


Nowever, notice issues with the viz:

* points do not appear until after hitting the droplet - recall some viz problem with sim time starting at exactly zero ?

  * confirmed that setting AFOLD_RECORD_TNUDGE=0.1 enables viz of records prior to hitting the droplet

* compositing with the geometry looses most of the record animation - zdepth consistency from view math perhaps ?

  * looks like making the containers invisible avoids this issue to some extent


Try making container volumes invisible to see if changes record viz issue
--------------------------------------------------------------------------

::

    cat /home/blyth/.opticks/GEOM/RaindropRockAirWater/CSGFoundry/meshname.txt
    G4_WATER_solid
    G4_AIR_solid
    G4_Pb_solid
    VACUUM_solid




Symptom : init crash
---------------------


::

    [lo] A[blyth@localhost CSGOptiX]$ FULLSCREEN=0 cxr_min.sh dbg
                             GEOM_METHOD : local sourcing of ~/.opticks/GEOM/GEOM.sh
                                    GEOM : RaindropRockAirWater
                                 NOXGEOM :
                 External_CFBaseFromGEOM : _CFBaseFromGEOM
                         _CFBaseFromGEOM :
                   BASH_SOURCE : /home/blyth/.opticks/GEOM/ELV.sh
                          GEOM : RaindropRockAirWater
                      elv_name : skip_big
                      ELV_NAME : skip_big
                           elv :
                           ELV :
                    elv_branch : -1
                    ELV_BRANCH : -1

    ...

    [Thread 0x7ff649069000 (LWP 521604) exited]
    [New Thread 0x7ff649069000 (LWP 521605)]
    SGLFW__check
    ( vi /data1/blyth/local/opticks_Debug/include/SysRap/SGLFW_Buffer.h  +49 )
     ctx     SGLFW_Record.buf
     id    3
     act              glBufferData/uplo
     err 501
     errstr  : GL_INVALID_VALUE
     [SGLFW_check__level] 0

    CSGOptiXRenderInteractiveTest: /data1/blyth/local/opticks_Debug/include/SysRap/SGLFW_check.h:53: void SGLFW__check(const char*, int, const char*, int, const char*): Assertion `ok' failed.

    Thread 1 "CSGOptiXRenderI" received signal SIGABRT, Aborted.
    0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install dbus-libs-1.12.20-8.el9.x86_64 glibc-2.34-168.el9_6.23.x86_64 libX11-1.7.0-11.el9.x86_64 libX11-xcb-1.7.0-11.el9.x86_64 libXau-1.0.9-8.el9.x86_64 libXext-1.3.4-8.el9.x86_64 libXfixes-5.0.3-16.el9.x86_64 libXi-1.7.10-8.el9.x86_64 libXinerama-1.1.4-10.el9.x86_64 libXrandr-1.5.2-8.el9.x86_64 libXrender-0.9.10-16.el9.x86_64 libXxf86vm-1.1.4-18.el9.x86_64 libcap-2.48-9.el9_2.x86_64 libdrm-2.4.123-2.el9.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libgcrypt-1.10.0-11.el9.x86_64 libglvnd-1.3.4-1.el9.x86_64 libglvnd-glx-1.3.4-1.el9.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64 libxcb-1.13.1-9.el9.x86_64 lz4-libs-1.9.3-5.el9.x86_64 nvidia-driver-common-610.43.02-1.el9.x86_64 nvidia-driver-cuda-libs-610.43.02-1.el9.x86_64 nvidia-driver-libs-610.43.02-1.el9.x86_64 openssl-libs-3.5.1-7.el9_7.x86_64 systemd-libs-252-55.el9_7.7.alma.1.x86_64 xz-libs-5.2.5-8.el9_0.x86_64
    (gdb) bt
    #0  0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff483eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff4828833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff482875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff4837886 in __assert_fail () from /lib64/libc.so.6
    #5  0x00000000004860d8 in SGLFW__check (path=0x4cfed0 "/data1/blyth/local/opticks_Debug/include/SysRap/SGLFW_Buffer.h", line=49, ctx=0x45ea890 "SGLFW_Record.buf", id=3, act=0x4cff21 "glBufferData/uplo")
        at /data1/blyth/local/opticks_Debug/include/SysRap/SGLFW_check.h:53
    #6  0x0000000000486268 in SGLFW_Buffer::upload (this=0x45ea860) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLFW_Buffer.h:49
    #7  0x000000000048c8e9 in SGLFW_Record::init (this=0x45ea7f0) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLFW_Record.h:63
    #8  0x000000000048c824 in SGLFW_Record::SGLFW_Record (this=0x45ea7f0, _record=0x54a8c0, _timeparam_ptr=0x60a59c) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLFW_Record.h:45
    #9  0x000000000048c79a in SGLFW_Record::Create (_record=0x54a8c0, _timeparam_ptr=0x60a59c) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLFW_Record.h:33
    #10 0x000000000048cd4a in SGLFW_Evt::SGLFW_Evt (this=0x45e8b90, _gl=...) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLFW_Evt.h:69
    #11 0x000000000048d629 in CSGOptiXRenderInteractiveTest::initRender (this=0x7fffffffb4f0) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:153
    #12 0x000000000048d55c in CSGOptiXRenderInteractiveTest::init (this=0x7fffffffb4f0) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:124
    #13 0x000000000048d4c8 in CSGOptiXRenderInteractiveTest::CSGOptiXRenderInteractiveTest (this=0x7fffffffb4f0) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:115
    #14 0x0000000000446831 in main (argc=1, argv=0x7fffffffb698) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:231
    (gdb)



::

    100 // CTOR
    101 inline CSGOptiXRenderInteractiveTest::CSGOptiXRenderInteractiveTest()
    102     :
    103     level(ssys::getenvint(_level,0)),
    104     ALLOW_REMOTE(ssys::getenvbool(_ALLOW_REMOTE)),
    105     irc(Initialize(ALLOW_REMOTE)),
    106     ar(SRecord::Load("$AFOLD", "$AFOLD_RECORD_SLICE")),
    107     br(SRecord::Load("$BFOLD", "$BFOLD_RECORD_SLICE")),
    108     fd(CSGFoundry::Load()),
    109     gm(new SGLM),
    110     cx(nullptr),
    111     gl(nullptr),
    112     interop(nullptr),
    113     glev(nullptr)
    114 {
    115     init();
    116 }
    117
    118 inline void CSGOptiXRenderInteractiveTest::init()
    119 {
    120     assert( irc == 0 );
    121     assert(fd);
    122
    123     initRecord();
    124     initRender();
    125 }
    126



::

    148 inline void CSGOptiXRenderInteractiveTest::initRender()
    149 {
    150     cx = CSGOptiX::Create(fd) ;
    151     gl = new SGLFW(*gm);
    152     interop = new SGLFW_CUDA(*gm);
    153     glev    = new SGLFW_Evt(*gl);
    154
    155     if(gl->level > 0) std::cout << "CSGOptiXRenderInteractiveTest::initRender before render loop  gl.get_wanted_frame_idx " <<  gl->get_wanted_frame_idx() << "\n" ;
    156     if(level > 0) std::cout << "CSGOptiXRenderInteractiveTest::initRender [" << _level << "][" << level << "]\n" << desc() ;
    157 }
    158



    064 inline SGLFW_Evt::SGLFW_Evt(SGLFW& _gl )
     65     :
     66     level(ssys::getenvint(_SGLFW_Evt__level,0)),
     67     gl(_gl),
     68     gm(gl.gm),
     69     ar(SGLFW_Record::Create(gm.ar, gm.timeparam_ptr)),   ############### FAILING HERE
     70     br(SGLFW_Record::Create(gm.br, gm.timeparam_ptr)),
     71     gs(SGLFW_Gen::Create(   gm.gs, gm.timeparam_ptr)),
     72     shader_fold(nullptr),
     73     rec_shader_name(nullptr),
     74     rec_shader_dir(nullptr),
     75     gen_shader_name(nullptr),
     76     gen_shader_dir(nullptr),
     77     rec_prog(nullptr),
     78     gen_prog(nullptr)
     79 {
     80     init();
     81 }



HUH : would have expected this line to do nothing when AFOLD not defined ?
AHHA -  AFOLD IS DEFINED FOR RAINDROP by EVT.sh AND record is 20G::

    [lo] A[blyth@localhost CSGOptiX]$ du -h /data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000/*.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000/domain.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000/genstep.npy
    611M	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000/photon.npy
    20G	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000/record.npy
    39M	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000/seqnib.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000/seqnib_table.npy
    306M	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000/seq.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000/sframe.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000/sfr.npy


    [lo] A[blyth@localhost CSGOptiX]$ du -h /data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/*.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/domain.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/genstep.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/hit.npy
    611M	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/photon.npy
    20G	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/record.npy
    39M	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/seqnib.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/seqnib_table.npy
    306M	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/seq.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/SEventConfig.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/sframe.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/sfr.npy
    4.0K	/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000/U4R.npy



Disable AFOLD BFOLD setting, gets the raindrop to render::

    264 elif [ "$GEOM" == "RaindropRockAirWater" ]; then
    265
    266     unset AFOLD
    267     unset BFOLD
    268     unset AFOLD_RECORD_SLICE
    269     unset BFOLD_RECORD_SLICE
    270
    271     #export AFOLD=/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000
    272     #export BFOLD=/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000
    273
    274 fi
    277
    "~/.opticks/GEOM/EVT.sh" 277L, 11493B written


Slicing the 20G record arrays, gets the record animation to run::

    264 elif [ "$GEOM" == "RaindropRockAirWater" ]; then
    265
    266     unset AFOLD
    267     unset BFOLD
    268     unset AFOLD_RECORD_SLICE
    269     unset BFOLD_RECORD_SLICE
    270
    271     export AFOLD=/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000
    272     export BFOLD=/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000
    273
    274     export AFOLD_RECORD_SLICE="[::10]"
    275     export BFOLD_RECORD_SLICE="[::10]"
    276
    277 fi
    278
    279
    280
    "~/.opticks/GEOM/EVT.sh" 280L, 11572B






