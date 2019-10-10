cluster-tboolean-fail-GMeshLib-paired-assert
===============================================


Issue
-----------

This fail:

* doesnt happen on workstation 
* just started today on cluster, yesterday evening no such problem  
* note some use of tmp paths from tboolean test running when oktmp expected : maybe python tmp divergence ?



::
     955         Start 415: IntegrationTests.tboolean.box
     956 415/415 Test #415: IntegrationTests.tboolean.box ...............................***Failed    9.37 sec


    1019 2019-10-10 14:47:35.157 INFO  [314044] [Opticks::loadOriginCacheMeta@1778]  gdmlpath /hpcfs/juno/junogpu/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v3.gdml
    1020 2019-10-10 14:47:35.158 ERROR [314044] [OpticksHub::configure@332] FORCED COMPUTE MODE : as remote session detected
    1021 2019-10-10 14:47:35.158 INFO  [314044] [OpticksHub::loadGeometry@542] [ /hpcfs/juno/junogpu/blyth/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
    1022 2019-10-10 14:47:38.678 ERROR [314044] [GItemList::load_@75]  MISSING ITEMLIST TXT  txtpath /hpcfs/juno/junogpu/blyth/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe     846377973af10a/1/GItemList/GMeshLib.txt txtname GMeshLib.txt reldir GItemList m_itemtype GMeshLib
    1023 OKG4Test: /hpcfs/juno/junogpu/blyth/opticks/ggeo/GMeshLib.cc:462: void GMeshLib::loadMeshes(const char*): Assertion soliddir_exists && "GMeshLib persisted GMesh are expected to have paired GMeshLibNCSG dirs" failed.
    1024 /hpcfs/juno/junogpu/blyth/opticks/bin/o.sh: line 254: 314044 Aborted                 (core dumped) /hpcfs/juno/junogpu/blyth/local/opticks/lib/OKG4Test --okg4test --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --generateoverride 10000 -     -envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 --up 0,0,1 --test --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box_outerfirst=1_analytic=1_csgpath=/hpcfs/juno/junogpu/blyth/tmp/blyth/opticks/tboolean-box_mode=PyCsgInBox_     autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autocontainer=Rock//perfectAbso     rbSurface/Vacuum --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.0_radiu     s=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --torchdbg --tag 1 --anakey tboolean --args --save
    1025 === o-main : /hpcfs/juno/junogpu/blyth/local/opticks/lib/OKG4Test --okg4test --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --generateoverride 10000 --envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 --up 0,0,1 --t     est --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box_outerfirst=1_analytic=1_csgpath=/hpcfs/juno/junogpu/blyth/tmp/blyth/opticks/tboolean-box_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,     wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autocontainer=Rock//perfectAbsorbSurface/Vacuum --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarizati     on=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.0_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --torchdbg --t     ag 1 --anakey tboolean --args --save ======= PWD /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-




Try running this on lxslc the fail from no-GPU should be later than this::

    [blyth@lxslc701 tests]$ LV=box tboolean.sh --generateoverride 10000 -D


It appears to get further::

    (gdb) bt
    #0  0x00007fffe1fcc207 in raise () from /usr/lib64/libc.so.6
    #1  0x00007fffe1fcd8f8 in abort () from /usr/lib64/libc.so.6
    #2  0x00007fffe28db7d5 in __gnu_cxx::__verbose_terminate_handler() () from /usr/lib64/libstdc++.so.6
    #3  0x00007fffe28d9746 in ?? () from /usr/lib64/libstdc++.so.6
    #4  0x00007fffe28d9773 in std::terminate() () from /usr/lib64/libstdc++.so.6
    #5  0x00007fffe28d9993 in __cxa_throw () from /usr/lib64/libstdc++.so.6
    #6  0x00007ffff65727f4 in thrust::system::cuda::detail::cuda_memory_resource<&cudaMalloc, &cudaFree, thrust::cuda_cub::pointer<void> >::do_allocate(unsigned long, unsigned long) ()
           from /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/lib/../lib64/libOptiXRap.so
    #7  0x00007ffff65721fe in thrust::device_ptr_memory_resource<thrust::system::cuda::detail::cuda_memory_resource<&cudaMalloc, &cudaFree, thrust::cuda_cub::pointer<void> > >::do_allocate(unsigned long, unsigned long) ()
           from /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/lib/../lib64/libOptiXRap.so
    #8  0x00007fffea265c96 in thrust::detail::vector_base<double, thrust::device_allocator<double> >::vector_base(unsigned long) ()
           from /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/lib/../lib64/libThrustRap.so
    #9  0x00007fffea265d94 in TCURANDImp<double>::TCURANDImp(unsigned int, unsigned int, unsigned int) () from /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/lib/../lib64/libThrustRap.so
    #10 0x00007fffea25fe74 in TCURAND<double>::TCURAND (this=0x61eb430, ni=100000, nj=16, nk=16) at /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TCURAND.cc:30
    #11 0x00007ffff4ca7743 in CRandomEngine::CRandomEngine (this=0x61ed4a0, g4=0x61e6960) at /hpcfs/juno/junogpu/blyth/opticks/cfg4/CRandomEngine.cc:104
    #12 0x00007ffff4c9c14c in CG4::CG4 (this=0x61e6960, hub=0x6bbc30) at /hpcfs/juno/junogpu/blyth/opticks/cfg4/CG4.cc:155
    #13 0x00007ffff7bd45af in OKG4Mgr::OKG4Mgr (this=0x7fffffffbfd0, argc=32, argv=0x7fffffffc318) at /hpcfs/juno/junogpu/blyth/opticks/okg4/OKG4Mgr.cc:107
    #14 0x0000000000403a0a in main (argc=32, argv=0x7fffffffc318) at /hpcfs/juno/junogpu/blyth/opticks/okg4/tests/OKG4Test.cc:27
    (gdb) 

