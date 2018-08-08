ab-lvname : MeshIndex lvIdx to lvName mapping is totally off for live geometry : Mesh ordering issue
======================================================================================================

Noticed this from getting incorrect lvNames from ab-prim when 
it was using IDPATH from the direct geocache. 

::

    epsilon:1 blyth$ ab-;ab-lv2name
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/104
    -rw-r--r--  1 blyth  staff  9448 Aug  7 13:18 MeshIndex/GItemIndexSource.json
      0 : near_top_cover_box0xc23f970 
      1 : RPCStrip0xc04bcb0 
      2 : RPCGasgap140xbf4c660 
      3 : RPCBarCham140xc2ba760 
      4 : RPCGasgap230xbf50468 
    ...
    244 : near-radslab-box-80xcd308c0 
    245 : near-radslab-box-90xcd31ea0 
    246 : near_hall_bot0xbf3d718 
    247 : near_rock0xc04ba08 
    248 : WorldBox0xc15cf40 
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/0dce832a26eb41b58a000497a3127cb8/1
    -rw-r--r--  1 blyth  staff  9472 Aug  7 13:19 MeshIndex/GItemIndexSource.json
      0 : WorldBox0xc15cf40 
      1 : near_rock0xc04ba08 
      2 : near_hall_top_dwarf0xc0316c8 
      3 : near_top_cover_box0xc23f970 
      4 : RPCMod0xc13bfd8 
    ...
    244 : near-radslab-box-50xccefd60 
    245 : near-radslab-box-60xccefda0 
    246 : near-radslab-box-70xccefde0 
    247 : near-radslab-box-80xcd308c0 
    248 : near-radslab-box-90xcd31ea0 
    epsilon:1 blyth$ 


FIXED : by reworking YOG Model and X4PhysicalVolume to move convertSolids prior to convertStructure
----------------------------------------------------------------------------------------------------

::

    epsilon:opticks blyth$ o
    M bin/ab.bash
    M extg4/X4Mesh.cc
    M extg4/X4Mesh.hh
    M extg4/X4PhysicalVolume.cc
    M extg4/X4PhysicalVolume.hh
    M ggeo/GGeo.cc
    M ggeo/GMesh.cc
    M ggeo/GMesh.hh
    M ggeo/GMeshLib.cc
    M ggeo/GMeshLib.hh
    M notes/issues/ab-lvname.rst
    M npy/NTreeProcess.cpp
    M optickscore/Opticks.cc
    M optickscore/Opticks.hh
    M optickscore/OpticksCfg.cc
    M optickscore/OpticksCfg.hh
    M optickscore/OpticksDbg.cc
    M optickscore/OpticksDbg.hh
    M yoctoglrap/YOG.cc
    M yoctoglrap/YOG.hh
    M yoctoglrap/YOGMaker.cc
    M yoctoglrap/YOGMaker.hh
    M yoctoglrap/YOGTF.cc
    epsilon:opticks blyth$ 

    epsilon:opticks blyth$ hg commit -m "rework X4PhysicalVolume and YOG model : adding Pr and removing mtIdx from Mh : to move convertSolids prior to convertStructure, fixing ab-lvname issue "



G4DAE COLLADA order
---------------------

::

    epsilon:issues blyth$ grep geometry\ id /tmp/g4_00.dae 
        <geometry id="near_top_cover_box0xc23f970" name="near_top_cover_box0xc23f970">
        <geometry id="RPCStrip0xc04bcb0" name="RPCStrip0xc04bcb0">
        <geometry id="RPCGasgap140xbf4c660" name="RPCGasgap140xbf4c660">
        <geometry id="RPCBarCham140xc2ba760" name="RPCBarCham140xc2ba760">
        <geometry id="RPCGasgap230xbf50468" name="RPCGasgap230xbf50468">
        <geometry id="RPCBarCham230xc125900" name="RPCBarCham230xc125900">
        <geometry id="RPCFoam0xc21f3f8" name="RPCFoam0xc21f3f8">
        <geometry id="RPCMod0xc13bfd8" name="RPCMod0xc13bfd8">
        <geometry id="NearRPCRoof0xc135b28" name="NearRPCRoof0xc135b28">
        ...
        <geometry id="near-radslab-box-40xcd53658" name="near-radslab-box-40xcd53658">
        <geometry id="near-radslab-box-50xccefd60" name="near-radslab-box-50xccefd60">
        <geometry id="near-radslab-box-60xccefda0" name="near-radslab-box-60xccefda0">
        <geometry id="near-radslab-box-70xccefde0" name="near-radslab-box-70xccefde0">
        <geometry id="near-radslab-box-80xcd308c0" name="near-radslab-box-80xcd308c0">
        <geometry id="near-radslab-box-90xcd31ea0" name="near-radslab-box-90xcd31ea0">
        <geometry id="near_hall_bot0xbf3d718" name="near_hall_bot0xbf3d718">
        <geometry id="near_rock0xc04ba08" name="near_rock0xc04ba08">
        <geometry id="WorldBox0xc15cf40" name="WorldBox0xc15cf40">
    epsilon:issues blyth$ 


GDML solids includes constituent solids that dont get their own mesh, but its plausibly 
the same order::

     .401   <solids>
      402     <box lunit="mm" name="near_top_cover0xc5843d8" x="16000" y="10000" z="44"/>
      403     <box lunit="mm" name="near_top_cover_sub00xc584418" x="4249.00272282321" y="4249.00272282321" z="54"/>
      404     <subtraction name="near_top_cover-ChildFornear_top_cover_box0xc241498">
      405       <first ref="near_top_cover0xc5843d8"/>
      406       <second ref="near_top_cover_sub00xc584418"/>
      407       <position name="near_top_cover-ChildFornear_top_cover_box0xc241498_pos" unit="mm" x="8000" y="5000" z="0"/>
      408       <rotation name="near_top_cover-ChildFornear_top_cover_box0xc241498_rot" unit="deg" x="0" y="0" z="45"/>
      409     </subtraction>
      410     <box lunit="mm" name="near_top_cover_sub10xc5844c0" x="4249.00272282321" y="4249.00272282321" z="54"/>
      411     <subtraction name="near_top_cover-ChildFornear_top_cover_box0xc04f720">
      412       <first ref="near_top_cover-ChildFornear_top_cover_box0xc241498"/>
      413       <second ref="near_top_cover_sub10xc5844c0"/>
      414       <position name="near_top_cover-ChildFornear_top_cover_box0xc04f720_pos" unit="mm" x="8000" y="-5000" z="0"/>
      415       <rotation name="near_top_cover-ChildFornear_top_cover_box0xc04f720_rot" unit="deg" x="0" y="0" z="45"/>
      416     </subtraction>
      ...
      431     <box lunit="mm" name="RPCStrip0xc04bcb0" x="2080" y="260" z="2"/>
      432     <box lunit="mm" name="RPCGasgap140xbf4c660" x="2080" y="2080" z="2"/>
      433     <box lunit="mm" name="RPCBarCham140xc2ba760" x="2100" y="2100" z="6"/>
      434     <box lunit="mm" name="RPCGasgap230xbf50468" x="2080" y="2080" z="2"/>
      435     <box lunit="mm" name="RPCBarCham230xc125900" x="2100" y="2100" z="6"/>
      436     <box lunit="mm" name="RPCFoam0xc21f3f8" x="2110" y="2110" z="75"/>
      437     <box lunit="mm" name="RPCMod0xc13bfd8" x="2170" y="2200" z="78"/>
     ....
     2155     <subtraction name="near-radslab-box-90xcd31ea0">
     2156       <first ref="near-radslab-box-9-box-ChildFornear-radslab-box-90xcd31d48"/>
     2157       <second ref="near-radslab-box-9-sub30xcd31c98"/>
     2158       <position name="near-radslab-box-90xcd31ea0_pos" unit="mm" x="-8000" y="-5000" z="0"/>
     2159       <rotation name="near-radslab-box-90xcd31ea0_rot" unit="deg" x="0" y="0" z="45"/>
     2160     </subtraction>
     2161     <box lunit="mm" name="near_hall_bot0xbf3d718" x="16600" y="10600" z="10300"/>
     2162     <box lunit="mm" name="near_rock_main0xc21d4f0" x="50000" y="50000" z="50000"/>
     2163     <box lunit="mm" name="near_rock_void0xc21d6c8" x="50010" y="50010" z="12010"/>
     2164     <subtraction name="near_rock0xc04ba08">
     2165       <first ref="near_rock_main0xc21d4f0"/>
     2166       <second ref="near_rock_void0xc21d6c8"/>
     2167       <position name="near_rock0xc04ba08_pos" unit="mm" x="0" y="0" z="-19000"/>
     2168     </subtraction>
     2169     <box lunit="mm" name="WorldBox0xc15cf40" x="4800000" y="4800000" z="4800000"/>
     2170   </solids>



MeshIndex is written by GMeshLib
----------------------------------

::

     872 void GGeo::add(const GMesh* mesh)
     873 {
     874     m_meshlib->add(mesh);
     875 }


A : G4DAE route 
~~~~~~~~~~~~~~~~~~

::

    frame #4: 0x00000001018edaa8 libGGeo.dylib`GGeo::add(this=0x000000010ae175d0, mesh=0x0000000111d1bb70) at GGeo.cc:874
    frame #5: 0x000000010063c87b libAssimpRap.dylib`AssimpGGeo::convertMeshes(this=0x00007ffeefbfc288, scene=0x000000010af01a20, gg=0x000000010ae175d0, (null)="") at AssimpGGeo.cc:811
    frame #6: 0x000000010063a94e libAssimpRap.dylib`AssimpGGeo::convert(this=0x00007ffeefbfc288, ctrl="") at AssimpGGeo.cc:192
    frame #7: 0x000000010063a6df libAssimpRap.dylib`AssimpGGeo::load(ggeo=0x000000010ae175d0) at AssimpGGeo.cc:176
    frame #8: 0x00000001018eacdc libGGeo.dylib`GGeo::loadFromG4DAE(this=0x000000010ae175d0) at GGeo.cc:594
    frame #9: 0x00000001018ea911 libGGeo.dylib`GGeo::loadGeometry(this=0x000000010ae175d0) at GGeo.cc:554
    frame #10: 0x00000001005eb942 libOpticksGeo.dylib`OpticksGeometry::loadGeometryBase(this=0x000000010ae16c90) at OpticksGeometry.cc:140
    frame #11: 0x00000001005eb064 libOpticksGeo.dylib`OpticksGeometry::loadGeometry(this=0x000000010ae16c90) at OpticksGeometry.cc:89
    frame #12: 0x00000001005ef3f2 libOpticksGeo.dylib`OpticksHub::loadGeometry(this=0x000000010ae12ab0) at OpticksHub.cc:395
    frame #13: 0x00000001005ee2c2 libOpticksGeo.dylib`OpticksHub::init(this=0x000000010ae12ab0) at OpticksHub.cc:176
    frame #14: 0x00000001005ee1a5 libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010ae12ab0, ok=0x000000010ae00000) at OpticksHub.cc:158
    frame #15: 0x00000001005ee3cd libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010ae12ab0, ok=0x000000010ae00000) at OpticksHub.cc:157
    frame #16: 0x00000001000d3d74 libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe9a8, argc=4, argv=0x00007ffeefbfea80, argforced=0x0000000000000000) at OKMgr.cc:44
    frame #17: 0x00000001000d41bb libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe9a8, argc=4, argv=0x00007ffeefbfea80, argforced=0x0000000000000000) at OKMgr.cc:52
    frame #18: 0x000000010000b995 OKTest`main(argc=4, argv=0x00007ffeefbfea80) at OKTest.cc:13

::

    (lldb) f 6
    frame #6: 0x000000010063a94e libAssimpRap.dylib`AssimpGGeo::convert(this=0x00007ffeefbfc288, ctrl="") at AssimpGGeo.cc:192
       189 	    m_ggeo->afterConvertMaterials(); 
       190 	
       191 	    convertSensors( m_ggeo ); 
    -> 192 	    convertMeshes(scene, m_ggeo, ctrl);
       193 	
       194 	    convertStructure(m_ggeo);
       195 	
    (lldb) 


* presumably the mesh order is just that from the COLLADA 



B : Direct route is adding meshes during the structure traverse : NEED TO MOVE THIS 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff74570b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7473b080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff744cc1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff744941ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010b3c7aa8 libGGeo.dylib`GGeo::add(this=0x000000010e78f260, mesh=0x00000001141cbaf0) at GGeo.cc:874
        frame #5: 0x0000000106bb93c3 libExtG4.dylib`X4PhysicalVolume::convertSolid(this=0x00007ffeefbfe4b0, lvIdx=248, mh=0x00000001141caab0, nd=0x00000001141c7e20, solid=0x000000011529f2c0) at X4PhysicalVolume.cc:725
        frame #6: 0x0000000106bb7ac9 libExtG4.dylib`X4PhysicalVolume::convertNode(this=0x00007ffeefbfe4b0, pv=0x00000001125d9f30, parent=0x0000000000000000, depth=0, pv_p=0x0000000000000000, recursive_select=0x00007ffeefbfd8b3) at X4PhysicalVolume.cc:606
        frame #7: 0x0000000106bb6dad libExtG4.dylib`X4PhysicalVolume::convertTree_r(this=0x00007ffeefbfe4b0, pv=0x00000001125d9f30, parent=0x0000000000000000, depth=0, parent_pv=0x0000000000000000, recursive_select=0x00007ffeefbfd8b3) at X4PhysicalVolume.cc:494
        frame #8: 0x0000000106bb4a2a libExtG4.dylib`X4PhysicalVolume::convertStructure(this=0x00007ffeefbfe4b0) at X4PhysicalVolume.cc:481
        frame #9: 0x0000000106bb36db libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe4b0) at X4PhysicalVolume.cc:142
        frame #10: 0x0000000106bb351b libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe4b0, ggeo=0x000000010e78f260, top=0x00000001125d9f30) at X4PhysicalVolume.cc:124
        frame #11: 0x0000000106bb2d25 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe4b0, ggeo=0x000000010e78f260, top=0x00000001125d9f30) at X4PhysicalVolume.cc:118
        frame #12: 0x000000010001492f OKX4Test`main(argc=1, argv=0x00007ffeefbfeaa0) at OKX4Test.cc:89
        frame #13: 0x00007fff74420015 libdyld.dylib`start + 1
    (lldb) 


Need to move convertSolid into a new convertSolids done prior to convertStructure.


::

    605      if(mh->csgnode == NULL)
    606      {
    607          convertSolid( lvIdx, mh, nd, solid);
    608      }



