some-new-fails.txt
======================


Linux 2018/7/16 monday 10/337 fails, mainly in cfg4
----------------------------------------------------


::

    totals  10  / 337 

    FAILS:
      1  /22  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.41   
      2  /22  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.40   
      3  /22  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    0.98   
      4  /22  Test #4  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    0.89   
      5  /22  Test #5  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.87   
      6  /22  Test #6  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.92   
      17 /22  Test #17 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    0.93   
      19 /22  Test #19 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.40   
      22 /22  Test #22 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.89   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    1.06   

All look to be from same cause of a material named Bialkali with no associated sensor surface.
Take that over to :doc:`bialkali-material-with-no-associated-sensor-surface`



macos 2018/7/16 monday
-----------------------

::

    o ; om- ; om-test

    FAILS:
      37 /49  Test #37 : GGeoTest.GMakerTest                           ***Exception: Child aborted    0.81   

      ## skipped the assert for changing parent links, to observe occurrence

      8  /10  Test #8  : ExtG4Test.X4PhysicalVolumeTest                ***Exception: Child aborted    0.03   

      ## avoided by adhoc switch from glisur to unified in OpNovice

      1  /3   Test #1  : AssimpRapTest.AssimpRapTest                   ***Exception: SegFault         1.04   
      3  /3   Test #3  : AssimpRapTest.AssimpGGeoTest                  ***Exception: SegFault         0.94   
      3  /3   Test #3  : OpticksGeoTest.OpenMeshRapTest                ***Exception: SegFault         0.95   

      ## fixed with Opticks::setGeocache(false) in the tests



FIXED : AssimpRapTest + AssimpGGeoTest + OpenMeshRapTest : failing from lack of libs in GGeo 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fix was to force a full instanciation, not a load from geocache, using 
Opticks::setGeocache in the tests.
This corresponds to the environment of loading from G4DAE that the 
test is exercising so it is appropriate anyhow.


::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x50)
      * frame #0: 0x000000010120869c libGGeo.dylib`GPropertyLib::getStandardDomain(this=0x0000000000000000) at GPropertyLib.cc:148
        frame #1: 0x00000001000d995c libAssimpRap.dylib`AssimpGGeo::convertMaterials(this=0x00007ffeefbfd860, scene=0x0000000104306f30, gg=0x0000000104305270, query="") at AssimpGGeo.cc:391
        frame #2: 0x00000001000d96fd libAssimpRap.dylib`AssimpGGeo::convert(this=0x00007ffeefbfd860, ctrl="") at AssimpGGeo.cc:187
        frame #3: 0x00000001000d94cf libAssimpRap.dylib`AssimpGGeo::load(ggeo=0x0000000104305270) at AssimpGGeo.cc:175
        frame #4: 0x00000001012ab76c libGGeo.dylib`GGeo::loadFromG4DAE(this=0x0000000104305270) at GGeo.cc:578
        frame #5: 0x000000010000606b AssimpRapTest`main(argc=1, argv=0x00007ffeefbfea88) at AssimpRapTest.cc:89
        frame #6: 0x00007fff74420015 libdyld.dylib`start + 1
        frame #7: 0x00007fff74420015 libdyld.dylib`start + 1
    (lldb) 

::

    (lldb) f 2
    frame #2: 0x00000001000db6fd libAssimpRap.dylib`AssimpGGeo::convert(this=0x00007ffeefbfe338, ctrl="") at AssimpGGeo.cc:187
       184 	
       185 	    const aiScene* scene = m_tree->getScene();
       186 	
    -> 187 	    convertMaterials(scene, m_ggeo, ctrl );
       188 	    convertSensors( m_ggeo ); 
       189 	    convertMeshes(scene, m_ggeo, ctrl);
       190 	

    (lldb) f 1
    frame #1: 0x00000001000db95c libAssimpRap.dylib`AssimpGGeo::convertMaterials(this=0x00007ffeefbfe338, scene=0x0000000104604c90, gg=0x0000000104603520, query="") at AssimpGGeo.cc:391
       388 	             ;
       389 	
       390 	    //GDomain<float>* standard_domain = gg->getBoundaryLib()->getStandardDomain(); 
    -> 391 	    GDomain<float>* standard_domain = gg->getBndLib()->getStandardDomain(); 
       392 	
       393 	
       394 	    for(unsigned int i = 0; i < scene->mNumMaterials; i++)
    (lldb) 





AVOIDED : X4PhysicalVolumeTest failing due to use of glisur model AirSurface in OpNovice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Avoided by adhoc switch from glisur to unified


::

    (lldb) f 10
    frame #10: 0x000000010010bdd3 libExtG4.dylib`X4PhysicalVolume::convertSurfaces(this=0x00007ffeefbfe760) at X4PhysicalVolume.cc:182
       179 	    X4LogicalBorderSurfaceTable::Convert(m_slib);
       180 	    size_t num_lbs = m_slib->getNumSurfaces() ; 
       181 	
    -> 182 	    X4LogicalSkinSurfaceTable::Convert(m_slib);
       183 	    size_t num_sks = m_slib->getNumSurfaces() - num_lbs ; 
       184 	
       185 	    LOG(info) << "convertSurfaces"

    (lldb) p num_lbs
    (size_t) $0 = 1


GMakerTest
~~~~~~~~~~~

* just skip the changing parent links assert to see if this is just a problem from the nnode::Tests


::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff74570b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7473b080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff744cc1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff744941ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001007a6b64 libNPY.dylib`nnode::Set_parent_links_r(node=0x00000001033d6100, parent=0x00000001033d6660) at NNode.cpp:1258
        frame #5: 0x00000001007a6b9d libNPY.dylib`nnode::Set_parent_links_r(node=0x00000001033d6660, parent=0x0000000000000000) at NNode.cpp:1263
        frame #6: 0x000000010086f8f3 libNPY.dylib`NCSG::FromNode(root=0x00000001033d6660, config=0x00000001033d7300, soIdx=0, lvIdx=0) at NCSG.cpp:1610
        frame #7: 0x00000001000051a3 GMakerTest`GMakerTest::makeFromCSG(this=0x00007ffeefbfe888) at GMakerTest.cc:82
        frame #8: 0x00000001000056dc GMakerTest`main(argc=1, argv=0x00007ffeefbfea90) at GMakerTest.cc:111
        frame #9: 0x00007fff74420015 libdyld.dylib`start + 1

    (lldb) f 8
    frame #8: 0x00000001000056dc GMakerTest`main(argc=1, argv=0x00007ffeefbfea90) at GMakerTest.cc:111
       108 	    GMakerTest tst(&ok, blib);
       109 	
       110 	    tst.makeSphere();
    -> 111 	    tst.makeFromCSG();
       112 	
       113 	}
       114 	

    (lldb) f 7
    frame #7: 0x00000001000051a3 GMakerTest`GMakerTest::makeFromCSG(this=0x00007ffeefbfe888) at GMakerTest.cc:82
       79  	        unsigned soIdx = 0 ; 
       80  	        unsigned lvIdx = 0 ; 
       81  	
    -> 82  	        NCSG* csg = NCSG::FromNode( n, config, soIdx, lvIdx );
       83  	
       84  	        csg->setMeta<std::string>("poly", "IM");
       85  	

    (lldb) f 6
    frame #6: 0x000000010086f8f3 libNPY.dylib`NCSG::FromNode(root=0x00000001033d6660, config=0x00000001033d7300, soIdx=0, lvIdx=0) at NCSG.cpp:1610
       1607	
       1608	NCSG* NCSG::FromNode(nnode* root, const NSceneConfig* config, unsigned soIdx, unsigned lvIdx )
       1609	{
    -> 1610	    nnode::Set_parent_links_r(root, NULL);
       1611	
       1612	    root->set_treeidx(lvIdx) ;  // without this no nudging is done
       1613	
    (lldb) 







GPU Workstation 
---------------------

::
 
   o ; om- ; om-test

    FAILS:
      37 /49  Test #37 : GGeoTest.GMakerTest                           ***Exception: Child aborted    1.43   
      8  /10  Test #8  : ExtG4Test.X4PhysicalVolumeTest                ***Exception: Child aborted    0.12   
      1  /3   Test #1  : AssimpRapTest.AssimpRapTest                   ***Exception: SegFault         0.97   
      3  /3   Test #3  : AssimpRapTest.AssimpGGeoTest                  ***Exception: SegFault         0.94   
      3  /3   Test #3  : OpticksGeoTest.OpenMeshRapTest                ***Exception: SegFault         0.95   


      13 /18  Test #13 : OptiXRapTest.bufferTest                       ***Exception: Child aborted    0.19   
      14 /18  Test #14 : OptiXRapTest.OEventTest                       ***Exception: Child aborted    0.45   

      1  /22  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.43   
      2  /22  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.42   
      3  /22  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    0.92   
      4  /22  Test #4  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    0.88   
      5  /22  Test #5  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.87   
      6  /22  Test #6  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.93   
      17 /22  Test #17 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    0.93   
      19 /22  Test #19 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.38   
      22 /22  Test #22 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.92   

              ## last ~half are from lack of surf ?

      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    1.07   




