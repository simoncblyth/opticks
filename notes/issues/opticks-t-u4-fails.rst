opticks-t-u4-fails
====================

Full GEOM : V1J011 : ALL PASSING CURRENTLY
------------------------------------------------

::

    85% tests passed, 5 tests failed out of 33

    Total Test time (real) =  32.41 sec

    The following tests FAILED:
         16 - U4Test.U4VolumeMakerTest (Child aborted)
         22 - U4Test.U4TreeCreateTest (Child aborted)
         23 - U4Test.U4TreeCreateSSimTest (Child aborted)
         25 - U4Test.U4Material_MakePropertyFold_LoadTest (SEGFAULT)
         32 - U4Test.U4NavigatorTest (Child aborted)

    The following tests FAILED:
         16 - U4Test.U4VolumeMakerTest (Child aborted)
         24 - U4Test.U4Material_MakePropertyFold_LoadTest (SEGFAULT)  ## NON-EXISTING FOLD ERROR
         31 - U4Test.U4NavigatorTest (Child aborted)     ## THIS ONE IS EXPECTED TO FAIL


    100% tests passed, 0 tests failed out of 32

    Total Test time (real) =  74.93 sec



::

    u4 ; TESTARG="-R U4TreeCreate" om-test 
    u4 ; TESTARG="-R U4TreeCreateSSim" om-test 

    u4 ; TESTARG="-R U4NavigatorTest" om-test   
    u4 ; TESTARG="-R U4Material_MakePropertyFold_LoadTest" om-test   


Minimal GEOM : RaindropRockAirWater 
-------------------------------------

::

        
    100% tests passed, 0 tests failed out of 32


Added universe wrapper volume to avoid three fails::

    The following tests FAILED:
          9 - U4Test.U4GDMLReadTest (Child aborted)
         20 - U4Test.U4TraverseTest (Child aborted)
         21 - U4Test.U4TreeTest (Child aborted)
    Errors while running CTest

    The following tests FAILED:
          9 - U4Test.U4GDMLReadTest (Child aborted)      
         20 - U4Test.U4TraverseTest (Child aborted)
         21 - U4Test.U4TreeTest (Child aborted)
                 ## same issue : need to fix geometry 
                 ## its too simple : bordersurface doesnt find pv of outer 

         22 - U4Test.U4TreeCreateTest (Child aborted)
         23 - U4Test.U4TreeCreateSSimTest (Child aborted)
         25 - U4Test.U4Material_MakePropertyFold_LoadTest (SEGFAULT)
         32 - U4Test.U4NavigatorTest (SEGFAULT)



    84% tests passed, 5 tests failed out of 32

    Total Test time (real) =   5.78 sec

    The following tests FAILED:
          9 - U4Test.U4GDMLReadTest (Child aborted)
         20 - U4Test.U4TraverseTest (Child aborted)
         21 - U4Test.U4TreeTest (Child aborted)

         22 - U4Test.U4TreeCreateTest (Child aborted)
         23 - U4Test.U4TreeCreateSSimTest (Child aborted)
    Errors while running CTest
    Sat Nov  4 14:57:34 PST 2023


    u4 ; TESTARG="-R U4TreeCreateTest" om-test 




U4TreeCreateTest with minimal GEOM : mt assert
--------------------------------------------------

::

    Assertion failed: (mt), function initMaterials_r, file /Users/blyth/opticks/u4/U4Tree.h, line 383.
    Process 14170 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff54914b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff54914b66 <+10>: jae    0x7fff54914b70            ; <+20>
        0x7fff54914b68 <+12>: movq   %rax, %rdi
        0x7fff54914b6b <+15>: jmp    0x7fff5490bae9            ; cerror_nocancel
        0x7fff54914b70 <+20>: retq   
    Target 0: (U4TreeCreateTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff54914b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff54adf080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff548701ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff548381ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001000c4b5b U4TreeCreateTest`U4Tree::initMaterials_r(this=0x000000010751dae0, pv=0x000000010751cac0) at U4Tree.h:383
        frame #5: 0x00000001000c4af9 U4TreeCreateTest`U4Tree::initMaterials_r(this=0x000000010751dae0, pv=0x000000010751ce50) at U4Tree.h:379
        frame #6: 0x00000001000c4af9 U4TreeCreateTest`U4Tree::initMaterials_r(this=0x000000010751dae0, pv=0x000000010751cec0) at U4Tree.h:379
        frame #7: 0x00000001000c23e3 U4TreeCreateTest`U4Tree::initMaterials(this=0x000000010751dae0) at U4Tree.h:290
        frame #8: 0x00000001000c0106 U4TreeCreateTest`U4Tree::init(this=0x000000010751dae0) at U4Tree.h:249
        frame #9: 0x00000001000bfecf U4TreeCreateTest`U4Tree::U4Tree(this=0x000000010751dae0, st_=0x000000010751baf0, top_=0x000000010751cec0, sid_=0x0000000000000000) at U4Tree.h:235
        frame #10: 0x00000001000bee8d U4TreeCreateTest`U4Tree::U4Tree(this=0x000000010751dae0, st_=0x000000010751baf0, top_=0x000000010751cec0, sid_=0x0000000000000000) at U4Tree.h:234
        frame #11: 0x00000001000542af U4TreeCreateTest`U4Tree::Create(st=0x000000010751baf0, top=0x000000010751cec0, sid=0x0000000000000000) at U4Tree.h:204
        frame #12: 0x0000000100053906 U4TreeCreateTest`main(argc=1, argv=0x00007ffeefbfe7c8) at U4TreeCreateTest.cc:29
        frame #13: 0x00007fff547c4015 libdyld.dylib`start + 1
        frame #14: 0x00007fff547c4015 libdyld.dylib`start + 1
    (lldb) 


HUH: hows that possible, no material ?::

    (lldb) f 4
    frame #4: 0x00000001000c4b5b U4TreeCreateTest`U4Tree::initMaterials_r(this=0x000000010751dae0, pv=0x000000010751cac0) at U4Tree.h:383
       380 	
       381 	    // postorder visit after recursive call  
       382 	    G4Material* mt = lv->GetMaterial() ; 
    -> 383 	    assert(mt);  
       384 	
       385 	    std::vector<const G4Material*>& m = materials ;  
       386 	    if(std::find(m.begin(), m.end(), mt) == m.end()) initMaterial(mt);  
    (lldb) p lv
    (G4LogicalVolume *) $0 = 0x000000010751c530
    (lldb) p lv->GetName()
    (const G4String) $1 = (std::__1::string = "drop_lv")
    (lldb) p mt
    (G4Material *) $2 = 0x0000000000000000
    (lldb) 


Is was was creating the geometry on the fly and using defaults "Rock,Air,Water" so it got 
no materials.


U4GDMLReadTest FAILS with mininimal GEOM  : geometry issue : bordersurface doesnt find pv of outer 
---------------------------------------------------------------------------------------------------------

::

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : ReadError
          issued by : G4GDMLReadStructure::GetPhysvol()
    Referenced physvol 'container_pv0x7fef10d3d9c0' was not found!
    *** Fatal Exception ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------


AHH: I recall seeing this before the issue is cannot have a border surface with the outer 
"world" volume, need to to put a virtual volume on the outside so have 
a pv with which to bordersurface with.



::

    (lldb) f 11
    frame #11: 0x0000000100023ced U4GDMLReadTest`main(argc=1, argv=0x00007ffeefbfe8b0) at U4GDMLReadTest.cc:63
       60  	        << path ; 
       61  	
       62  	
    -> 63  	    const G4VPhysicalVolume* world = U4GDML::Read(path) ;  
       64  	
       65  	    Traverse trv(world);
       66  	
    (lldb) f 10
    frame #10: 0x0000000100024812 U4GDMLReadTest`U4GDML::Read(path_="/Users/blyth/.opticks/GEOM/RaindropRockAirWater/origin.gdml") at U4GDML.h:73
       70  	    if(!exists) return nullptr ; 
       71  	
       72  	    U4GDML g ; 
    -> 73  	    g.read(path); 
       74  	    return g.world ; 
       75  	}
       76  	inline const G4VPhysicalVolume* U4GDML::Read(const char* base, const char* name)
    (lldb) f 9
    frame #9: 0x0000000100031acc U4GDMLReadTest`U4GDML::read(this=0x00007ffeefbfe280, path_="/Users/blyth/.opticks/GEOM/RaindropRockAirWater/origin.gdml") at U4GDML.h:141
       138 	        sdirect::cout_(coutbuf.rdbuf());
       139 	        sdirect::cerr_(cerrbuf.rdbuf());
       140 	
    -> 141 	        parser->Read(path, read_validate);  // noisy code 
       142 	
       143 	    }
       144 	    std::string out = coutbuf.str();
    (lldb) f 8
    frame #8: 0x0000000100031f75 U4GDMLReadTest`G4GDMLParser::Read(this=0x0000000107400070, filename=0x00007ffeefbfdc90, validate=false) at G4GDMLParser.icc:37
       34  	inline 
       35  	void G4GDMLParser::Read(const G4String& filename, G4bool validate)
       36  	{   
    -> 37  	  reader->Read(filename,validate,false,strip);
       38  	  ImportRegions();
       39  	}
       40  	
    (lldb) f 7
    frame #7: 0x00000001013a98a5 libG4persistency.dylib`G4GDMLRead::Read(this=0x0000000107405050, fileName=0x00007ffeefbfdc90, validation=false, isModule=false, strip=false) at G4GDMLRead.cc:447
       444 	      if (tag=="materials") { MaterialsRead(child); } else
       445 	      if (tag=="solids")    { SolidsRead(child);    } else
       446 	      if (tag=="setup")     { SetupRead(child);     } else
    -> 447 	      if (tag=="structure") { StructureRead(child); } else
       448 	      if (tag=="userinfo")  { UserinfoRead(child);  } else
       449 	      if (tag=="extension") { ExtensionRead(child); }
       450 	      else
    (lldb) f 6
    frame #6: 0x00000001013f17e9 libG4persistency.dylib`G4GDMLReadStructure::StructureRead(this=0x0000000107405050, structureElement=0x0000000108903d10) at G4GDMLReadStructure.cc:815
       812 	      }
       813 	      const G4String tag = Transcode(child->getTagName());
       814 	
    -> 815 	      if (tag=="bordersurface") { BorderSurfaceRead(child); } else
       816 	      if (tag=="skinsurface") { SkinSurfaceRead(child); } else
       817 	      if (tag=="volume") { VolumeRead(child); } else
       818 	      if (tag=="assembly") { AssemblyRead(child); } else
    (lldb) f 5
    frame #5: 0x00000001013ea1f6 libG4persistency.dylib`G4GDMLReadStructure::BorderSurfaceRead(this=0x0000000107405050, bordersurfaceElement=0x0000000108906280) at G4GDMLReadStructure.cc:117
       114 	      if (index==0)
       115 	        { pv1 = GetPhysvol(GenerateName(RefRead(child))); index++; } else
       116 	      if (index==1)
    -> 117 	        { pv2 = GetPhysvol(GenerateName(RefRead(child))); index++; } else
       118 	      break;
       119 	   }
       120 	
    (lldb) f 4
    frame #4: 0x00000001013ea5d2 libG4persistency.dylib`G4GDMLReadStructure::GetPhysvol(this=0x0000000107405050, ref=0x00007ffeefbfd258) const at G4GDMLReadStructure.cc:838
       835 	   if (!physvolPtr)
       836 	   {
       837 	     G4String error_msg = "Referenced physvol '" + ref + "' was not found!";
    -> 838 	     G4Exception("G4GDMLReadStructure::GetPhysvol()", "ReadError",
       839 	                 FatalException, error_msg);
       840 	   }
       841 	
    (lldb) 



