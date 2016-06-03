High Level Refactor
=====================

Objective
------------

Op and G4 pathways with the same high-level API 

Approach
---------

* common base OpticksEngine
* consistent externalized event handling, allow multiple g4gun firing yielding multiple events 

* keep in mind : what needs to be done only at startup, and only at each event

* streamlining to make cfg4-/CG4Test clean and simple
* streamlining to make ggv-/GGeoView clean and simple
* most of GGeoView is geometry setup, need a GGeoManager to 
  mop this up

* remove duplication between pathways
* move infrastructure like GCache/OpticksResource into Opticks


Move GCache/OpticksResource inside Opticks ?
---------------------------------------------- 

* how does CG4 use GCache ? Can it use OpticksResource directly ?

* move from GCache to OpticksResource held up only by GColors use of GBuffer

  * migrating GColors from GBuffer to NPY would allow GColors to become OpticksColors
    and then the GCache could be replaced by OpticksResource  


Tests Commands To Run Whilst Refactoring
------------------------------------------


G4 running::

    ## G4 test running starting from genstep, which means non-dynamic record/photon/sequence collection

    ggv-pmt-test --cfg4  --dbg                ## geant4 with test PMT in box geometry

    ggv-pmt-test --cfg4  --dbg --load         ## loading and vizualizing into GGeoView

    
    ## G4 gdml geometry running starting from nothing, dynamic collection

    ggv-g4gun --dbg 

    ggv-g4gun --dbg  --load

        ## photon selection menu item doesnt show up in GUI
 

Op running::

    ## Op full geometry running from torch genstep 

    op 




[FIXED] OP record vis broken by OpticksEvent simplifications
----------------------------------------------------------------

Using::

  op --save 

And checking the saved evt with npy-/torchevt.py it looks OK.

Viz seems fixed by:
  
* modified OContext::createIOBuffer to handle non-flat records using NPYBase::getNumQuads
  (however suspect this setting does nothing, as the buffer is handed to OptiX via OpenGL buffer id)

* change glBuffer setup in Rdr/NPYBase/ViewNPY to handle count/item dimension boundary 
  of either 

  * 1: the default, 1st dimension is count the rest is the item 
  * 2: structured records, 1st*2nd dim is count, the rest is the item

* but colors are funny, see below 


[FIXED] Left field flakiness
------------------------------

Probably from changes to some fundamental NPY headers, that were not reflected in compiled versions.

Flaky segv::

    [2016-Jun-02 20:26:34.667548]:info: OScintillatorLib::makeReemissionTexture  nx 4096 ny 1 ni 2 nj 4096 nk 1 step 0.000244141 empty false
    [2016-Jun-02 20:26:34.667710]:info: OPropertyLib::makeTexture bufShape 2,4096,1 numBytes 32768 nx 4096 ny 1 empty false
    /Users/blyth/env/bin/op.sh: line 372: 81353 Segmentation fault: 11  /usr/local/opticks/bin/GGeoView

::

    (lldb) bt

        frame #12: 0x00000001025d0cb7 liboptix.1.dylib`___lldb_unnamed_function258$$liboptix.1.dylib + 279
        frame #13: 0x00007fff8805f63c libc++.1.dylib`std::__1::basic_ostream<char, std::__1::char_traits<char> >::operator<<(std::__1::basic_streambuf<char, std::__1::char_traits<char> >*) + 108
        frame #14: 0x00000001028437b2 liboptix.1.dylib`___lldb_unnamed_function4143$$liboptix.1.dylib + 594
        frame #15: 0x000000010263a884 liboptix.1.dylib`___lldb_unnamed_function972$$liboptix.1.dylib + 212
        frame #16: 0x0000000102594001 liboptix.1.dylib`rtProgramCreateFromPTXFile + 545
        frame #17: 0x0000000103510fbc libOptiXRap.dylib`optix::ContextObj::createProgramFromPTXFile(this=0x000000011fb58f80, filename=0x00007fff5fbfcf80, program_name=0x00007fff5fbfcf68) + 620 at optixpp_namespace.h:2166
        frame #18: 0x000000010350f3a8 libOptiXRap.dylib`OConfig::createProgram(this=0x00000001135a74d0, filename=0x00000001035db32b, progname=0x00000001035db33f) + 2120 at OConfig.cc:30
        frame #19: 0x00000001034f1850 libOptiXRap.dylib`OContext::createProgram(this=0x00000001135a16d0, filename=0x00000001035db32b, progname=0x00000001035db33f) + 48 at OContext.cc:89
        frame #20: 0x00000001034fc77a libOptiXRap.dylib`OGeo::makeTriangulatedGeometry(this=0x00000001135aa9b0, mm=0x00000001115a06e0) + 138 at OGeo.cc:520
        frame #21: 0x00000001034fa93e libOptiXRap.dylib`OGeo::makeGeometry(this=0x00000001135aa9b0, mergedmesh=0x00000001115a06e0) + 174 at OGeo.cc:410
        frame #22: 0x00000001034f9fbf libOptiXRap.dylib`OGeo::convert(this=0x00000001135aa9b0) + 927 at OGeo.cc:163
        frame #23: 0x00000001043f0e2f libOpticksOp.dylib`OpEngine::prepareOptiX(this=0x0000000109940f10) + 5199 at OpEngine.cc:94
        frame #24: 0x000000010454234e libGGeoViewLib.dylib`App::prepareOptiX(this=0x00007fff5fbfe388) + 366 at App.cc:1130
        frame #25: 0x000000010000b902 GGeoView`main(argc=2, argv=0x00007fff5fbfe4e0) + 626 at main.cc:43
        frame #26: 0x00007fff89e755fd libdyld.dylib`start + 1


Full build seems to fix::

    opticks-wipe
    opticks-full


Nope its back::

    simon:~ blyth$ op -c
    ...
    [2016-Jun-03 10:21:38.910721]:info: OPropertyLib::makeTexture bufShape 2,4096,1 numBytes 32768 nx 4096 ny 1 empty false
    [2016-Jun-03 10:21:38.911186]:info: OpEngine::prepareOptiX (OSourceLib)
    [2016-Jun-03 10:21:38.911296]:info: OPropertyLib::makeTexture bufShape 1,1024,1 numBytes 4096 nx 1024 ny 1 empty false
    /Users/blyth/env/bin/op.sh: line 372: 97061 Segmentation fault: 11  /usr/local/opticks/bin/GGeoView -c
    simon:~ blyth$ 


Cannot reproduce. Five invokations without trouble::

    op
    op -c
    op -s 


* maybe an issue with build system dependencies, on rare occasions have trouble getting an update in NPY thru to optickscore 



FIXED: Photon record coloring M key seems wrong 
----------------------------------------------------------

Suspect the offsets are wrong in Rdr::address 

* covered in ggv-/issues/gui_broken_photon_record_colors :doc:`../gui_broken_photon_record_colors`



[FIXED] CFG4 load count mismatch assert
------------------------------------------

* fixed by removing the reshaping on load

::

    simon:geant4_opticks_integration blyth$ 
    simon:geant4_opticks_integration blyth$ ggv-pmt-test --cfg4  --load

    ...


