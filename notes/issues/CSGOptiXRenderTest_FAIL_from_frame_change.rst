CSGOptiXRenderTest_FAIL_from_frame_change
=============================================


FIXED with extra sframe::prepare::


    [blyth@localhost tests]$ source $HOME/.opticks/GEOM/GEOM.sh
    [blyth@localhost tests]$ CSGOptiXRenderTest
    Segmentation fault (core dumped)
    [blyth@localhost tests]$ gdb CSGOptiXRenderTest
    GNU gdb (GDB) 12.1
    Copyright (C) 2022 Free Software Foundation, Inc.
    ...
    Thread 1 "CSGOptiXRenderT" received signal SIGSEGV, Segmentation fault.
    0x00007ffff7e7501a in sframe::spawn_lite (this=0x7fffffff0ae0) at /home/blyth/junotop/ExternalLibs/opticks/head/include/SysRap/sframe.h:251
    251     l.m2w = tr_m2w->t ;  
    (gdb) bt
    #0  0x00007ffff7e7501a in sframe::spawn_lite (this=0x7fffffff0ae0) at /home/blyth/junotop/ExternalLibs/opticks/head/include/SysRap/sframe.h:251
    #1  0x00007ffff7e5158e in CSGOptiX::setFrame (this=0x9c4f200, frs=0x7ffff7f7ae5b "-1") at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:805
    #2  0x00007ffff7e5144f in CSGOptiX::setFrame (this=0x9c4f200) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:796
    #3  0x00007ffff7e50985 in CSGOptiX::initRender (this=0x9c4f200) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:590
    #4  0x00007ffff7e4f4d9 in CSGOptiX::init (this=0x9c4f200) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:472
    #5  0x00007ffff7e4efe7 in CSGOptiX::CSGOptiX (this=0x9c4f200, foundry_=0x682f980) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:443
    #6  0x00007ffff7e4e68d in CSGOptiX::Create (fd=0x682f980) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:365
    #7  0x000000000040f4ca in CSGOptiXRenderTest::CSGOptiXRenderTest (this=0x7fffffff1bb0) at /home/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:85
    #8  0x000000000040fd8e in main (argc=1, argv=0x7fffffff2128) at /home/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:176
    (gdb) f 0
    #0  0x00007ffff7e7501a in sframe::spawn_lite (this=0x7fffffff0ae0) at /home/blyth/junotop/ExternalLibs/opticks/head/include/SysRap/sframe.h:251
    251     l.m2w = tr_m2w->t ;  
    (gdb) p tr_m2w
    $1 = (Tran<double> *) 0x0
    (gdb) 


Change to different CSGFoundry::getFrame avoids this as the prepare is run::

    void CSGOptiX::setFrame(const char* frs)
     {
         LOG(LEVEL) << " frs " << frs ; 
    -    sframe fr ; 
    -    foundry->getFrame(fr, frs) ; 
    -
    +    sframe fr = foundry->getFrame(frs) ; 
         sfr lfr = fr.spawn_lite(); 
         setFrame(lfr); 

