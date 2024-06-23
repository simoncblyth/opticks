torus_assert
================


::


    (gdb) f 12
    #12 0x00007fffcd314e14 in U4Tree::initSolid (this=0xaf31690, lv=0xaeccc80) at /data/blyth/opticks_Debug/include/U4/U4Tree.h:574
    574     initSolid(so, lvid); 
    (gdb) f 11
    #11 0x00007fffcd314ebc in U4Tree::initSolid (this=0xaf31690, so=0xaecacd0, lvid=136) at /data/blyth/opticks_Debug/include/U4/U4Tree.h:606
    606     sn* root = U4Solid::Convert(so, lvid, d );  
    (gdb) f 10
    #10 0x00007fffcd3109bc in U4Solid::Convert (solid=0xaecacd0, lvid=136, depth=0, level=-1) at /data/blyth/opticks_Debug/include/U4/U4Solid.h:360
    360     U4Solid so(solid, lvid, depth, level ); 
    (gdb) f 9
    #9  0x00007fffcd310a99 in U4Solid::U4Solid (this=0x7fffffff26f0, solid_=0xaecacd0, lvid_=136, depth_=0, level_=-1) at /data/blyth/opticks_Debug/include/U4/U4Solid.h:382
    382     init() ; 
    (gdb) f 8
    #8  0x00007fffcd310bde in U4Solid::init (this=0x7fffffff26f0) at /data/blyth/opticks_Debug/include/U4/U4Solid.h:397
    397     init_Tree() ; 
    (gdb) f 7
    #7  0x00007fffcd310f45 in U4Solid::init_Tree (this=0x7fffffff26f0) at /data/blyth/opticks_Debug/include/U4/U4Solid.h:475
    475         root->postconvert(lvid); 
    (gdb) f 6
    #6  0x00007fffcd2fd514 in sn::postconvert (this=0xb4d7490, lvid=136) at /data/blyth/opticks_Debug/include/SysRap/sn.h:4313
    4313        setAABB_TreeFrame_All();  
    (gdb) f 5
    #5  0x00007fffcd2fd2b2 in sn::setAABB_TreeFrame_All (this=0xb4d7490) at /data/blyth/opticks_Debug/include/SysRap/sn.h:4272
    4272            _p->setAABB_LeafFrame() ; 
    (gdb) f 4
    #4  0x00007fffcd2fd17c in sn::setAABB_LeafFrame (this=0xb4d7490) at /data/blyth/opticks_Debug/include/SysRap/sn.h:4219
    4219            assert(0);
    (gdb) 
    

::

     385 inline void U4Solid::init()
     386 {
     387     if(level > 0) std::cerr
     388         <<  ( depth == 0 ? "\n" : "" )
     389         << "[U4Solid::init " << brief()
     390         << " " << ( depth == 0 ? brief_KEY : "" )
     391         << " " << ( depth == 0 ? name : "" )
     392         << std::endl
     393         ;
     394 
     395     init_Constituents();
     396     init_Check();
     397     init_Tree() ;
     398 
     399     if(level > 0) std::cerr << "]U4Solid::init " << brief() << std::endl ;
     400 }
     401 

