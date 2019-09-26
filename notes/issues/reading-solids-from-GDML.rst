reading-solids-from-GDML
=========================


Issue
-------

For debugging solids passing around GDML containing just 
solids is convenient.   Need a way to parse such incomplete 
GDML files into G4VSolid


G4GDMLReadSolids
-------------------


* gets first solid within solidsElement then sibling iterate

::
           
    2690 void G4GDMLReadSolids::SolidsRead(const xercesc::DOMElement* const solidsElement)
    2691 {
    2692 #ifdef G4VERBOSE
    2693    G4cout << "G4GDML: Reading solids..." << G4endl;
    2694 #endif
    2695    for (xercesc::DOMNode* iter = solidsElement->getFirstChild();
    2696         iter != 0; iter = iter->getNextSibling())
    2697    {
    2698       if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE)  { continue; }
    2699 
    2700       const xercesc::DOMElement* const child
    2701             = dynamic_cast<xercesc::DOMElement*>(iter);
    2702       if (!child)
    2703       {
    2704         G4Exception("G4GDMLReadSolids::SolidsRead()",
    2705                     "InvalidRead", FatalException, "No child found!");
    2706         return;
    2707       }
    2708       const G4String tag = Transcode(child->getTagName());
    2709       if (tag=="define") { DefineRead(child);  }  else
    2710       if (tag=="box")    { BoxRead(child); } else
    2711       if (tag=="cone")   { ConeRead(child); } else



Curious the resulting solid just gets instanciated with no holding onto pointers visible::

     180 void G4GDMLReadSolids::BoxRead(const xercesc::DOMElement* const boxElement)
     181 {
     ...
     227    new G4Box(name,x,y,z);
     228 }


Gets held in G4SolidStore::

     60 G4VSolid::G4VSolid(const G4String& name)
     61   : fshapeName(name)
     62 {
     63     kCarTolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
     64 
     65     // Register to store
     66     //
     67     G4SolidStore::GetInstance()->Register(this);
     68 }
     69 
     70 //////////////////////////////////////////////////////////////////////////
     71 //
     72 // Copy constructor
     73 //
     74 
     75 G4VSolid::G4VSolid(const G4VSolid& rhs)
     76   : kCarTolerance(rhs.kCarTolerance), fshapeName(rhs.fshapeName)
     77 {
     78     // Register to store
     79     //
     80     G4SolidStore::GetInstance()->Register(this);
     81 }






::


    [blyth@localhost extg4]$ gdb --args X4GDMLParserTest $HOME/Opticks_install_guide/x376.gdml
    GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-114.el7
    Copyright (C) 2013 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
    and "show warranty" for details.
    This GDB was configured as "x86_64-redhat-linux-gnu".
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>...
    Reading symbols from /home/blyth/local/opticks/lib/X4GDMLParserTest...done.
    (gdb) r
    Starting program: /home/blyth/local/opticks/lib/X4GDMLParserTest /home/blyth/Opticks_install_guide/x376.gdml
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    G4GDML: Reading '/home/blyth/Opticks_install_guide/x376.gdml'...
    G4GDML: Reading solids...
    G4GDML: Reading '/home/blyth/Opticks_install_guide/x376.gdml' done!
    2019-09-26 20:31:08.367 INFO  [72929] [X4SolidStore::Dump@34]  num_solid 29
    2019-09-26 20:31:08.367 INFO  [72929] [X4SolidStore::Dump@38] 0x6fefe0
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x6fd310
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x6ff100
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x721590
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x721650
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x7217b0
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x721870
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x721a10
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x721ad0
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x721c70
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x721d30
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x721ed0
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x721f90
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x722130
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x7221f0
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x722390
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x722450
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x7225f0
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x7226b0
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x722850
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x722910
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x722ab0
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x722b70
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x722d10
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x722dd0
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x722fb0
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x723070
    2019-09-26 20:31:08.368 INFO  [72929] [X4SolidStore::Dump@38] 0x723210
    2019-09-26 20:31:08.369 INFO  [72929] [X4SolidStore::Dump@38] 0x7232d0

    Program received signal SIGABRT, Aborted.
    0x00007fffeb47a207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffeb47a207 in raise () from /lib64/libc.so.6
    #1  0x00007fffeb47b8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffeff3ef8b in G4Exception (originOfException=0x7ffff601c9df "G4GDMLWriteSolids::AddSolid()", exceptionCode=0x7ffff601c9d4 "WriteError", severity=FatalException, description=0x70a548 "Unknown solid: placedB; Type: G4DisplacedSolid")
            at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/global/management/src/G4Exception.cc:100
    #3  0x00007ffff5fcaba4 in G4GDMLWriteSolids::AddSolid (this=0x6f41f0, solidPtr=0x7232d0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/persistency/gdml/src/G4GDMLWriteSolids.cc:1249
    #4  0x00007ffff7b8c449 in X4GDMLWriteStructure::add (this=0x6f41f0, solid=0x7232d0) at /home/blyth/opticks/extg4/X4GDMLWriteStructure.cc:88
    #5  0x00007ffff7b8c101 in X4GDMLWriteStructure::write (this=0x6f41f0, solid=0x7232d0, path=0x0) at /home/blyth/opticks/extg4/X4GDMLWriteStructure.cc:43
    #6  0x00007ffff7b8b8cc in X4GDMLParser::write (this=0x7fffffffd640, solid=0x7232d0, path=0x0) at /home/blyth/opticks/extg4/X4GDMLParser.cc:96
    #7  0x00007ffff7b8b50c in X4GDMLParser::Write (solid=0x7232d0, path=0x0, refs=false) at /home/blyth/opticks/extg4/X4GDMLParser.cc:43
    #8  0x0000000000404125 in test_read_solid (path=0x7fffffffdde0 "/home/blyth/Opticks_install_guide/x376.gdml") at /home/blyth/opticks/extg4/tests/X4GDMLParserTest.cc:125
    #9  0x0000000000404390 in main (argc=2, argv=0x7fffffffd988) at /home/blyth/opticks/extg4/tests/X4GDMLParserTest.cc:145
        (gdb) f 3
    #3  0x00007ffff5fcaba4 in G4GDMLWriteSolids::AddSolid (this=0x6f41f0, solidPtr=0x7232d0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/persistency/gdml/src/G4GDMLWriteSolids.cc:1249
    1249                     FatalException, error_msg);
    (gdb) list
    1244       else
    1245       {
    1246         G4String error_msg = "Unknown solid: " + solidPtr->GetName()
    1247                            + "; Type: " + solidPtr->GetEntityType();
    1248         G4Exception("G4GDMLWriteSolids::AddSolid()", "WriteError",
    1249                     FatalException, error_msg);
    1250       }
    1251    }
    (gdb) quit
    A debugging session is active.

        Inferior 1 [process 72929] will be killed.

    Quit anyway? (y or n) y
    [blyth@localhost extg4]$ 
    [blyth@localhost extg4]$ 
    [blyth@localhost extg4]$ grep G4DisplacedSolid *.cc
    X4Entity.cc:    n.push_back("G4DisplacedSolid")     ; t.push_back(_G4DisplacedSolid)      ;
    X4Solid.cc:    case _G4DisplacedSolid    : convertDisplacedSolid()        ; break ; 
    X4Solid.cc:      if (G4DisplacedSolid* disp = dynamic_cast<G4DisplacedSolid*>(*pp))
    X4Solid.cc:    const G4DisplacedSolid* const disp = static_cast<const G4DisplacedSolid*>(m_solid);
    X4Solid.cc:    assert( dynamic_cast<G4DisplacedSolid*>(moved) == NULL ); // only a single displacement is handled
    X4Solid.cc:    bool is_left_displaced = dynamic_cast<G4DisplacedSolid*>(left) != NULL ;
    X4Solid.cc:    bool is_right_displaced = dynamic_cast<G4DisplacedSolid*>(right) != NULL ;
    X4Solid.cc:        const G4DisplacedSolid* const disp = static_cast<const G4DisplacedSolid*>(right);
    X4Transform3D.cc:#include "G4DisplacedSolid.hh"
    X4Transform3D.cc:glm::mat4 X4Transform3D::GetDisplacementTransform(const G4DisplacedSolid* const disp)
    [blyth@localhost extg4]$ 
    [blyth@localhost extg4]$ 

