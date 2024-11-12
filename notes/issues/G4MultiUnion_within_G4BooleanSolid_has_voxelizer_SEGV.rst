G4MultiUnion_within_G4BooleanSolid_has_voxelizer_SEGV
========================================================


NEXT
-----

* statistical A-B input photon comparison 

  * OIM/OPTICKS_INTEGRATION_MODE:3 


Quick jok-ana check on laptop for input photons targetting PMT visually OK
------------------------------------------------------------------------------

::

    jok-
    jok-grab
    jok-ana


DONE
-----

* DONE : pyvista input photon visual check intersect 
* DONE : check full geometry conversion
* DONE : visualize
* DONE : check again input photon running 



Issue with G4MultiUnion within a boolean
--------------------------------------------

::

    Thread 1 "python" received signal SIGSEGV, Segmentation fault.
    0x00007fffc651c260 in G4Voxelizer::GetCandidatesVoxelArray(CLHEP::Hep3Vector const&, std::vector<int, std::allocator<int> >&, G4SurfBits*) const () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4geometry.so
    (gdb) bt
    #0  0x00007fffc651c260 in G4Voxelizer::GetCandidatesVoxelArray(CLHEP::Hep3Vector const&, std::vector<int, std::allocator<int> >&, G4SurfBits*) const () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4geometry.so
    #1  0x00007fffc644f3d0 in G4MultiUnion::InsideWithExclusion(CLHEP::Hep3Vector const&, G4SurfBits*) const () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4geometry.so
    #2  0x00007fffc644742d in G4DisplacedSolid::Inside(CLHEP::Hep3Vector const&) const () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4geometry.so
    #3  0x00007fffc6455e37 in G4UnionSolid::Inside(CLHEP::Hep3Vector const&) const () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4geometry.so
    #4  0x00007fffc641cbc3 in G4Navigator::LocateGlobalPointAndSetup(CLHEP::Hep3Vector const&, CLHEP::Hep3Vector const*, bool, bool) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4geometry.so
    #5  0x00007fffc77b8ccb in G4Transportation::PostStepDoIt(G4Track const&, G4Step const&) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4processes.so
    #6  0x00007fffcc0ab679 in G4SteppingManager::InvokePSDIP(unsigned long) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #7  0x00007fffcc0aba7b in G4SteppingManager::InvokePostStepDoItProcs() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #8  0x00007fffcc0a92b4 in G4SteppingManager::Stepping() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #9  0x00007fffcc0b487f in G4TrackingManager::ProcessOneTrack(G4Track*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #10 0x00007fffcc0f056d in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #11 0x00007fffbe3f568e in G4SvcRunManager::SimulateEvent (this=0x6554510, i_event=0) at /data/blyth/junotop/junosw/Simulation/DetSimV2/G4Svc/src/G4SvcRunManager.cc:29
    #12 0x00007fffbdabdd3e in DetSimAlg::execute (this=0x6c046d0) at /data/blyth/junotop/junosw/Simulation/DetSimV2/DetSimAlg/src/DetSimAlg.cc:112




WIP : workaround this by providing way to convert a boolean tree with suitable hinting into a list-node rather than requiring a G4MultiUnion 
----------------------------------------------------------------------------------------------------------------------------------------------

HMM: do this after initial conversion into sn.h tree ? 

* but sn nodes have no name, so will use note and add enum::

     418     enum {
     419             NOTE_INCREASE_ZMAX = 0x1 << 0,
     420             NOTE_DECREASE_ZMIN = 0x1 << 1,
     421             HINT_LISTNODE_PRIM = 0x1 << 2 
     422          };
     




::

    142 void
    143 FastenerAcrylicConstruction::makeFastenerLogical() {
    144     int CONFIG = EGet::Get<int>("FastenerAcrylicConstruction__CONFIG", 0) ;
    145     G4MultiUnion* muni = nullptr ;
    146 
    147     if(CONFIG == 1 || CONFIG == 2 )
    148     {
    149         const char* name = CONFIG == 1 ? "muni" : "muni_CSG_DISCONTIGUOUS" ;
    150         muni = new G4MultiUnion(name) ;
    151     }
    152 
    153     G4Tubs *IonRing = new G4Tubs("IonRing",123*mm,206.2*mm,7*mm,0.0*deg,360.0*deg);
    154 
    155     G4Tubs* screw = new G4Tubs("screw",0,13*mm,50.*mm,0.0*deg,360.0*deg);
    156     uni_Addition = IonRing;
    157     for(int i=0;i<8;i++)
    158     {
    159         G4ThreeVector pos(164.*cos(i*pi/4)*mm, 164.*sin(i*pi/4)*mm,-65.0*mm) ;
    160         G4UnionSolid* uni1 = new G4UnionSolid("uni1",uni_Addition, screw, 0, pos);
    161         uni_Addition = uni1;
    162 
    163         if(muni)
    164         {
    165             G4RotationMatrix rot(0, 0, 0);
    166             G4Transform3D tr(rot, pos) ;
    167             muni->AddNode( *screw, tr );
    168         }
    169     }
    170 
    171 
    172     G4VSolid* theSolid = muni == nullptr ?
    173                                   uni_Addition
    174                               :
    175                                   new G4UnionSolid( "uni1", IonRing, muni, 0, G4ThreeVector(0.,0.,0.) )
    176                               ;
    177 
    178    
    179       logicFasteners = new G4LogicalVolume(
    180       theSolid,
    181       Steel,
    182       "lFasteners",
    183       0,
    184       0,





Need to:

1. identify a tree with a union-string of hinted listnode prim
2. find the "joint" node [un] : "first node with LHS not having any hinted prim ?"
3. grab the hinted prim and transforms and form the listnode from them 
4. replace RHS of the "joint" node with the list node
5. delete the extraneous "un" nodes (without deleting their RHS prim)

* it might be easier to grab and reconstruct from scratch ? 


::

    sn::desc pid   18 idx   18 typecode   1 num_node  19 num_leaf  10 maxdepth  9 is_positive_form Y lvid   0 tag un
    sn::render mode 4 TYPETAG
                                                       un       
                                                                
                                                 un       cy    
                                                                
                                           un       cy          
                                                                
                                     un       cy                
                                                                
                               un       cy                      
                                                                
                         un       cy                            
                                                                
                   un       cy                                  
                                                                
            [un]      cy                                        
                                                                
       in       cy                                              
                                                                
    cy    !cy                                                   
                                                                
                          



Implemented sn::CreateSmallerTreeWithListNode following cleanup of sn/s_bb/s_pa/s_tv::

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
        modified:   notes/issues/G4MultiUnion_within_G4BooleanSolid_has_voxelizer_SEGV.rst
        modified:   sysrap/s_bb.h
        modified:   sysrap/s_pa.h
        modified:   sysrap/s_pool.h
        modified:   sysrap/s_tv.h
        modified:   sysrap/sn.h
        modified:   sysrap/tests/sn_test.cc
        modified:   sysrap/tests/sn_test.sh
        modified:   u4/U4Solid.h
        modified:   u4/U4SolidMaker.cc
        modified:   u4/tests/U4SolidTest.cc



This allows getting the translation to create listnodes from a structurally unchanged source solid,
only the names of some prim are changed to provide hints as to which solids should be incorporated
into the listnode within the translated Opticks geometry. 

This allows the voxelization problem with G4MultiUnion within a boolean solid to be avoided. 

::

    P[blyth@localhost opticks]$ git log -n1
    commit 079896e0481eaa3ea9a0b214d88ff93f135ae917 (HEAD -> master, origin/master, origin/HEAD)
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Tue Nov 5 21:21:00 2024 +0800

        enable geometry translation to create smaller trees with listnode using sn::CreateSmallerTreeWithListNode rather than requiring G4MultiUnion in the G4 geometry, to avoid G4 voxelization SEGV
    P[blyth@localhost opticks]$ 




First left field issue : looks to be caused by deepcopy effectively scrubbing the parent pointers of the nodes
------------------------------------------------------------------------------------------------------------------

Try FIX where set_lvid sets the parent links : that seems to work.  


::

    jok-;jok-tds-gdb 



     45610 sid    52398
      45611 sid    52399
    ]]stree::postcreate
    2024-11-06 11:27:20.227 INFO  [202444] [U4Tree::Create@236] ]stree::postcreate
    [Detaching after fork from child process 203836]
    python: /data/blyth/opticks_Debug/include/SysRap/sn.h:3815: static sn* sn::GetLVRoot(int): Assertion `count == 0 || count == 1' failed.

    Thread 1 "python" received signal SIGABRT, Aborted.
    0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6b35a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6b2d1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6b2d252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffc59b0e72 in sn::GetLVRoot (lvid=101) at /data/blyth/opticks_Debug/include/SysRap/sn.h:3815
    #5  0x00007fffc59b12ec in sn::GetLVNodesComplete (nds=std::vector of length 0, capacity 0, lvid=101) at /data/blyth/opticks_Debug/include/SysRap/sn.h:4016
    #6  0x00007fffc59acd32 in CSGImport::importPrim (this=0x1ab3f0e0, primIdx=0, node=...) at /home/blyth/opticks/CSG/CSGImport.cc:304
    #7  0x00007fffc59acb60 in CSGImport::importSolidFactor (this=0x1ab3f0e0, ridx=6, ridx_type=70 'F') at /home/blyth/opticks/CSG/CSGImport.cc:251
    #8  0x00007fffc59abfd9 in CSGImport::importSolid (this=0x1ab3f0e0) at /home/blyth/opticks/CSG/CSGImport.cc:92
    #9  0x00007fffc59abdc1 in CSGImport::import (this=0x1ab3f0e0) at /home/blyth/opticks/CSG/CSGImport.cc:55
    #10 0x00007fffc5908dcb in CSGFoundry::importSim (this=0x1ab3ede0) at /home/blyth/opticks/CSG/CSGFoundry.cc:1696
    #11 0x00007fffc590e3e2 in CSGFoundry::CreateFromSim () at /home/blyth/opticks/CSG/CSGFoundry.cc:3000
    #12 0x00007fffcd2c2469 in G4CXOpticks::setGeometry (this=0xaf3b640, world=0x97b5dc0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:321
    #13 0x00007fffcd2c04a5 in G4CXOpticks::SetGeometry (world=0x97b5dc0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:58
    #14 0x00007fffcd2c0740 in G4CXOpticks::SetGeometry_JUNO (world=0x97b5dc0, sd=0x99a3a80, jpmt=0xaef3090, jlut=0xaf35b80) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:96
    #15 0x00007fffbe3462f9 in LSExpDetectorConstruction_Opticks::Setup (opticksMode=1, world=0x97b5dc0, sd=0x99a3a80, ppd=0x55e7d0, psd=0x6638fd0, pmtscan=0x0)
        at /data/blyth/junotop/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc:46



    #61 0x000000000040108e in _start ()
    (gdb) f 5
    #5  0x00007fffc59b12ec in sn::GetLVNodesComplete (nds=std::vector of length 0, capacity 0, lvid=101) at /data/blyth/opticks_Debug/include/SysRap/sn.h:4016
    4016        const sn* root = GetLVRoot(lvid);  // first sn from pool with requested lvid that is_root
    (gdb) f 4
    #4  0x00007fffc59b0e72 in sn::GetLVRoot (lvid=101) at /data/blyth/opticks_Debug/include/SysRap/sn.h:3815
    3815        assert( count == 0 || count == 1 ); 
    (gdb) p count
    $1 = 4
    (gdb) 



Second shakedown issue : failed to Get some nd 
---------------------------------------------------

::

    jok-;jok-tds-gdb 


::

    ]]stree::postcreate
    2024-11-06 15:33:06.279 INFO  [158185] [U4Tree::Create@236] ]stree::postcreate
    [Detaching after fork from child process 159585]
    [CSGImport::importPrim.dump_LVID:1 node.lvid 101 LVID -1 name uni1 soname uni1 primIdx 0 bn 7 ln(subset of bn) 1 num_sub_total 8
    .CSGImport::importPrim dumping as ln > 0 : solid contains listnode
    python: /data/blyth/opticks_Debug/include/SysRap/sn.h:4593: static void sn::NodeTransformProduct(int, glm::tmat4x4<double>&, glm::tmat4x4<double>&, bool, std::ostream*): Assertion `nd' failed.

    Thread 1 "python" received signal SIGABRT, Aborted.
    0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6b35a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6b2d1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6b2d252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffc59b1973 in sn::NodeTransformProduct (idx=425, t=..., v=..., reverse=false, out=0x0) at /data/blyth/opticks_Debug/include/SysRap/sn.h:4593
    #5  0x00007fffc59b3de1 in stree::get_combined_transform (this=0xaf359c0, t=..., v=..., node=..., nd=0xb4cb4a0, out=0x0) at /data/blyth/opticks_Debug/include/SysRap/stree.h:2655
    #6  0x00007fffc59b4264 in stree::get_combined_tran_and_aabb (this=0xaf359c0, aabb=0x7ffffffef2b0, node=..., nd=0xb4cb4a0, out=0x0) at /data/blyth/opticks_Debug/include/SysRap/stree.h:2710
    #7  0x00007fffc59adc30 in CSGImport::importNode (this=0x1ab3e3c0, nodeOffset=15603, partIdx=3, node=..., nd=0xb4cb4a0) at /home/blyth/opticks/CSG/CSGImport.cc:541
    #8  0x00007fffc59ad230 in CSGImport::importPrim (this=0x1ab3e3c0, primIdx=0, node=...) at /home/blyth/opticks/CSG/CSGImport.cc:387
    #9  0x00007fffc59acb90 in CSGImport::importSolidFactor (this=0x1ab3e3c0, ridx=6, ridx_type=70 'F') at /home/blyth/opticks/CSG/CSGImport.cc:251
    #10 0x00007fffc59ac009 in CSGImport::importSolid (this=0x1ab3e3c0) at /home/blyth/opticks/CSG/CSGImport.cc:92
    #11 0x00007fffc59abdf1 in CSGImport::import (this=0x1ab3e3c0) at /home/blyth/opticks/CSG/CSGImport.cc:55
    #12 0x00007fffc5908dfb in CSGFoundry::importSim (this=0x1ab3e0c0) at /home/blyth/opticks/CSG/CSGFoundry.cc:1696
    #13 0x00007fffc590e412 in CSGFoundry::CreateFromSim () at /home/blyth/opticks/CSG/CSGFoundry.cc:3000
    #14 0x00007fffcd2c2489 in G4CXOpticks::setGeometry (this=0xaf3a9d0, world=0x97b5100) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:321
    #15 0x00007fffcd2c04c5 in G4CXOpticks::SetGeometry (world=0x97b5100) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:58
    #16 0x00007fffcd2c0760 in G4CXOpticks::SetGeometry_JUNO (world=0x97b5100, sd=0x99a2dc0, jpmt=0xaef2420, jlut=0xaf34f10) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:96
    #17 0x00007fffbe3462f9 in LSExpDetectorConstruction_Opticks::Setup (opticksMode=1, world=0x97b5100, sd=0x99a2dc0, ppd=0x55e560, psd=0x66381f0, pmtscan=0x0)
        at /data/blyth/junotop/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc:46
    #18 0x00007fffbe31b07c in LSExpDetectorConstruction::setupOpticks (this=0x95ca850, world=0x97b5100) at /data/blyth/junotop/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc:454
    #19 0x00007fffbe31a91c in LSExpDetectorConstruction::Construct (this=0x95ca850) at /data/blyth/junotop/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc:375
    #20 0x00007fffcc18795e in G4RunManager::InitializeGeometry() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so


::

    (gdb) f 4
    #4  0x00007fffc59b1973 in sn::NodeTransformProduct (idx=425, t=..., v=..., reverse=false, out=0x0) at /data/blyth/opticks_Debug/include/SysRap/sn.h:4593
    4593        assert(nd); 
    (gdb) list
    4588        glm::tmat4x4<double>& v, 
    4589        bool reverse, 
    4590        std::ostream* out)  // static
    4591    {
    4592        sn* nd = Get(idx); 
    4593        assert(nd); 
    4594        nd->getNodeTransformProduct(t,v,reverse,out) ; 
    4595    }
    4596    
    4597    inline std::string sn::DescNodeTransformProduct(
    (gdb) p idx
    $1 = 425
    (gdb) 

Potentially are trying to use stale idx post the deepcopy ?::

    (gdb) f 7 
    #7  0x00007fffc59adc30 in CSGImport::importNode (this=0x1ab3e3c0, nodeOffset=15603, partIdx=3, node=..., nd=0xb4cb4a0) at /home/blyth/opticks/CSG/CSGImport.cc:541
    541     const Tran<double>* tv = leaf ? st->get_combined_tran_and_aabb( aabb, node, nd, nullptr ) : nullptr ; 
    (gdb) p leaf 
    $2 = true
    (gdb) 





    520 CSGNode* CSGImport::importNode(int nodeOffset, int partIdx, const snode& node, const sn* nd)
    521 {
    522     if(nd) assert( node.lvid == nd->lvid );
    523 
    524     int  typecode = nd ? nd->typecode : CSG_ZERO ;
    525     bool leaf = CSG::IsLeaf(typecode) ;
    526 
    527     bool external_bbox_is_expected = CSG::ExpectExternalBBox(typecode);
    528     // CSG_CONVEXPOLYHEDRON, CSG_CONTIGUOUS, CSG_DISCONTIGUOUS, CSG_OVERLAP
    529 
    530     bool expect = external_bbox_is_expected == false ;
    531     LOG_IF(fatal, !expect)
    532         << " NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED "
    533         << " for node of type " << CSG::Name(typecode)
    534         << " nd.lvid " << ( nd ? nd->lvid : -1 )
    535         ;
    536     assert(expect);
    537     if(!expect) std::raise(SIGINT);
    538 
    539     std::array<double,6> bb ;
    540     double* aabb = leaf ? bb.data() : nullptr ;
    541     const Tran<double>* tv = leaf ? st->get_combined_tran_and_aabb( aabb, node, nd, nullptr ) : nullptr ;
    542     unsigned tranIdx = tv ?  1 + fd->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms
    543 
    544     CSGNode* n = fd->addNode();
    545     n->setTypecode(typecode);
    546     n->setBoundary(node.boundary);
    547     n->setComplement( nd ? nd->complement : false );
    548     n->setTransform(tranIdx);
    549     n->setParam_Narrow( nd ? nd->getPA_data() : nullptr );
    550     n->setAABB_Narrow(aabb ? aabb : nullptr  );
    551 
    552     return n ;
    553 }


::


    (gdb) p nd
    $3 = (const sn *) 0xb4cb4a0
    (gdb) p nd->desc()
    $4 = "sn::desc pid  444 idx  425 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 101 tag cy"
    (gdb) p nd->render(sn::PID)
    $5 = "\nsn::desc pid  444 idx  425 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 101 tag cy\nsn::render mode 5 PID\n444   \n      \n      \n      \n\npreorder  sn::desc_order [444 ]\nino"...
    (gdb) p *nd
    $6 = {typecode = 105, complement = 0, lvid = 101, xform = 0x0, param = 0xb4cb560, aabb = 0xb4cb5e0, parent = 0xb4cb3e0, child = std::vector of length 0, capacity 0, depth = 2, note = 0, coincide = 0, label = '\000' <repeats 15 times>, 
      pid = 444, subdepth = 0, static pool = 0xaf33be0, static VERSION = 0, static zero = 0, static Z_EPSILON = 0.001, static UNBOUNDED_DEFAULT_EXTENT = 0, static LEAK = false}
    (gdb) p nd->parent
    $7 = (sn *) 0xb4cb3e0
    (gdb) p *nd->parent
    $8 = {typecode = 2, complement = 0, lvid = 101, xform = 0x0, param = 0x0, aabb = 0x0, parent = 0xb4cb320, child = std::vector of length 2, capacity 2 = {0xb4cb4a0, 0xb4cb660}, depth = 1, note = 0, coincide = 0, 
      label = '\000' <repeats 15 times>, pid = 443, subdepth = 0, static pool = 0xaf33be0, static VERSION = 0, static zero = 0, static Z_EPSILON = 0.001, static UNBOUNDED_DEFAULT_EXTENT = 0, static LEAK = false}
    (gdb) p *nd->parent->parent
    $9 = {typecode = 1, complement = 0, lvid = 101, xform = 0x0, param = 0x0, aabb = 0x0, parent = 0x0, child = std::vector of length 2, capacity 2 = {0xb4cb3e0, 0xb4cd3b0}, depth = 0, note = 0, coincide = 0, label = '\000' <repeats 15 times>, 
      pid = 442, subdepth = 0, static pool = 0xaf33be0, static VERSION = 0, static zero = 0, static Z_EPSILON = 0.001, static UNBOUNDED_DEFAULT_EXTENT = 0, static LEAK = false}
    (gdb) 


::


    sn::render mode 4 TYPETAG
                                                       un       
                                                                
                                                 un       cy    
                                                                
                                           un       cy          
                                                                
                                     un       cy                
                                                                
                               un       cy                      
                                                                
                         un       cy                            
                                                                
                   un       cy                                  
                                                                
            [un]      cy                                        
                                                                
       in       cy                                              
                                                                
    cy    !cy                                                   
    ^^
    ??

Some transform problem with leftmost node. Could be with all but thats the first. 

 



::

    2678 /**
    2679 stree::get_combined_tran_and_aabb
    2680 --------------------------------------
    2681 
    2682 Critical usage of ths from CSGImport::importNode
    2683 
    2684 0. early exits returning nullptr for non leaf nodes
    2685 1. gets combined structural(snode.h) and CSG tree(sn.h) transform 
    2686 2. collects that combined transform and its inverse (t,v) into Tran instance
    2687 3. copies leaf frame bbox values from the CSG nd into callers aabb array
    2688 4. transforms the bbox of the callers aabb array using the combined structural node 
    2689    + tree node transform
    2690 
    2691 
    2692 Note that sn::uncoincide needs CSG tree frame AABB but whereas this needs leaf 
    2693 frame AABB. These two demands are met by changing the AABB frame 
    2694 within sn::postconvert
    2695 
    2696 **/
    2697 
    2698 inline const Tran<double>* stree::get_combined_tran_and_aabb(
    2699     double* aabb,
    2700     const snode& node,
    2701     const sn* nd,
    2702     std::ostream* out
    2703     ) const
    2704 {
    2705     assert( nd );
    2706     if(!CSG::IsLeaf(nd->typecode)) return nullptr ;
    2707 
    2708     glm::tmat4x4<double> t(1.) ;
    2709     glm::tmat4x4<double> v(1.) ;
    2710     get_combined_transform(t, v, node, nd, out );
    2711 
    2712     // NB ridx:0 full stack of transforms from root down to CSG constituent nodes
    2713     //    ridx>0 only within the instance and within constituent CSG tree 
    2714      
    2715     const Tran<double>* tv = new Tran<double>(t, v);
    2716 
    2717     nd->copyBB_data( aabb );
    2718     stra<double>::Transform_AABB_Inplace(aabb, t);
    2719 
    2720     return tv ;
    2721 }





When do not delete the source can see that this is one ahead of the check::

    _pool::remove nd pid 23
    ] sn::~sn pid 23
    ]sn::CreateSmallerTreeWithListNode
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 31 msg sn_test::deepcopy_2.r1.bef
     this.desc sn::desc pid   32 idx   31 typecode  12 num_node   9 num_leaf   8 maxdepth  1 is_positive_form Y lvid  -1 tag di
     chk.desc  sn::desc pid   31 idx   30 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 23 msg sn_test::deepcopy_2.r1.bef
     this.desc sn::desc pid   24 idx   23 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
     chk.desc  -
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 24 msg sn_test::deepcopy_2.r1.bef
     this.desc sn::desc pid   25 idx   24 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
     chk.desc  sn::desc pid   24 idx   23 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 25 msg sn_test::deepcopy_2.r1.bef
     this.desc sn::desc pid   26 idx   25 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
     chk.desc  sn::desc pid   25 idx   24 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 26 msg sn_test::deepcopy_2.r1.bef
     this.desc sn::desc pid   27 idx   26 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
     chk.desc  sn::desc pid   26 idx   25 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 27 msg sn_test::deepcopy_2.r1.bef
     this.desc sn::desc pid   28 idx   27 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
     chk.desc  sn::desc pid   27 idx   26 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 28 msg sn_test::deepcopy_2.r1.bef
     this.desc sn::desc pid   29 idx   28 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
     chk.desc  sn::desc pid   28 idx   27 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 29 msg sn_test::deepcopy_2.r1.bef
     this.desc sn::desc pid   30 idx   29 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
     chk.desc  sn::desc pid   29 idx   28 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 100 tag sp
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 30 msg sn_test::deepcopy_2.r1.bef


Suspect the sn::set_right deletion of the former RHS could be implicated due to the resulting 
ordering of node deletion and node creation. 



Issue still there 
---------------------------------------

::

    ]]stree::postcreate
    2024-11-06 20:23:21.654 INFO  [215762] [U4Tree::Create@236] ]stree::postcreate
    [Detaching after fork from child process 217126]
    [CSGImport::importPrim.dump_LVID:1 node.lvid 101 LVID -1 name uni1 soname uni1 primIdx 0 bn 7 ln(subset of bn) 1 num_sub_total 8
    .CSGImport::importPrim dumping as ln > 0 : solid contains listnode
    python: /data/blyth/opticks_Debug/include/SysRap/sn.h:4689: static void sn::NodeTransformProduct(int, glm::tmat4x4<double>&, glm::tmat4x4<double>&, bool, std::ostream*): Assertion `nd' failed.

    Thread 1 "python" received signal SIGABRT, Aborted.
    0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6b35a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6b2d1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6b2d252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffc59b19b1 in sn::NodeTransformProduct (idx=434, t=..., v=..., reverse=false, out=0x0) at /data/blyth/opticks_Debug/include/SysRap/sn.h:4689
    #5  0x00007fffc59b3e1f in stree::get_combined_transform (this=0xaf32b90, t=..., v=..., node=..., nd=0xb4ca190, out=0x0) at /data/blyth/opticks_Debug/include/SysRap/stree.h:2655
    #6  0x00007fffc59b42a2 in stree::get_combined_tran_and_aabb (this=0xaf32b90, aabb=0x7ffffffefb60, node=..., nd=0xb4ca190, out=0x0) at /data/blyth/opticks_Debug/include/SysRap/stree.h:2710
    #7  0x00007fffc59adc30 in CSGImport::importNode (this=0x1ab3b9b0, nodeOffset=15603, partIdx=3, node=..., nd=0xb4ca190) at /home/blyth/opticks/CSG/CSGImport.cc:542
    #8  0x00007fffc59ad230 in CSGImport::importPrim (this=0x1ab3b9b0, primIdx=0, node=...) at /home/blyth/opticks/CSG/CSGImport.cc:388
    #9  0x00007fffc59acb90 in CSGImport::importSolidFactor (this=0x1ab3b9b0, ridx=6, ridx_type=70 'F') at /home/blyth/opticks/CSG/CSGImport.cc:251
    #10 0x00007fffc59ac009 in CSGImport::importSolid (this=0x1ab3b9b0) at /home/blyth/opticks/CSG/CSGImport.cc:92
    #11 0x00007fffc59abdf1 in CSGImport::import (this=0x1ab3b9b0) at /home/blyth/opticks/CSG/CSGImport.cc:55
    #12 0x00007fffc5908dfb in CSGFoundry::importSim (this=0x1ab3b6b0) at /home/blyth/opticks/CSG/CSGFoundry.cc:1696
    #13 0x00007fffc590e412 in CSGFoundry::CreateFromSim () at /home/blyth/opticks/CSG/CSGFoundry.cc:3000
    #14 0x00007fffcd2c2499 in G4CXOpticks::setGeometry (this=0xaf37ba0, world=0x97b2490) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:321
    #15 0x00007fffcd2c04d5 in G4CXOpticks::SetGeometry (world=0x97b2490) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:58
    #16 0x00007fffcd2c0770 in G4CXOpticks::SetGeometry_JUNO (world=0x97b2490, sd=0x99a0150, jpmt=0xaeef5f0, jlut=0xaf320e0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:96
    #17 0x00007fffbe3462f9 in LSExpDetectorConstruction_Opticks::Setup (opticksMode=1, world=0x97b2490, sd=0x99a0150, ppd=0x5a9510, psd=0x6635900, pmt



Off the rails by 20 sn::

    ]]stree::postcreate
    2024-11-06 20:43:36.510 INFO  [255098] [U4Tree::Create@236] ]stree::postcreate
    [Detaching after fork from child process 256447]
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 753 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  773 idx  753 typecode 110 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 301 tag bo
     chk.desc  sn::desc pid  753 idx  733 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 293 tag sp
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 752 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  772 idx  752 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 300 tag cy
     chk.desc  sn::desc pid  752 idx  732 typecode   1 num_node   3 num_leaf   2 maxdepth  1 is_positive_form Y lvid 292 tag un
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 751 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  771 idx  751 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 299 tag cy
     chk.desc  sn::desc pid  751 idx  731 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 292 tag cy
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 750 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  770 idx  750 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 298 tag cy
     chk.desc  sn::desc pid  750 idx  730 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 292 tag sp
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 735 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  755 idx  735 typecode   1 num_node   3 num_leaf   2 maxdepth  1 is_positive_form Y lvid 293 tag un
     chk.desc  sn::desc pid  735 idx  715 typecode 116 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 277 tag to
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 733 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  753 idx  733 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 293 tag sp
     chk.desc  sn::desc pid  733 idx  713 typecode 116 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 275 tag to
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 734 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  754 idx  734 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 293 tag cy
     chk.desc  sn::desc pid  734 idx  714 typecode 116 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 276 tag to
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 732 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  752 idx  732 typecode   1 num_node   3 num_leaf   2 maxdepth  1 is_positive_form Y lvid 292 tag un
     chk.desc  sn::desc pid  732 idx  712 typecode 116 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 274 tag to
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 730 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  750 idx  730 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 292 tag sp
     chk.desc  sn::desc pid  730 idx  710 typecode 116 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 272 tag to
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 731 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  751 idx  731 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 292 tag cy
     chk.desc  sn::desc pid  731 idx  711 typecode 116 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 273 tag to
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 573 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  593 idx  573 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 135 tag cy
     chk.desc  sn::desc pid  573 idx  553 typecode 103 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 125 tag zs
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 568 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  588 idx  568 typecode   2 num_node   3 num_leaf   2 maxdepth  1 is_positive_form Y lvid 132 tag in
     chk.desc  sn::desc pid  568 idx  548 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 122 tag cy
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 566 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  586 idx  566 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 132 tag cy
     chk.desc  sn::desc pid  566 idx  546 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 121 tag cy
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 567 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  587 idx  567 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 132 tag cy
     chk.desc  sn::desc pid  567 idx  547 typecode   2 num_node   3 num_leaf   2 maxdepth  1 is_positive_form Y lvid 121 tag in
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 569 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  589 idx  569 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 133 tag cy
     chk.desc  sn::desc pid  569 idx  549 typecode 105 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 122 tag cy
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 572 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  592 idx  572 typecode   2 num_node   3 num_leaf   2 maxdepth  1 is_positive_form Y lvid 134 tag in
     chk.desc  sn::desc pid  572 idx  552 typecode 103 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 124 tag zs
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 570 msg CSGImport::importPrim.check_idx



Try to fail sooner::

    ]]stree::postcreate
    2024-11-06 21:11:41.893 INFO  [349586] [U4Tree::Create@236] ]stree::postcreate
    [Detaching after fork from child process 350988]
    sn::check_idx_r idx_ OBJECT DOES NOT MATCH THIS OBJECT : POOL MIXUP ?  idx_ 753 msg CSGImport::importPrim.check_idx
     this.desc sn::desc pid  773 idx  753 typecode 110 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 301 tag bo
     chk.desc  sn::desc pid  753 idx  733 typecode 101 num_node   1 num_leaf   1 maxdepth  0 is_positive_form Y lvid 293 tag sp
    python: /data/blyth/opticks_Debug/include/SysRap/sn.h:2581: void sn::check_idx_r(int, const char*) const: Assertion `expect' failed.

    Thread 1 "python" received signal SIGABRT, Aborted.
    0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff6b34387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff6b35a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6b2d1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff6b2d252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffcd2f0169 in sn::check_idx_r (this=0xbf38630, d=0, msg=0x7fffc5a552f0 "CSGImport::importPrim.check_idx") at /data/blyth/opticks_Debug/include/SysRap/sn.h:2581
    #5  0x00007fffcd2eff41 in sn::check_idx (this=0xbf38630, msg=0x7fffc5a552f0 "CSGImport::importPrim.check_idx") at /data/blyth/opticks_Debug/include/SysRap/sn.h:2560
    #6  0x00007fffc59acdc2 in CSGImport::importPrim (this=0x1ab3b9b0, primIdx=0, node=...) at /home/blyth/opticks/CSG/CSGImport.cc:303
    #7  0x00007fffc59ac62a in CSGImport::importSolidGlobal (this=0x1ab3b9b0, ridx=0, ridx_type=82 'R') at /home/blyth/opticks/CSG/CSGImport.cc:179
    #8  0x00007fffc59ac00b in CSGImport::importSolid (this=0x1ab3b9b0) at /home/blyth/opticks/CSG/CSGImport.cc:90
    #9  0x00007fffc59abe21 in CSGImport::import (this=0x1ab3b9b0) at /home/blyth/opticks/CSG/CSGImport.cc:55
    #10 0x00007fffc5908e2b in CSGFoundry::importSim (this=0x1ab3b6b0) at /home/blyth/opticks/CSG/CSGFoundry.cc:1696
    #11 0x00007fffc590e442 in CSGFoundry::CreateFromSim () at /home/blyth/opticks/CSG/CSGFoundry.cc:3000
    #12 0x00007fffcd2c24e9 in G4CXOpticks::setGeometry (this=0xaf37ba0, world=0x97b2490) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:321
    #13 0x00007fffcd2c0525 in G4CXOpticks::SetGeometry (world=0x97b2490) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:58
    #14 0x00007fffcd2c07c0 in G4CXOpticks::SetGeometry_JUNO (world=0x97b2490, sd=0x99a0150, jpmt=0xaeef5f0, jlut=0xaf320e0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:96
    #15 0x00007fffbe3462f9 in LSExpDetectorConstruction_Opticks::Setup (opticksMode=1, world=0x97b2490, sd=0x99a0150, ppd=0x5a9510, psd=0x6635900, pmtscan=0x0)
        at /data/blyth/junotop/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc:46
    #16 0x00007fffbe31b07c in LSExpDetectorConstruction::setupOpticks (this=0x95c7be0, world=0x97b2490) at /data/blyth/junotop/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc:454
    #17 0x00007fffbe31a91c in LSExpDetectorConstruction::Construct (this=0x95c7be0) at /data/blyth/junotop/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc:375
    #18 0x00007fffcc18795e in G4RunManager::InitializeGeometry() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #19 0x00007fffcc187b2c in G4RunManager::Initialize() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so


Skipping all sn deletions avoids the pool mixup, and gets the input photon simulation to complete::

    P[blyth@localhost CSGOptiX]$ o
    On branch master
    Your branch is ahead of 'origin/master' by 1 commit.
      (use "git push" to publish your local commits)

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
        modified:   CSG/CSGImport.cc
        modified:   notes/issues/G4MultiUnion_within_G4BooleanSolid_has_voxelizer_SEGV.rst
        modified:   sysrap/sn.h
        modified:   sysrap/stree.h
        modified:   sysrap/tests/sn_test.cc
        modified:   sysrap/tests/sn_test.sh
        modified:   u4/U4Solid.h

    no changes added to commit (use "git add" and/or "git commit -a")
    P[blyth@localhost opticks]$ 

::

   P[blyth@localhost opticks]$ git commit -m "inconclusive debugging s_pool sn.h node inconsistency after node deletion, disabling all deletions gets things to complete with the G4MultiUnion avoided and listnode on GPU" 


Avoiding all deletions  (no longer needed after s_pool::getbyidx adoption)
-----------------------------------------------------------------------------


::

     443 inline void U4Solid::init_Tree_Shrink()
     444 {
     445     if( depth != 0 )  return ;
     446 
     447     sn* root0 = root ;
     448 
     449     if(root0->has_candidate_listnode_discontiguous())
     450     {
     451         root = sn::CreateSmallerTreeWithListNode_discontiguous(root0);
     452         root->check_idx("U4Solid::init_Tree_Shrink.discontiguous");
     453     }
     454     else if(root0->has_candidate_listnode_contiguous())
     455     {
     456         root = sn::CreateSmallerTreeWithListNode_contiguous(root0);
     457         root->check_idx("U4Solid::init_Tree_Shrink.contiguous");
     458     }
     459 
     460     if(root != root0)
     461     {
     462 
     463         std::cerr << "U4Solid::init_Tree_Shrink CHANGED root with sn::CreateSmallerTreeWithListNode_discontiguous/contiguous\n" ;
     464         std::cerr << "U4Solid::init_Tree_Shrink NOT DELETING \n" ;
     465         //delete root0 ; 
     466     }
     467 }


::
 
    4598 inline sn* sn::CreateSmallerTreeWithListNode(sn* root0, int q_note ) // static
    4599 {
    4600     std::cerr << "[sn::CreateSmallerTreeWithListNode\n" ;
    4601 
    4602     std::vector<sn*> prim0 ;  // populated with the hinted listnode prim 
    4603     sn* j0 = root0->find_joint_to_candidate_listnode(prim0, q_note);
    4604     if(j0 == nullptr) return nullptr ;
    4605 
    4606     std::vector<sn*> prim1 ;
    4607     sn::DeepCopy(prim1, prim0);
    4608 
    4609     sn* ln = sn::Compound( prim1, TypeFromNote(q_note) );
    4610 
    4611     sn* j1 = j0->deepcopy();
    4612 
    4613     //j1->set_right( ln, false );  // NB this deletes the extraneous RHS just copied by j0->deepcopy  
    4614     j1->set_child_leaking_prior(1, ln, false);
    4615 
    4616 
    4617     // ordering may be critical here as nodes get created and deleted by the above 
    4618 
    4619     std::cerr << "]sn::CreateSmallerTreeWithListNode\n" ;
    4620     return j1 ;
    4621 }



Do I need garbage collection ? NO : just some bug fix reworking of s_pool::get which was actually s_pool::lookup
-------------------------------------------------------------------------------------------------------------------------

Perhaps can implement something like garbage collection, such that I control when the 
deletions happen rather than interleaving them with creations that causes a complicated situation. 

Attempting to capture the problem in sysrap/tests/s_pool_test.sh revealed the source of the
issue to be a s_pool::get impl that did not cope to any deletions. 



After replacing s_pool::get with s_pool::getbyidx (which should cope with deletions) try putting back the deletions
---------------------------------------------------------------------------------------------------------------------

This seems to be working, but needs more testing. 


::

    calhost issues]$ o
    On branch master
    Your branch is up to date with 'origin/master'.

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
        modified:   notes/issues/G4MultiUnion_within_G4BooleanSolid_has_voxelizer_SEGV.rst
        modified:   sysrap/s_csg.h
        modified:   sysrap/s_pool.h
        modified:   sysrap/sn.h
        modified:   sysrap/tests/Obj.h
        modified:   sysrap/tests/s_pool_test.cc
        modified:   sysrap/tests/s_pool_test.sh
        modified:   u4/U4Solid.h

    Untracked files:
      (use "git add <file>..." to include in what will be committed)
        sysrap/tests/Obj.cc

