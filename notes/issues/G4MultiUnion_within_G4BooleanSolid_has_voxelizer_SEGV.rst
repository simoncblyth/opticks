G4MultiUnion_within_G4BooleanSolid_has_voxelizer_SEGV
========================================================

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

 
