listnode_review_shrinking_trees.rst
=====================================

Context
---------

* prior :doc:`listnode_review`


Summary of what is needed for listnode support ?
------------------------------------------------------

0. propagating hints from G4VSolid names into the sn.h binary tree
1. detecting binary trees with hints and modifying the tree at sn.h level into n-ary tree
2. DONE : CSGImport handling n-ary tree conversion into CSGPrim/CSGNode setting subNum/subOffset 
3. DONE : testing pure listnodes
4. testing binary trees with 1 listnode
5. testing binary trees with >1 listnode (probably not important, as would be rare)

* step 1 is most difficult, step 0 and 2 are mostly similar to things done before
* with G4Polycone/G4MultiUnion no detection is needed, can directly create the n-ary tree
  so essentially only step 2 is needed for those see :doc:`listnode_review`

* One shortcut would be to switch to G4MultiUnion in the source geometry, 
  actually not a panacea as G4MultiUnion supports booleans within it, unlike listnode

* Best to start with  G4MultiUnion of G4VSolid that translate to single CSGNode leaves  




Q: Does listnode need externally defined bbox ?
--------------------------------------------------------

Put another way should the solid frame bbox be taken from Geant4 and propagated thru to GPU ?
Or should it be computed from the params of the prims and their known transforms ? 
A wrinkle with this is the transformation of bbox according to whether the listnode 
is within the instanced volumes or the global remainder.  

A: Doing both ways is useful as cross check.  

G4MultiUnion is now directly converted into listnode : but what about deep binary trees ?
-----------------------------------------------------------------------------------------------

When converting from G4MultiUnion/G4Polycone can know directly to create the listnode 
but with big boolean trees its more involved. Have to go looking for hints
in G4VSolid names and pluck nodes from tree and form the new tree. 

How/where to convert big boolean trees into smaller boolean trees with listnodes ? 
-------------------------------------------------------------------------------------

Looks like needs to within first stage (from G4VSolid to sn) 
although it doesnt need to be first pass. 

* for access to hinting in G4VSolid names 
* need sn.h flexibilty : it was designed for this task

  * n-ary tree  (vector of child nodes)
  * delete-able nodes




U4SolidMaker::AltLocalFastenerAcrylicConstruction
---------------------------------------------------

Defining source geometry as union of G4Tubs and G4MultiUnion 
avoids the need for hinting/detection/tree-rearrangement 
as the G4MultiUnion just becomes the listnode. 

::

    2058 const G4VSolid* U4SolidMaker::AltLocalFastenerAcrylicConstruction(const char* name) // static
    2059 {
    2060     const char* PREFIX = "AltLocalFastenerAcrylicConstruction" ; 
    2061     assert( sstr::StartsWith(name,PREFIX ));
    2062     long num_column = sstr::ExtractLong(name, 1) ;
    2063 
    2064     LOG(info) 
    2065         << " name " <<  ( name ? name : "-" )
    2066         << " num_column " << num_column
    2067         ;
    2068 
    2069     assert( num_column > 0 ); 
    2070 
    2071     G4Tubs* IonRing = new G4Tubs("IonRing",123*mm,206.2*mm,7*mm,0.0*deg,360.0*deg);
    2072 
    2073     G4MultiUnion* muni = new G4MultiUnion(name);
    2074     G4Tubs* screw = new G4Tubs("screw",0,13*mm,50.*mm,0.0*deg,360.0*deg);
    2075 
    2076     G4RotationMatrix rot(0, 0, 0);
    2077     for(long i=0;i<num_column;i++)
    2078     {
    2079        G4ThreeVector tlate(164.*cos(i*pi/4)*mm, 164.*sin(i*pi/4)*mm,-65.0*mm);
    2080        G4Transform3D tr(rot, tlate) ;
    2081        muni->AddNode( *screw, tr );
    2082     }
    2083 
    2084     G4UnionSolid* uni1 = new G4UnionSolid(name, IonRing, muni, 0, G4ThreeVector(0.,0.,0.));
    2085     return uni1 ;
    2086 }


DONE : Quick test of listnode within binary tree before impl the detection+tree shrinkage using AltLocalFastenerAcrylicConstructionSimple5
-----------------------------------------------------------------------------------------------------------------------------------------------

::

   GEOM ## set to AltLocalFastenerAcrylicConstructionSimple5
   ~/o/u4/tests/U4TreeCreateSSimTest.sh            ## create stree+scene 
   SCENE=3 ~/o/sysrap/tests/ssst.sh run            ## triangulated viz : get expected 5 Orb in a line along X

   ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh  ## full convert

   ~/o/cxr_min.sh                                      ## FIXED:EMPTY WORLD BOX  NOW GET 5 ANALYTIC ORB IN A LINE
   TRIMESH=1  ~/o/cxr_min.sh                           ## tri fallback is there, get 5 tri orb in line 
   TRIMESH=1 EYE=-0.1,0,0 TMIN=0.001 ~/o/cxr_min.sh    ## adjust viewpoint inside the Orb 



Some big trees can become a single listnode : if name hinting indicates it should
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

      L[A,B,C,D,E,F,G]    ## can drop all those U-nodes

      .            U
                  / \
                 U   G
                / \
               U   F
              / \
             U   E
            / \
           U   D
          / \
         U   C
        / \
       A   B


More typically big trees will become smaller with one listnode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Presence of non-union operator nodes(and user hinting) 
constrains which parts of the tree have to stay binary and what can become n-ary leaves.::



      .            U
                  / \
                 U   G
                / \
               U   F
              / \
             U   E
            / \
           U   D
          / \
         I   C
        / \
       A  !B

      .          
            
           U              <-- need to find the crux node (parent of first hinted prim in postorder traversal perhaps)
          / \
         I   L[C,D,E,F,G]
        / \
       A  !B


For G4VSolid name hinting, use integer suffix to indicate any separate listnodes::

   CSG_DISCONTIGUOUS_0
   CSG_DISCONTIGUOUS_0
   CSG_DISCONTIGUOUS_0

   CSG_DISCONTIGUOUS_1


Procedure:

1. first normal binary conversion creating binary sn tree 

   * (HMM: need to pass in the G4VSolid name hints somehow : have 16 char label)

2. postorder traversal looking for hinting and collecting prim nodes to be plucked from tree
   into list-nodes holding the prim within child vector 

3. clone the part of the original tree that must remain binary 

4. hookup list-node "heads" into the binary tree 


::

     599 inline void U4Tree::initSolid(const G4VSolid* const so, int lvid )
     600 {
     601     G4String _name = so->GetName() ; // bizarre: G4VSolid::GetName returns by value, not reference
     602     const char* name = _name.c_str();
     603 
     604     assert( int(solids.size()) == lvid );
     605     int d = 0 ;
     606 #ifdef WITH_SND
     607     int root = U4Solid::Convert(so, lvid, d );
     608     assert( root > -1 );
     609 #else
     610     sn* root = U4Solid::Convert(so, lvid, d );
     611     assert( root );
     612 #endif
     613 
     614     solids.push_back(so);
     615     st->soname_raw.push_back(name);
     616     st->solids.push_back(root);
     617 
     618    
     619 
     620 }



