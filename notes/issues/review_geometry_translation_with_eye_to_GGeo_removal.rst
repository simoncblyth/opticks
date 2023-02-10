review_geometry_translation_with_eye_to_GGeo_removal
=======================================================






Comparison of stree.py and CSGFoundry.py python dumping of geometry
------------------------------------------------------------------------

::


    epsilon:opticks blyth$ GEOM=J007 RIDX=1 ./sysrap/tests/stree_load_test.sh ana
    epsilon:opticks blyth$ GEOM=J007 RIDX=1 ./CSG/tests/CSGFoundryLoadTest.sh ana


Old workflow refs
-------------------

NCSG::export
    writes nodetree into transport buffers 

NCSG::export_tree
NCSG::export_list
NCSG::export_leaf

NCSG::export_tree_list_prepare
    explains subNum/subOffet in serialization 
    of trees with list nodes

nnode::find_list_nodes_r
nnode::is_list
    CSG::IsList(type)   CSG_CONTIGUOUS or CSG_DISCONTIGUOUS or CSG_OVERLAP      

nnode::subNum
nnode::subOffset

    CSG::IsCompound

CSGNode re:subNum subOffset
    Used by compound node types such as CSG_CONTIGUOUS, CSG_DISCONTIGUOUS and 
    the rootnode of boolean trees CSG_UNION/CSG_INTERSECTION/CSG_DIFFERENCE...
    Note that because subNum uses q0.u.x and subOffset used q0.u.y 
    this should not be used for leaf nodes. 

NCSG::export_tree_r
    assumes pure binary tree serializing to 2*idx+1 2*idx+2 




Consider lvid:103
---------------------

::

    CSGImport::importPrim@246:  primIdx 3078 lvid 103 num_nd  17 num_non_binary   0 max_binary_depth   6 : solidXJfixture0x5bbd6b0
    snd::render_v - ix:  475 dp:    0 sx:   -1 pt:   -1     nc:    2 fc:  469 ns:   -1 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 11
    snd::render_v - ix:  469 dp:    1 sx:    0 pt:  475     nc:    2 fc:  467 ns:  474 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 9
    snd::render_v - ix:  467 dp:    2 sx:    0 pt:  469     nc:    2 fc:  465 ns:  468 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 7
    snd::render_v - ix:  465 dp:    3 sx:    0 pt:  467     nc:    2 fc:  463 ns:  466 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 5
    snd::render_v - ix:  463 dp:    4 sx:    0 pt:  465     nc:    2 fc:  461 ns:  464 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 3
    snd::render_v - ix:  461 dp:    5 sx:    0 pt:  463     nc:    2 fc:  459 ns:  462 lv:  103     tc:    3 pa:   -1 bb:   -1 xf:   -1    di ordinal 1
    snd::render_v - ix:  459 dp:    6 sx:    0 pt:  461     nc:    0 fc:   -1 ns:  460 lv:  103     tc:  105 pa:  281 bb:  281 xf:   -1    cy ordinal 0
    snd::render_v - ix:  460 dp:    6 sx:    1 pt:  461     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  105 pa:  282 bb:  282 xf:   -1    cy ordinal 2
    snd::render_v - ix:  462 dp:    5 sx:    1 pt:  463     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  283 bb:  283 xf:  170    bo ordinal 4
    snd::render_v - ix:  464 dp:    4 sx:    1 pt:  465     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  284 bb:  284 xf:  171    bo ordinal 6
    snd::render_v - ix:  466 dp:    3 sx:    1 pt:  467     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  285 bb:  285 xf:  172    bo ordinal 8
    snd::render_v - ix:  468 dp:    2 sx:    1 pt:  469     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  286 bb:  286 xf:  173    bo ordinal 10
    snd::render_v - ix:  474 dp:    1 sx:    1 pt:  475     nc:    2 fc:  472 ns:   -1 lv:  103     tc:    3 pa:   -1 bb:   -1 xf:  176    di ordinal 15
    snd::render_v - ix:  472 dp:    2 sx:    0 pt:  474     nc:    2 fc:  470 ns:  473 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 13
    snd::render_v - ix:  470 dp:    3 sx:    0 pt:  472     nc:    0 fc:   -1 ns:  471 lv:  103     tc:  110 pa:  287 bb:  287 xf:   -1    bo ordinal 12
    snd::render_v - ix:  471 dp:    3 sx:    1 pt:  472     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  288 bb:  288 xf:  174    bo ordinal 14
    snd::render_v - ix:  473 dp:    2 sx:    1 pt:  474     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  289 bb:  289 xf:  175    bo ordinal 16
    *CSGImport::importPrim@256: 
    snd::rbrief
    - ix:  475 dp:    0 sx:   -1 pt:   -1     nc:    2 fc:  469 ns:   -1 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  469 dp:    1 sx:    0 pt:  475     nc:    2 fc:  467 ns:  474 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  467 dp:    2 sx:    0 pt:  469     nc:    2 fc:  465 ns:  468 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  465 dp:    3 sx:    0 pt:  467     nc:    2 fc:  463 ns:  466 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  463 dp:    4 sx:    0 pt:  465     nc:    2 fc:  461 ns:  464 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  461 dp:    5 sx:    0 pt:  463     nc:    2 fc:  459 ns:  462 lv:  103     tc:    3 pa:   -1 bb:   -1 xf:   -1    di
    - ix:  459 dp:    6 sx:    0 pt:  461     nc:    0 fc:   -1 ns:  460 lv:  103     tc:  105 pa:  281 bb:  281 xf:   -1    cy
    - ix:  460 dp:    6 sx:    1 pt:  461     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  105 pa:  282 bb:  282 xf:   -1    cy
    - ix:  462 dp:    5 sx:    1 pt:  463     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  283 bb:  283 xf:  170    bo
    - ix:  464 dp:    4 sx:    1 pt:  465     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  284 bb:  284 xf:  171    bo
    - ix:  466 dp:    3 sx:    1 pt:  467     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  285 bb:  285 xf:  172    bo
    - ix:  468 dp:    2 sx:    1 pt:  469     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  286 bb:  286 xf:  173    bo
    - ix:  474 dp:    1 sx:    1 pt:  475     nc:    2 fc:  472 ns:   -1 lv:  103     tc:    3 pa:   -1 bb:   -1 xf:  176    di
    - ix:  472 dp:    2 sx:    0 pt:  474     nc:    2 fc:  470 ns:  473 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  470 dp:    3 sx:    0 pt:  472     nc:    0 fc:   -1 ns:  471 lv:  103     tc:  110 pa:  287 bb:  287 xf:   -1    bo
    - ix:  471 dp:    3 sx:    1 pt:  472     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  288 bb:  288 xf:  174    bo
    - ix:  473 dp:    2 sx:    1 pt:  474     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  289 bb:  289 xf:  175    bo


    snd::render width 17 height 6 mode 3

                                                un                          
                                                                            
                                        un                      di          
                                                                            
                                un          bo          un          bo      
                                                                            
                        un          bo              bo      bo              
                                                                            
                un          bo                                              
                                                                            
        di          bo                                                      
                                                                            
    cy      cy                                                              
                                                                            
                                                                            


    CSGImport::importNode_v@310:  idx 0
    CSGImport::importNode_v@310:  idx 1
    CSGImport::importNode_v@310:  idx 3
    CSGImport::importNode_v@310:  idx 7
    CSGImport::importNode_v@310:  idx 15
    CSGImport::importNode_v@310:  idx 31
    CSGImport::importNode_v@310:  idx 63
    CSGImport::importNode_v@310:  idx 64
    CSGImport::importNode_v@310:  idx 32
    CSGImport::importNode_v@310:  idx 16
    CSGImport::importNode_v@310:  idx 8
    CSGImport::importNode_v@310:  idx 4
    CSGImport::importNode_v@310:  idx 2
    CSGImport::importNode_v@310:  idx 5
    CSGImport::importNode_v@310:  idx 11
    CSGImport::importNode_v@310:  idx 12
    CSGImport::importNode_v@310:  idx 6

::

    In [7]: w = cf.prim.view(np.int32)[:,1,1] == 103

    In [10]: cf.prim[w].shape
    Out[10]: (56, 4, 4)

    In [13]: cf.prim[w].view(np.int32)[:,0]
    Out[13]: 
    array([[  127, 16087,  7438,     0],    ## numNode, nodeOffset, tranOffset, planOffset
           [  127, 16214,  7447,     0],
           [  127, 16341,  7456,     0],
           [  127, 16468,  7465,     0],
           [  127, 16595,  7474,     0],
           [  127, 16722,  7483,     0],
           [  127, 16849,  7492,     0],


    In [27]: np.c_[np.arange(127),cf.node[16087:16087+127,3,2:].view(np.int32) ]
    Out[27]: 
    array([[          0,           1,           0],      # i, tc, complement~gtransformIdx
           [          1,           1,           0],
           [          2,           2,           0],
           [          3,           1,           0],
           [          4,         110,        7439],
           [          5,           1,           0],
           [          6,         110, -2147476208],
           [          7,           1,           0],
           [          8,         110,        7441],
           [          9,           0,           0],
           [         10,           0,           0],
           [         11,         110,        7442],
           [         12,         110,        7443],
           [         13,           0,           0],
           [         14,           0,           0],
           [         15,           1,           0],
           [         16,         110,        7444],
           [         17,           0,           0],
           [         18,           0,           0],
           [         19,           0,           0],
           [         20,           0,           0],
           [         21,           0,           0],
           [         22,           0,           0],
           [         23,           0,           0],

           ...

           [         28,           0,           0],
           [         29,           0,           0],
           [         30,           0,           0],
           [         31,           2,           0],
           [         32,         110,        7445],
           [         33,           0,           0],
           [         34,           0,           0],
           [         35,           0,           0],

           ...

           [         61,           0,           0],
           [         62,           0,           0],
           [         63,         105,        7446],
           [         64,         105, -2147476201],
           [         65,           0,           0],
           [         66,           0,           0],



Consider lvid:100 base_steel which is a single polycone prim within ridx 7
-------------------------------------------------------------------------------

::

    CSGImport::importPrim@201:  primIdx    0 lvid 100 snd::GetLVID   7 : base_steel0x5b335a0





Hmm this stree still using contiguous::

    GEOM=J007 RIDX=7 ./sysrap/tests/stree_load_test.sh ana


     lv:100 nlv: 1                                         base_steel csg  7 tcn 105:cylinder 105:cylinder 11:contiguous 105:cylinder 105:cylinder 11:contiguous 3:difference 
    desc_csg lvid:100 st.f.soname[100]:base_steel 
            ix   dp   sx   pt   nc   fc   sx   lv   tc   pm   bb   xf
    array([[444,   2,   0, 446,   0,  -1, 445, 100, 105, 272, 272,  -1,   0,   0,   0,   0],
           [445,   2,   1, 446,   0,  -1,  -1, 100, 105, 273, 273,  -1,   0,   0,   0,   0],
           [446,   1,   0, 450,   2, 444, 449, 100,  11,  -1,  -1,  -1,   0,   0,   0,   0],
           [447,   2,   0, 449,   0,  -1, 448, 100, 105, 274, 274,  -1,   0,   0,   0,   0],
           [448,   2,   1, 449,   0,  -1,  -1, 100, 105, 275, 275,  -1,   0,   0,   0,   0],
           [449,   1,   1, 450,   2, 447,  -1, 100,  11,  -1,  -1,  -1,   0,   0,   0,   0],
           [450,   0,  -1,  -1,   2, 446,  -1, 100,   3,  -1,  -1,  -1,   0,   0,   0,   0]], dtype=int32)

    stree.descSolids numSolid:10 detail:0 





    CSGFoundry.descSolid ridx  7 label               r7 numPrim      1 primOffset   3127 lv_one 1 
     pidx 3127 lv 100 pxl    0 :                                         base_steel : no 23403 nn    7 tcn 2:intersection 1:union 2:intersection 105:cylinder 105:cylinder 105:!cylinder 105:!cylinder  






Further thoughts on CSGImport::importTree
----------------------------------------------

Further thoughts now solidifying into CSG/CSGImport.cc CSGImport::importTree

CSGSolid
    main role is to hold (numPrim, primOffset) : ie specify a contiguous range of CSGPrim
CSGPrim
    main role is to hold (numNode, nodeOffset) : ie specify a contiguous range of CSGNode 


Difficulty 1 : polycone compounds
------------------------------------

X4Solid::convertPolycone uses NTreeBuilder<nnode> to 
generate a suitably sized complete binary tree of CSG_ZERO gaps
and then populates it with the available nodes.

::

    1706 void X4Solid::convertPolycone()
    1707 {
    ....
    1785     std::vector<nnode*> outer_prims ;
    1786     Polycone_MakePrims( zp, outer_prims, m_name, true  );
    1787     bool dump = false ;
    1788     nnode* outer = NTreeBuilder<nnode>::UnionTree(outer_prims, dump) ;
    1789 

Whilst validating the conversion (because want to do identicality check between old and new workflows) 
will need to implement the same within snd/scsg for example steered from U4Solid::init_Polycone U4Polycone::Convert

Because snd uses n-ary tree can subsequently enhance to using CSG_CONTIGUOUS 
bringing the compound thru to the GPU. 




Thoughts : How difficulty to go direct Geant4 -> CSGFoundry ?
--------------------------------------------------------------

* Materials and surfaces : pretty easily as GGeo/GMaterialLib/GSurfaceLib 
  are fairly simple containers that can easily be replaced with more modern 
  and generic approaches using NPFold/NP/NPX/SSim

  * WIP: U4Material.h .cc U4Surface.h 
  * TODO: boundary array standardizing the data already collected by U4Material, U4Surface


* Structure : U4Tree/stree : already covers most of whats needed (all the
  transforms and doing the factorization)

* Solids : MOST WORK NEEDED : MADE RECENT PROGRESS WITH U4Solid

  * WIP: U4Solid snd scsg stree CSGFoundry::importTree
  * DECIDE NO NEED FOR C4 PKG  

  * intricate web of translation and primitives code across x4/npy/GGeo 
  * HOW TO PROCEED : START AFRESH : CONVERTING G4VSolid trees into CSGPrim/CSGNode trees

    * aiming for much less code : avoiding intermediaries

    * former persisting approach nnode/GParts/GPts needs to be ripped out
  
      * "ripping out" is the wrong approach : simpler to start without heed to 
        what was done before : other than where the code needed is directly 
        analogous : in which case methods should be extracted and name changed 

    * CSGFoundary/CSGSolid/CSGPrim/CSGNode : handles all the persisting much more simply 
      so just think of mapping CSG trees of G4VSolid into CSGPrim/CSGNode trees

    * U4SolidTree (developed for Z cutting) has lots of of general stuff 
      that could be pulled out into a U4Solid.h to handle the conversion 


   
Solids : Central Issue : How to handle the CSG node tree ?  
-------------------------------------------------------------

* Geant4 CSG trees have G4DisplacedSolid complications with transforms held in illogical places  
* can an intermediate node tree be avoided ? 
* old way far too complicated :  nnode, nsphere,..., NCSG, GParts, GPts, GMesh, ... 

  * nnode, nsphere,... : raw node tree
  * NCSG/GParts/GPts : persist related  
  * GMesh : triangles and holder of analytic GParts 


* U4SolidTree avoids an intermediate tree but at the expense of 
  having lots of maps keyed on the G4VSolid nodes of the tree 

  * it might actually be simpler with a transient minimal intermediate node tree 
    to provide a convenient place for annotation during conversion 


Solid Conversion Complications
---------------------------------

* balancing (this has been shown to cause missed intersects in some complex trees, so need to live without it)
* nudging : avoiding coincident faces 


Old Solid Conversion Code
---------------------------

::

    0890 void X4PhysicalVolume::convertSolids()
     891 {
     895     const G4VPhysicalVolume* pv = m_top ;
     896     int depth = 0 ;
     897     convertSolids_r(pv, depth);
     907 }

    0909 /**
     910 X4PhysicalVolume::convertSolids_r
     911 ------------------------------------
     912 
     913 G4VSolid is converted to GMesh with associated analytic NCSG 
     914 and added to GGeo/GMeshLib.
     915 
     916 If the conversion from G4VSolid to GMesh/NCSG/nnode required
     917 balancing of the nnode then the conversion is repeated 
     918 without the balancing and an alt reference is to the alternative 
     919 GMesh/NCSG/nnode is kept in the primary GMesh. 
     920 
     921 Note that only the nnode is different due to the balancing, however
     922 its simpler to keep a one-to-one relationship between these three instances
     923 for persistency convenience.
     924 
     925 Note that convertSolid is called for newly encountered lv
     926 in the postorder tail after the recursive call in order for soIdx/lvIdx
     927 to match Geant4. 
     928 
     929 **/
     930 
     931 void X4PhysicalVolume::convertSolids_r(const G4VPhysicalVolume* const pv, int depth)
     932 {
     933     const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
     934 
     935     // G4LogicalVolume::GetNoDaughters returns 1042:G4int, 1062:size_t
     936     for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ;i++ )
     937     {
     938         const G4VPhysicalVolume* const daughter_pv = lv->GetDaughter(i);
     939         convertSolids_r( daughter_pv , depth + 1 );
     940     }
     941 
     942     // for newly encountered lv record the tail/postorder idx for the lv
     943     if(std::find(m_lvlist.begin(), m_lvlist.end(), lv) == m_lvlist.end())
     944     {
     945         convertSolid( lv );
     946     } 
     947 }

    0961 void X4PhysicalVolume::convertSolid( const G4LogicalVolume* lv )
     962 {
     963     const G4VSolid* const solid = lv->GetSolid();
     964 
     965     G4String  lvname_ = lv->GetName() ;      // returns by reference, but take a copied value 
     966     G4String  soname_ = solid->GetName() ;   // returns by value, not reference
     967 
     968     const char* lvname = strdup(lvname_.c_str());  // may need these names beyond this scope, so strdup     
     969     const char* soname = strdup(soname_.c_str());
     ...
     986     GMesh* mesh = ConvertSolid(m_ok, lvIdx, soIdx, solid, soname, lvname );
     987     mesh->setX4SkipSolid(x4skipsolid);
     988 
    1001     m_ggeo->add( mesh ) ;
    1002 
    1003     LOG(LEVEL) << "] " << std::setw(4) << lvIdx ;
    1004 }   


    1104 GMesh* X4PhysicalVolume::ConvertSolid_( const Opticks* ok, int lvIdx, int soIdx, const G4VSolid* const solid, const char* soname, const char* lvname,      bool balance_deep_tree ) // static
    1105 {   
    1129     const char* boundary = nullptr ; 
    1130     nnode* raw = X4Solid::Convert(solid, ok, boundary, lvIdx )  ;
    1131     raw->set_nudgeskip( is_x4nudgeskip );  
    1132     raw->set_pointskip( is_x4pointskip );
    1133     raw->set_treeidx( lvIdx );
    1134     
    1139     bool g4codegen = ok->isG4CodeGen() ;
    1140     
    1141     if(g4codegen) GenerateTestG4Code(ok, lvIdx, solid, raw);
    1142     
    1143     GMesh* mesh = ConvertSolid_FromRawNode( ok, lvIdx, soIdx, solid, soname, lvname, balance_deep_tree, raw );
    1144 
    1145     return mesh ;


::

    1156 GMesh* X4PhysicalVolume::ConvertSolid_FromRawNode( const Opticks* ok, int lvIdx, int soIdx, const G4VSolid* const solid, const char* soname, const ch     ar* lvname, bool balance_deep_tree,
    1157      nnode* raw)
    1158 {
    1159     bool is_x4balanceskip = ok->isX4BalanceSkip(lvIdx) ;
    1160     bool is_x4polyskip = ok->isX4PolySkip(lvIdx);   // --x4polyskip 211,232
    1161     bool is_x4nudgeskip = ok->isX4NudgeSkip(lvIdx) ;
    1162     bool is_x4pointskip = ok->isX4PointSkip(lvIdx) ;
    1163     bool do_balance = balance_deep_tree && !is_x4balanceskip ;
    1164 
    1165     nnode* root = do_balance ? NTreeProcess<nnode>::Process(raw, soIdx, lvIdx) : raw ;
    1166 
    1167     LOG(LEVEL) << " after NTreeProcess:::Process " ;
    1168 
    1169     root->other = raw ;
    1170     root->set_nudgeskip( is_x4nudgeskip );
    1171     root->set_pointskip( is_x4pointskip );
    1172     root->set_treeidx( lvIdx );
    1173 
    1174     const NSceneConfig* config = NULL ;
    1175 
    1176     LOG(LEVEL) << "[ before NCSG::Adopt " ;
    1177     NCSG* csg = NCSG::Adopt( root, config, soIdx, lvIdx );   // Adopt exports nnode tree to m_nodes buffer in NCSG instance
    1178     LOG(LEVEL) << "] after NCSG::Adopt " ;
    1179     assert( csg ) ;
    1180     assert( csg->isUsedGlobally() );
    1181 
    1182     bool is_balanced = root != raw ;
    1183     if(is_balanced) assert( balance_deep_tree == true );
    1184 
    1185     csg->set_balanced(is_balanced) ;
    1186     csg->set_soname( soname ) ;
    1187     csg->set_lvname( lvname ) ;
    1188 
    1189     LOG_IF(fatal, is_x4polyskip ) << " is_x4polyskip " << " soIdx " << soIdx  << " lvIdx " << lvIdx ;
    1190 
    1191     GMesh* mesh = nullptr ;
    1192     if(solid)
    1193     {
    1194         mesh =  is_x4polyskip ? X4Mesh::Placeholder(solid ) : X4Mesh::Convert(solid, lvIdx) ;
    1195     }
    1196     else
    1197     {





Old High Level Geometry Code
--------------------------------


::

    223 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    224 {   
    225     LOG(LEVEL) << " G4VPhysicalVolume world " << world ;
    226     assert(world);
    227     wd = world ;
    228     
    229     //sim = SSim::Create();  // its created in ctor  
    230     assert(sim) ;
    231     
    232     stree* st = sim->get_tree(); 
    233     // TODO: sim argument, not st : or do SSim::Create inside U4Tree::Create 
    234     tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    235 
    236     
    237     // GGeo creation done when starting from a gdml or live G4,  still needs Opticks instance
    238     Opticks::Configure("--gparts_transform_offset --allownokey" );
    239     
    240     GGeo* gg_ = X4Geo::Translate(wd) ;
    241     setGeometry(gg_);
    242 }
    243 
    244 
    245 void G4CXOpticks::setGeometry(GGeo* gg_)
    246 {
    247     LOG(LEVEL);
    248     gg = gg_ ;
    249 
    250 
    251     CSGFoundry* fd_ = CSG_GGeo_Convert::Translate(gg) ;
    252     setGeometry(fd_);
    253 }


::

     19 GGeo* X4Geo::Translate(const G4VPhysicalVolume* top)  // static 
     20 {
     21     bool live = true ;
     22 
     23     GGeo* gg = new GGeo( nullptr, live );   // picks up preexisting Opticks::Instance
     24 
     25     X4PhysicalVolume xtop(gg, top) ;  // lots of heavy lifting translation in here 
     26 
     27     gg->postDirectTranslation();
     28 
     29     return gg ;
     30 }


::

     199 void X4PhysicalVolume::init()
     200 {
     201     LOG(LEVEL) << "[" ;
     202     LOG(LEVEL) << " query : " << m_query->desc() ;
     203 
     204 
     205     convertWater();       // special casing in Geant4 forces special casing here
     206     convertMaterials();   // populate GMaterialLib
     207     convertScintillators();
     208 
     209 
     210     convertSurfaces();    // populate GSurfaceLib
     211     closeSurfaces();
     212     convertSolids();      // populate GMeshLib with GMesh converted from each G4VSolid (postorder traverse processing first occurrence of G4LogicalVo     lume)  
     213     convertStructure();   // populate GNodeLib with GVolume converted from each G4VPhysicalVolume (preorder traverse) 
     214     convertCheck();       // checking found some nodes
     215 
     216     postConvert();        // just reporting 
     217 
     218     LOG(LEVEL) << "]" ;
     219 }



