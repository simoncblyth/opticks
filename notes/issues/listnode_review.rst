listnode_review
===================

Context
--------

* next :doc:`listnode_review_shrinking_trees`


See Also
----------

* :doc:`AltXJfixtureConstruction`


Overview
-----------

* looks like listnodes more developed in old x4 workflow, and only partly migratated into new u4/sn 
* as G4MultiUnion with prim subs is easier did that first  


TODO : compare CSG_CONTIGUOUS and CSG_DISCONTIGUOUS
-------------------------------------------------------


WIP : cxr_min.sh CSGOptiXRenderInteractiveTest : screenshot with annotation + metadata incl time
----------------------------------------------------------------------------------------------------


CSGOptiXRenderInteractiveTest.cc::

     94         uchar4* d_pixels = interop.output_buffer->map() ;
     95 
     96         cx->setExternalDevicePixels(d_pixels);
     97         cx->render_launch();


CSGOptiXRMTest.cc::

     10 int main(int argc, char** argv)
     11 {
     12     OPTICKS_LOG(argc, argv); 
     13     return CSGOptiX::RenderMain();
     14 }   

     144 int CSGOptiX::RenderMain() // static
     145 {
     146     SEventConfig::SetRGModeRender();
     147     CSGFoundry* fd = CSGFoundry::Load();
     148     CSGOptiX* cx = CSGOptiX::Create(fd) ;
     149     cx->render();
     150     delete cx ;
     151     return 0 ;
     152 }


DONE : Split off CSGOptiX::render_save and hookup key K to invoke it from render loop of CSGOptiXRenderInteractiveTest.cc

Issues:

* repeated K write to same path overwriting previous
* jpg inverted in Y : FIXED with SIMG::setData/SIMG::flipVertical invoked from Frame::download


WIP : simplify snap path formation to use spath.h and avoid self overwriting
-------------------------------------------------------------------------------


Currently uses fixed NAMEPREFIX passed in from bash, but 
thats no longer appropriate with interactive viewpoint control 

Want to be able to generate at any point during interactive running 
a commandline that can reproduce the current view. 



::

    2024-05-18 16:51:03.796 INFO  [250343] [CSGOptiX::render_save@1209] 
           /home/blyth/tmp/GEOM/OrbGridMultiUnion10_30_YX/CSGOptiXRenderInteractiveTest/
                 cxr_min__eye_0,0.8,0__zoom_1.0__tmin_0.1__ALL.jpg 

    NAMEPREFIX : cxr_min__eye_0,0.8,0__zoom_1.0__tmin_0.1_  



::

    1164 void CSGOptiX::render_save(const char* stem_)
    1165 {
    1166     const char* stem = stem_ ? stem_ : getRenderStemDefault() ;  // without ext 
    1167     sglm->addlog("CSGOptiX::render_snap", stem );
    1168 
    1169     const char* topline = ssys::getenvvar("TOPLINE", sproc::ExecutableName() );
    1170     const char* botline_ = ssys::getenvvar("BOTLINE", nullptr );
    1171     const char* outdir = SEventConfig::OutDir();
    1172     const char* outpath = SEventConfig::OutPath(stem, -1, ".jpg" );
    1173     std::string _extra = GetGPUMeta();  // scontext::brief giving GPU name 
    1174     const char* extra = strdup(_extra.c_str()) ;
    1175 
    1176     std::string bottom_line = CSGOptiX::Annotation(launch_dt, botline_, extra );
    1177     const char* botline = bottom_line.c_str() ;
    1178 
    1179     LOG(LEVEL)
    1180           << SEventConfig::DescOutPath(stem, -1, ".jpg" );
    1181           ;
    1182 
    1183     LOG(LEVEL)
    1184           << " stem " << stem
    1185           << " outpath " << outpath
    1186           << " outdir " << ( outdir ? outdir : "-" )
    1187           << " launch_dt " << launch_dt
    1188           << " topline [" <<  topline << "]"
    1189           << " botline [" <<  botline << "]"
    1190           ;
    1191 
    1192     LOG(info) << outpath  << " : " << AnnotationTime(launch_dt, extra)  ;
    1193 
    1194     snap(outpath, botline, topline  );
    1195 
    1196     sglm->fr.save( outdir, stem );
    1197     sglm->writeDesc( outdir, stem, ".log" );
    1198 }


    727 const char* SEventConfig::OutPath( const char* stem, int index, const char* ext )
    728 {
    729     const char* outfold = OutFold();
    730     const char* outname = OutName();
    731     
    732     LOG(LEVEL)
    733         << " outfold " << ( outfold ? outfold : "-" )
    734         << " outname " << ( outname ? outname : "-" )
    735         << " stem " << ( stem ? stem : "-" )
    736         << " ext " << ( ext ? ext : "-" )
    737         ;
    738 
    739     return SPath::Make( outfold, outname, stem, index, ext, FILEPATH);
    740     // HMM: an InPath would use NOOP to not create the dir
    741 }   


    572 /**
    573 SPath::Make
    574 -------------
    575 
    576 Creates a path from the arguments::
    577 
    578     <base>/<reldir>/<stem><index><ext>
    579 
    580 * base and relname can be nullptr 
    581 * the stem index and ext are formatted using SPath::MakeName
    582 * directory is created 
    583 
    584 **/
    585 
    586 const char* SPath::Make( const char* base, const char* reldir, const char* stem, int index, const char* ext, int create_dirs )
    587 {
    588     assert( create_dirs == NOOP || create_dirs == FILEPATH );
    589     std::string name = MakeName(stem, index, ext); 
    590     const char* path = SPath::Resolve(base, reldir, name.c_str(), create_dirs ) ;
    591     return path ; 
    592 }   



DONE : ~/o/u4/tests/U4SolidTest.sh 
-------------------------------------------

* integrated U4SolidMaker into U4SolidTest for extending conversion to G4MultiUnion and looking at tree n-ary-ization 


DONE : check ~/o/u4/tests/U4TreeCreateSSimTest.sh with G4MultiUnion using GEOM
-------------------------------------------------------------------------------

::

   GEOM ## set to BoxGridMultiUnion10:30_YX  U4SolidMaker::Make   : causes problems later
   GEOM ## set to BoxGridMultiUnion10_30_YX  U4SolidMaker::Make 

   ~/o/u4/tests/U4TreeCreateSSimTest.sh     ## create stree+scene 

   SCENE=3 ~/o/sysrap/tests/ssst.sh run     ## triangulated viz

* 3x3x3 grid of 7x7x7 boxes 

* checking U4SolidMaker::GridMultiUnion_ the G4MultiUnion of 7x7x7 items is expected 
* the 3x3x3 gridding on top of the multiunion was inadvertant due to "Grid" in the name  
  being parsed by one of the U4VolumeMaker Wrap methods 


DONE : Try OrbGridMultiUnion10_30_YX because cannot see ana/tri difference with boxes
---------------------------------------------------------------------------------------

* setGeometry creation is slow, G4 takes a while to form the meshes for 3*3*3*7*7*7 = 9261 Orbs 

* remember that for trimesh fallback need to configure the name of the solid 
  to triangulate from the mmlabel.txt 

* listnode from G4MultiUnion working : making analytic render with 
  a solid of 7x7x7 = 343 subs that would be impossible using 
  complete binary tree : with render speed subjectively the same between tri and ana

::

   GEOM ## set to OrbGridMultiUnion10_30_YX  U4SolidMaker::Make 

   ~/o/u4/tests/U4TreeCreateSSimTest.sh     ## create stree+scene 
   SCENE=3 ~/o/sysrap/tests/ssst.sh run     ## triangulated viz

   ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh  ## full convert
   ~/o/cxr_min.sh                                      ## get 3x3x3 of 7x7x7 Orbs 
   TRIMESH=1  ~/o/cxr_min.sh                           ## tri fallback ok 
   TRIMESH=1 EYE=-0.1,0,0 TMIN=0.001 ~/o/cxr_min.sh    ## adjust viewpoint


DONE : Move to a simpler multiunion for debuugging : OrbOrbMultiUnionSimple
------------------------------------------------------------------------------

Issues:

* FIXED : tran all identity 

  * note that triangulated in disceptive as its getting the tri with transforms applied from g4

* FIXED : prim lack bbox for the listnode after changes to CSGImport::importPrim

::

    In [1]: cf.prim
    Out[1]: 
    array([[[    0.,     0.,     0.,     0.],
            [    0.,     0.,     0.,     0.],
            [-1000., -1000., -1000.,  1000.],
            [ 1000.,  1000.,     0.,     0.]],

           [[    0.,     0.,     0.,     0.],
            [    0.,     0.,     0.,     0.],
            [ -250.,   -50.,   -50.,   250.],
            [   50.,    50.,     0.,     0.]]], dtype=float32)



DONE : listnode CSGPrim bbox  
----------------------------------

binary tree node bbox comes from::

  CSGImport::importNode 
  stree::get_combined_tran_and_aabb 
  stree::get_combined_transform
  stree::get_node_product  

Correctly handling listnode needs some of that to be used from::

  CSGImport::importListNode 

Central question : how the above stree methods handle listnodes

First impl of sn(listnode) -> CSG in::

    CSGPrim* CSGImport::importPrim(int primIdx, const snode& node )


DONE : use U4TreeCreateSSimTest.sh with OrbOrbMultiUnionSimple2 to get transforms and bbox working in listnode
------------------------------------------------------------------------------------------------------------------

Checking in U4TreeCreateSSimTest.cc suggests the modified CSGImport::importPrim might be OK now::

   ~/o/u4/tests/U4TreeCreateSSimTest.sh
   ~/o/u4/tests/U4TreeCreateSSimTest.cc


Test commands
-----------------

::

   GEOM ## set to OrbOrbMultiUnionSimple2
   ~/o/u4/tests/U4TreeCreateSSimTest.sh            ## create stree+scene 
   SCENE=3 ~/o/sysrap/tests/ssst.sh run            ## triangulated viz : get expected 5 Orb in a line along X

   ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh  ## full convert

   ~/o/cxr_min.sh                                      ## FIXED:EMPTY WORLD BOX  NOW GET 5 ANALYTIC ORB IN A LINE
   TRIMESH=1  ~/o/cxr_min.sh                           ## tri fallback is there, get 5 tri orb in line 
   TRIMESH=1 EYE=-0.1,0,0 TMIN=0.001 ~/o/cxr_min.sh    ## adjust viewpoint inside the Orb 


DONE : full conversion + anaviz 
------------------------------------------

Full convert::

    GEOM ## check config is BoxGridMultiUnion10_30_YX
    ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh

FIXED: anaviz runs but gives empty box::

    ~/o/CSGOptiX/cxr_min.sh
    ~/o/cxr_min.sh   ## via symbolic link 


triviz gives expected triangulated geom 3x3x3x7x7x7 mid box::

     TRIMESH=1 ~/o/cxr_min.sh 
     EYE=-0.5,-0.5,0 TRIMESH=1 ~/o/cxr_min.sh

Find viewpoint inside one of the little boxes so every pixel is hitting the tri fallback multiunion:: 

     EYE=0,-0.01,0 TMIN=0.001 TRIMESH=1 ~/o/cxr_min.sh
     EYE=0,-0.01,0 TMIN=0.001 ~/o/cxr_min.sh


Issues:

* FIXED : prim lack bbox
* DONE : to calc the bbox of the listnode need to combine bbox of the subs accounting for their transforms


sn -> CSG with listnode
-------------------------

::

     793 CSGSolid* CSGMaker::makeList( const char* label, unsigned type, std::vector<CSGNode>& leaves, const std::vector<const Tran<double>*>* tran )
     794 {
     795     unsigned numSub = leaves.size() ;
     796     unsigned numTran = tran ? tran->size() : 0  ;
     797     if( numTran > 0 ) assert( numSub == numTran );
     798 
     799     unsigned numPrim = 1 ;
     800     CSGSolid* so = fd->addSolid(numPrim, label);
     801 
     802     unsigned numNode = 1 + numSub ;
     803     int nodeOffset_ = -1 ;
     804     CSGPrim* p = fd->addPrim(numNode, nodeOffset_ );
     805 
     806     unsigned subOffset = 1 ; // now using absolute offsets from "root" to the first sub  see notes/issues/ContiguousThreeSphere.rst
     807     CSGNode hdr = CSGNode::ListHeader(type, numSub, subOffset );
     808     CSGNode* n = fd->addNode(hdr);
     809 
     810     AABB bb = {} ;
     811     fd->addNodes( bb, leaves, tran );
     812     p->setAABB( bb.data() );
     813     so->center_extent = bb.center_extent()  ;
     814 
     815     fd->addNodeTran(n);   // setting identity transform 
     816 
     817     LOG(info) << "so.label " << so->label << " so.center_extent " << so->center_extent ;
     818     return so ;
     819 }







G4MultiUnion
---------------

::

    [blyth@localhost opticks]$ opticks-fl G4MultiUnion 
    ./extg4/X4Entity.cc
    ./extg4/X4Entity.hh
    ./extg4/X4Intersect.cc
    ./extg4/X4Intersect.hh
    ./extg4/X4Solid.cc
    ./extg4/X4SolidBase.cc
    ./extg4/X4SolidMaker.cc
    ./extg4/X4SolidTree.cc
    ./extg4/X4SolidTree.hh
    ./extg4/tests/convertMultiUnionTest.cc
    ./extg4/x4solid.h
    ./sysrap/SIntersect.h
    ./sysrap/ssolid.h
    ./u4/U4SolidMaker.cc
    ./u4/U4SolidTree.cc
    ./u4/U4SolidTree.hh
    ./u4/U4Solid.h
    [blyth@localhost opticks]$ 


* TODO: bring convertMultiUnionTest.cc into new workflow 



Review listnode
------------------

::

    1327 inline bool        sn::is_listnode() const { return CSG::IsList(typecode); }
    313     static bool IsList(int type){ return  (type == CSG_CONTIGUOUS || type == CSG_DISCONTIGUOUS || type == CSG_OVERLAP ) ; }



sn.h
----

::

    3399 /**
    3400 sn::max_binary_depth
    3401 -----------------------
    3402 
    3403 Maximum depth of the binary compliant portion of the n-ary tree, 
    3404 ie with listnodes not recursed and where nodes have either 0 or 2 children.  
    3405 The listnodes are regarded as leaf node primitives.  
    3406 
    3407 * Despite the *sn* tree being an n-ary tree (able to hold polycone and multiunion compounds)
    3408   it must be traversed as a binary tree by regarding the compound nodes as effectively 
    3409   leaf node "primitives" in order to generate the indices into the complete binary 
    3410   tree serialization in level order 
    3411 
    3412 * hence the recursion is halted at list nodes
    3413 
    3414 **/
    3415 
    3416 inline int sn::max_binary_depth() const
    3417 {
    3418     return max_binary_depth_r(0) ;
    3419 }
    3420 inline int sn::max_binary_depth_r(int d) const
    3421 {
    3422     int mx = d ;
    3423     if( is_listnode() == false )
    3424     {
    3425         int nc = num_child() ;
    3426         if( nc > 0 ) assert( nc == 2 ) ;
    3427         for(int i=0 ; i < nc ; i++)
    3428         {
    3429             sn* ch = get_child(i) ;
    3430             mx = std::max( mx,  ch->max_binary_depth_r(d + 1) ) ;
    3431         }
    3432     }
    3433     return mx ;
    3434 }
    3435 
    3436 
    3437 
    3438 
    3439 
    3440 /**
    3441 sn::getLVBinNode
    3442 ------------------
    3443 
    3444 Returns the number of nodes in a complete binary tree
    3445 of height corresponding to the max_binary_depth 
    3446 of this node. 
    3447 
    3448 **/
    3449 
    3450 inline uint64_t sn::getLVBinNode() const
    3451 {
    3452     int h = max_binary_depth();
    3453     uint64_t n = st::complete_binary_tree_nodes( h );
    3454     if(false) std::cout
    3455         << "sn::getLVBinNode"
    3456         << " h " << h
    3457         << " n " << n
    3458         << "\n"
    3459         ;
    3460     return n ;
    3461 }

    3463 /**
    3464 sn::getLVSubNode
    3465 -------------------
    3466 
    3467 Sum of children of compound nodes found beneath this node. 
    3468 HMM: this assumes compound nodes only contain leaf nodes 
    3469 
    3470 Notice that the compound nodes themselves are regarded as part of
    3471 the binary tree. 
    3472 
    3473 **/
    3474 
    3475 inline uint64_t sn::getLVSubNode() const
    3476 {
    3477     int constituents = 0 ;
    3478     std::vector<const sn*> subs ;
    3479     typenodes_(subs, CSG_CONTIGUOUS, CSG_DISCONTIGUOUS, CSG_OVERLAP );
    3480     int nsub = subs.size();

    3481     for(int i=0 ; i < nsub ; i++)
    3482     {
    3483         const sn* nd = subs[i] ;
    3484         assert( nd->typecode == CSG_CONTIGUOUS || nd->typecode == CSG_DISCONTIGUOUS );
    3485         constituents += nd->num_child() ;
    3486     }
    3487     return constituents ;
    3488 }
    3489 
    3490 
    3491 /**
    3492 sn::getLVNumNode
    3493 -------------------
    3494 
    3495 Returns total number of nodes that can contain 
    3496 a complete binary tree + listnode constituents
    3497 serialization of this node.  
    3498 
    3499 **/
    3500 
    3501 inline uint64_t sn::getLVNumNode() const
    3502 {
    3503     uint64_t bn = getLVBinNode() ;
    3504     uint64_t sn = getLVSubNode() ;
    3505     return bn + sn ;
    3506 }





IsList : note lots in old NCSG.cpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

May need to bring stuff from NCSG.cpp into sn.h ? 

::

    [blyth@localhost opticks]$ opticks-f IsList  | grep -v IsListed
    ./CSG/CSGDraw.cc:    else if( CSG::IsList((OpticksCSG_t)type) )
    ./CSG/CSGDraw.cc:    assert( CSG::IsList((OpticksCSG_t)type) ); 
    ./CSG_GGeo/CSG_GGeo_Convert.cc:    bool is_list = CSG::IsList((int)tc) ; 
    ./ggeo/GParts.hh:        // only valid for CSG::IsList(type) such as CSG_CONTIGUOUS/CSG_DISCONTIGUOUS multiunion 
    ./npy/NCSG.cpp:    else if(CSG::IsList(root_type))
    ./npy/NCSG.cpp:    bool is_list = CSG::IsList(type) ; 
    ./npy/NCSG.cpp:    bool is_list = CSG::IsList(node->type); 
    ./npy/NCSG.cpp:    bool is_list = CSG::IsList(typecode) ;  
    ./npy/NNode.cpp:    return CSG::IsList(type) ; 
    ./npy/NNode.cpp:       if(     ntyp == CSG_NODE && CSG::IsList(node->type)) collect = true ; 
    ./sysrap/OpticksCSG.h:    static bool IsList(int type){ return  (type == CSG_CONTIGUOUS || type == CSG_DISCONTIGUOUS || type == CSG_OVERLAP ) ; }
    ./sysrap/OpticksCSG.h:        else if( CSG::IsList(type) ) offset_type = type - CSG_LIST + CSG_OFFSET_LIST  ;   // -11 + 4  = -7
    ./sysrap/sn.h:inline bool        sn::is_listnode() const { return CSG::IsList(typecode); }
    ./sysrap/snd.cc:    return CSG::IsList(typecode); 
    ./sysrap/snd.cc:    return num_child == 0 || CSG::IsList(typecode ) ; 
    ./sysrap/tests/OpticksCSGTest.cc:              << " CSG::IsList(type) " << std::setw(2) << CSG::IsList(type)
    ./sysrap/tests/OpticksCSG_test.cc:              << " CSG::IsList(type) " << std::setw(2) << CSG::IsList(type)
    [blyth@localhost opticks]$ 


::

    1141 void NCSG::export_()
    1142 {
    1143     m_csgdata->prepareForExport() ;  //  create node buffer 
    1144 
    1145     NPY<float>* _nodes = m_csgdata->getNodeBuffer() ;
    1146     assert(_nodes);
    1147 
    1148     export_idx();
    1149 
    1150     if( m_root->is_tree() )
    1151     {
    1152         export_tree_();
    1153     }
    1154     else if( m_root->is_list() )
    1155     {
    1156         export_list_();
    1157     }
    1158     else if( m_root->is_leaf() )
    1159     {
    1160         export_leaf_();
    1161     }
    1162     else
    1163     {
    1164         assert(0) ;  // unexpected m_root type  
    1165     }
    1166 }






::

    [blyth@localhost opticks]$ opticks-f listnode
    ./CSG/tests/intersect_prim_test.cc:TODO: replace Sphere with boolean tree, listnode, tree with listnode, ...  

    ./npy/NCSG.cpp:Branching for listnode within trees is done 
    ./npy/NNode.cpp:TODO: update_gtransforms needs to be made listnode in tree aware ?
         listnode the old workflow  

    ./sysrap/sn.h:    bool is_listnode() const ; 
    ./sysrap/sn.h:inline bool        sn::is_listnode() const { return CSG::IsList(typecode); }
    ./sysrap/sn.h:ie with listnodes not recursed and where nodes have either 0 or 2 children.  
    ./sysrap/sn.h:The listnodes are regarded as leaf node primitives.  
    ./sysrap/sn.h:    if( is_listnode() == false )
    ./sysrap/sn.h:a complete binary tree + listnode constituents
    ./sysrap/sn.h:    if( nc > 0 && nd->is_listnode() == false ) // non-list operator node




    ./sysrap/snd.cc:a complete binary tree + listnode constituents
    ./sysrap/snd.cc:    if( nd->num_child > 0 && nd->is_listnode() == false ) // non-list operator node
    ./sysrap/snd.cc:bool snd::is_listnode() const 
    ./sysrap/snd.cc:ie with listnodes not recursed and where nodes have either 0 or 2 children.  
    ./sysrap/snd.cc:The listnodes are regarded as leaf node primitives.  
    ./sysrap/snd.cc:    if( is_listnode() == false )
    ./sysrap/snd.hh:    bool is_listnode() const ; 
    ./sysrap/snd.hh:    int max_binary_depth() const ;   // listnodes not recursed, listnodes regarded as leaf node primitives 
    ./sysrap/snd.hh:    bool is_binary_leaf() const ;   // listnodes are regarded as binary leaves
    [blyth@localhost opticks]$ 




CONTIGUOUS
-------------


::

    [blyth@localhost opticks]$ opticks-fl CONTIGUOUS
    ./CSG/csg_intersect_tree.h

        634 TREE_FUNC
        635 bool intersect_prim( float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
        636 {
        637     const unsigned typecode = node->typecode() ;
        638 #ifdef DEBUG 
        639     printf("//intersect_prim typecode %d name %s \n", typecode, CSG::Name(typecode) );
        640 #endif
        641 
        642     bool valid_intersect = false ;
        643     if( typecode >= CSG_LEAF )
        644     {
        645         valid_intersect = intersect_leaf(             isect, node, plan, itra, t_min, ray_origin, ray_direction ) ;
        646     }
        647     else if( typecode < CSG_NODE )
        648     {
        649         valid_intersect = intersect_tree(             isect, node, plan, itra, t_min, ray_origin, ray_direction ) ;
        650     }
        651 #ifdef WITH_CONTIGUOUS
        652     else if( typecode == CSG_CONTIGUOUS )
        653     {
        654         valid_intersect = intersect_node_contiguous(   isect, node, node, plan, itra, t_min, ray_origin, ray_direction ) ;
        655     }
        656 #endif
        657     else if( typecode == CSG_DISCONTIGUOUS )
        658     {
        659         valid_intersect = intersect_node_discontiguous( isect, node, node, plan, itra, t_min, ray_origin, ray_direction ) ;
        660     }
        661     else if( typecode == CSG_OVERLAP )
        662     {
        663         valid_intersect = intersect_node_overlap(       isect, node, node, plan, itra, t_min, ray_origin, ray_direction ) ;
        664     }
        665     return valid_intersect ;
        666 }

        intersect_node_contiguous hidden behing WITH_CONTIGUOUS but intersect_node_discontiguous is active


    ./CSG/CSGNode.cc
    ./CSG/CSGNode.h
    ./CSG/CSGImport.cc


    ./CSG/CMakeLists.txt

        137 target_compile_definitions( ${name} PUBLIC OPTICKS_CSG )
        138 target_compile_definitions( ${name} PUBLIC WITH_CONTIGUOUS )

        /// WITH_CONTIGUOUS is enabled  


    ./CSG/csg_intersect_node.h

        647 INTERSECT_FUNC
        648 bool intersect_node_discontiguous( float4& isect, const CSGNode* node, const CSGNode* root,
        649      const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
        650 {
        651     const unsigned num_sub = node->subNum() ;
        652     const unsigned offset_sub = node->subOffset() ;

        /// subNum/subOffset points to sequence of nodes after the binary tree nodes

        653 
        654     float4 closest = make_float4( 0.f, 0.f, 0.f, RT_DEFAULT_MAX ) ;
        655     float4 sub_isect = make_float4( 0.f, 0.f, 0.f, 0.f ) ;
        656 
        657     for(unsigned isub=0 ; isub < num_sub ; isub++)
        658     {
        659         const CSGNode* sub_node = root+offset_sub+isub ;
        660         if(intersect_leaf( sub_isect, sub_node, plan, itra, t_min, ray_origin, ray_direction ))
        661         {
        662             if( sub_isect.w < closest.w ) closest = sub_isect ;
        663         }
        664     }
        665 
        666     bool valid_isect = closest.w < RT_DEFAULT_MAX ;
        667     if(valid_isect)
        668     {
        669         isect = closest ;
        670     }
        671 
        672 #ifdef DEBUG
        673     printf("//intersect_node_discontiguous num_sub %d  closest.w %10.4f \n",
        674        num_sub, closest.w );
        675 #endif
        676 
        677     return valid_isect ;
        678 }


    ./CSG/CSGMaker.cc

         118     else if(StartsWith("ContiguousThreeSphere", name))    so = makeContiguousThreeSphere(name) ;
         119     else if(StartsWith("DiscontiguousThreeSphere", name))    so = makeDiscontiguousThreeSphere(name) ;
         120     else if(StartsWith("DiscontiguousTwoSphere", name))    so = makeDiscontiguousTwoSphere(name) ;
         121     else if(StartsWith("ContiguousBoxSphere", name))   so = makeContiguousBoxSphere(name) ;
         122     else if(StartsWith("DiscontiguousBoxSphere", name))   so = makeDiscontiguousBoxSphere(name) ;
         123     else if(StartsWith("DifferenceBoxSphere", name))   so = makeDifferenceBoxSphere(name) ;
         124     else if(StartsWith("ListTwoBoxTwoSphere", name))   so = makeListTwoBoxTwoSphere(name);
         125     else if(StartsWith("RotatedCylinder", name)) so = makeRotatedCylinder(name) ;

         /// do not see any checks of a binary tree combined with listnode, only direct listnode at "root" (pole more appropriate for listnode)


    ./CSGOptiX/cxr_overview.sh


    ./sysrap/OpticksCSG.h
    ./sysrap/OpticksCSG.py

    ./sysrap/sn.h


        3090 /**
        3091 sn::Compound
        3092 ------------
        3093 
        3094 Note there is no subNum/subOffset here, those are needed when 
        3095 serializing the n-ary sn tree of nodes into CSGNode presumably. 
        3096 
        3097 **/
        3098 
        3099 inline sn* sn::Compound(std::vector<sn*>& prims, int typecode_ )
        3100 {   
        3101     assert( typecode_ == CSG_CONTIGUOUS || typecode_ == CSG_DISCONTIGUOUS );
        3102     
        3103     int num_prim = prims.size();
        3104     assert( num_prim > 0 );
        3105     
        3106     sn* nd = Create( typecode_ );
        3107     
        3108     for(int i=0 ; i < num_prim ; i++)
        3109     {   
        3110         sn* pr = prims[i] ;
        3111 #ifdef WITH_CHILD
        3112         nd->add_child(pr) ;
        3113 #else   
        3114         assert(0 && "sn::Compound requires WITH_CHILD " );
        3115         assert(num_prim == 2 ); 
        3116         if(i==0) nd->set_left(pr,  false) ;
        3117         if(i==1) nd->set_right(pr, false) ;
        3118 #endif
        3119     }
        3120     return nd ;
        3121 }

    ./sysrap/snd.cc


    ./sysrap/tests/OpticksCSGTest.cc
    ./sysrap/tests/OpticksCSG_test.cc
    ./sysrap/tests/snd_test.cc


    ./u4/U4SolidMaker.cc

         144     else if(StartsWith("CylinderFourBoxUnion", qname))        solid = U4SolidMaker::CylinderFourBoxUnion(qname) ;
         145     else if(StartsWith("BoxFourBoxUnion", qname))             solid = U4SolidMaker::BoxFourBoxUnion(qname) ;
         146     else if(StartsWith("BoxCrossTwoBoxUnion", qname))         solid = U4SolidMaker::BoxCrossTwoBoxUnion(qname) ;
         147     else if(StartsWith("BoxThreeBoxUnion", qname))            solid = U4SolidMaker::BoxThreeBoxUnion(qname) ;
         148     else if(StartsWith("OrbGridMultiUnion", qname))           solid = U4SolidMaker::OrbGridMultiUnion(qname) ;
         149     else if(StartsWith("BoxGridMultiUnion", qname))           solid = U4SolidMaker::BoxGridMultiUnion(qname) ;
         150     else if(StartsWith("BoxFourBoxContiguous", qname))        solid = U4SolidMaker::BoxFourBoxContiguous(qname) ;
         151     else if(StartsWith("LHCbRichSphMirr", qname))             solid = U4SolidMaker::LHCbRichSphMirr(qname) ;
         152     else if(StartsWith("LHCbRichFlatMirr", qname))            solid = U4SolidMaker::LHCbRichFlatMirr(qname) ;



    ./CSG_GGeo/CSG_GGeo_Convert.cc
              just note


    ./extg4/X4Solid.cc

         369 void X4Solid::convertMultiUnion()
         370 {
         371     const G4MultiUnion* const compound = static_cast<const G4MultiUnion*>(m_solid);
         372     assert(compound);
         373 
         374     //OpticksCSG_t type = CSG_DISCONTIGUOUS ;   
         375     OpticksCSG_t type = CSG_CONTIGUOUS ;
         376     // TODO: set type depending on solid name 
         377 
         378     unsigned sub_num = compound->GetNumberOfSolids() ;
         379     nnode* n_comp = nmultiunion::Create(type, sub_num) ;
         380 
         381     int lvIdx = get_lvIdx();  // pass lvIdx to children 
         382     bool top = false ;
         383 
         384     for( unsigned isub=0 ; isub < sub_num ; isub++)
         385     {
         386         const G4VSolid* sub = compound->GetSolid(isub);
         387         // TODO: assert that the constituents are primitives, not booleans or G4MultiUnion 
         388 
         389         const G4Transform3D& tr = compound->GetTransformation(isub) ;
         390         glm::mat4 tr_sub = X4Transform3D::Convert(tr);
         391 
         392         X4Solid* x_sub = new X4Solid(sub, m_ok, top, lvIdx);
         393         nnode* n_sub = x_sub->getRoot();
         394 
         395         bool update_global = true ;
         396         n_sub->set_transform( tr_sub, update_global );
         397 
         398         n_comp->subs.push_back(n_sub);
         399     }
         400 
         401     setRoot(n_comp);
         402 }




         405 /**
         406 X4Solid::changeToListSolid
         407 ---------------------------------
         408 
         409 Hmm need to collect all leaves of the subtree rooted here into a
         410 compound like the above multiunion  
         411 
         412 Need to apply the X4Solid conversion to the leaves only
         413 and just collect flattened transforms from the operator nodes above them  
         414 
         415 Hmm probably simplest to apply the normal convertBooleanSolid and 
         416 then replace the nnode subtree. Because thats using the nnode 
         417 lingo should do thing within nmultiunion
         418 
         419 Just need to collect the list of nodes. Hmm maybe flatten transforms ?
         420 
         421 
         422 Q: what about a list node within an ordinary CSG tree ?
         423 A: see X4Solid::convertBooleanSolid the getRoot is called on the X4Solid from the 
         424    xleft and xright X4Solid instances and these are put together in an ordinary operator
         425    nnode. So what will happen is that the left or right of the operator node will 
         426    end up being set get set to the nmultiunion.
         427 
         428    To follow what happens next in the GeoChain need to see NCSG and how it handles
         429    the export on encountering the nmultiunion. 
         430 
         431 **/
         432 
         433 void X4Solid::changeToListSolid(unsigned hint)
         434 {
         435     LOG(LEVEL) << "[ hint " << CSG::Name(hint)  ;
         436     assert( hint == CSG_CONTIGUOUS || hint == CSG_DISCONTIGUOUS );  //  CSG_OVERLAP not implemented yet
         437 
         438     nnode* subtree = getRoot();
         439     OpticksCSG_t typecode = (OpticksCSG_t)hint ;
         440 
         441     nmultiunion* root = nmultiunion::CreateFromTree(typecode, subtree) ;
         442     setRoot(root);
         443     LOG(LEVEL) << "]" ;
         444 }


    ./extg4/X4SolidBase.cc
    ./extg4/X4SolidMaker.cc
    ./ggeo/GParts.hh
    ./npy/NCSG.cpp
    ./npy/NMultiUnion.cpp
    ./npy/NNode.cpp
    ./npy/NNode.hpp
    ./npy/NOpenMeshCfg.cpp
    ./npy/NOpenMeshCfg.hpp
    ./npy/NOpenMeshFind.cpp
    ./npy/tests/NMultiUnionTest.cc

    [blyth@localhost opticks]$ 



where is the translation ? subNum
-------------------------------------

::

    [blyth@localhost opticks]$ opticks-fl subNum
    ./CSG/csg_intersect_tree.h
    ./CSG/CSGDraw.cc

        140 void CSGDraw::draw_list()
        141 {
        142     assert( CSG::IsList((OpticksCSG_t)type) );
        143 
        144     unsigned idx = 0 ;
        145     const CSGNode* head = q->getSelectedNode(idx);
        146     unsigned sub_num = head->subNum() ;
        147 
        148     LOG(info)
        149         << " sub_num " << sub_num
        150         ;
        151 
        152     draw_list_item( head, idx );
        153 
        154     for(unsigned isub=0 ; isub < sub_num ; isub++)
        155     {
        156         idx = 1+isub ;   // 0-based node idx
        157         const CSGNode* sub = q->getSelectedNode(idx);
        158 
        159         draw_list_item( sub, idx );
        160     }
        161 }


    ./CSG/CSGNode.cc
    ./CSG/CSGNode.h

        190 struct CSG_API CSGNode
        191 {
        192     quad q0 ;
        193     quad q1 ;
        194     quad q2 ;
        195     quad q3 ;
        196 
        197     // only used for CSG_CONVEXPOLYHEDRON and similar prim like CSG_TRAPEZOID which are composed of planes 
        198     NODE_METHOD unsigned planeIdx()      const { return q0.u.x ; }  // 1-based, 0 meaning None
        199     NODE_METHOD unsigned planeNum()      const { return q0.u.y ; }
        200     NODE_METHOD void setPlaneIdx(unsigned idx){  q0.u.x = idx ; }
        201     NODE_METHOD void setPlaneNum(unsigned num){  q0.u.y = num ; }
        202 
        203     // used for compound node types such as CSG_CONTIGUOUS, CSG_DISCONTIGUOUS and the rootnode of boolean trees CSG_UNION/CSG_INTERSECTION/CSG_DIFFERENCE...
        204     NODE_METHOD unsigned subNum()        const { return q0.u.x ; }
        205     NODE_METHOD unsigned subOffset()     const { return q0.u.y ; }
        206 
        207     NODE_METHOD void setSubNum(unsigned num){    q0.u.x = num ; }
        208     NODE_METHOD void setSubOffset(unsigned num){ q0.u.y = num ; }


        200 CSGNode CSGNode::Overlap(      int num_sub, int sub_offset){ return CSGNode::ListHeader( CSG_OVERLAP, num_sub, sub_offset ); }
        201 CSGNode CSGNode::Contiguous(   int num_sub, int sub_offset){ return CSGNode::ListHeader( CSG_CONTIGUOUS, num_sub, sub_offset ); }
        202 CSGNode CSGNode::Discontiguous(int num_sub, int sub_offset){ return CSGNode::ListHeader( CSG_DISCONTIGUOUS, num_sub, sub_offset ); }
        203 
        204 CSGNode CSGNode::ListHeader(unsigned type, int num_sub, int sub_offset )   // static 
        205 {
        206     CSGNode nd = {} ;
        207     switch(type)
        208     {
        209         case CSG_OVERLAP:       nd.setTypecode(CSG_OVERLAP)       ; break ;
        210         case CSG_CONTIGUOUS:    nd.setTypecode(CSG_CONTIGUOUS)    ; break ;
        211         case CSG_DISCONTIGUOUS: nd.setTypecode(CSG_DISCONTIGUOUS) ; break ;
        212         default:   assert(0)  ;
        213     }
        214     if(num_sub > 0)
        215     {
        216         nd.setSubNum(num_sub);
        217     }
        218     if(sub_offset > 0)
        219     {
        220         nd.setSubOffset(sub_offset);
        221     }
        222     return nd ;
        223 }


    ./CSG/CSGQuery.cc
    ./CSG/CSGQuery.h



    ./CSG/CSGImport.cc

        204 /**
        205 CSGImport::importPrim
        206 ----------------------
        207 
        208 Converting *snd/scsg* n-ary tree with compounds (eg multiunion and polycone) 
        209 into the CSGNode serialized binary tree with list node constituents appended using 
        210 subNum/subOffset referencing.   
        211 
        212 * Despite the input *snd* tree being an n-ary tree (able to hold polycone and multiunion compounds)
        213   it must be traversed as a binary tree by regarding the compound nodes as effectively leaf node "primitives" 
        214   in order to generate the indices into the complete binary tree serialization in level order 
        215 
        216 **/
        217 
        218 
        219 CSGPrim* CSGImport::importPrim(int primIdx, const snode& node )
        220 {
        221 #ifdef WITH_SND
        222     CSGPrim* pr = importPrim_<snd>(primIdx, node ) ;
        223 #else
        224     CSGPrim* pr = importPrim_<sn>(primIdx, node ) ;
        225 #endif
        226     return pr ;
        227 }


        229 /**
        230 CSGImport::importPrim_
        231 ------------------------
        232 
        233 TODO: add listnode handling 
        234 
        235 
        236 **/
        237 
        238 
        239 template<typename N>
        240 CSGPrim* CSGImport::importPrim_(int primIdx, const snode& node )
        241 {
        242     int lvid = node.lvid ;
        243     const char* name = fd->getMeshName(lvid)  ;
        244     
        245     std::vector<const N*> nds ;
        246 
        247     N::GetLVNodesComplete(nds, lvid);   // many nullptr in unbalanced deep complete binary trees
        248     int numParts = nds.size(); 
        249     



    ./CSG/csg_intersect_node.h
    ./CSG/tests/CSGFoundryAB.py
    ./CSG/tests/CSGFoundryLoadTest.py
    ./CSG/tests/CSGNode_test.cc
    ./CSG/CSGMaker.cc
    ./CSG_GGeo/CSG_GGeo_Convert.cc
    ./npy/NCSG.cpp
    ./npy/NNode.cpp
    ./npy/NNode.hpp
    ./sysrap/sn.h
    [blyth@localhost opticks]$ 



