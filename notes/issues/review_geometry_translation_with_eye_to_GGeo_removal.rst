review_geometry_translation_with_eye_to_GGeo_removal
=======================================================






Comparison of stree.py and CSGFoundry.py python dumping of geometry
------------------------------------------------------------------------

::


    epsilon:opticks blyth$ GEOM=J007 RIDX=1 ./sysrap/tests/stree_load_test.sh ana
    epsilon:opticks blyth$ GEOM=J007 RIDX=1 ./CSG/tests/CSGFoundryLoadTest.sh ana



AB comparison using CSGFoundryAB.sh
--------------------------------------

::

    ## rebuild and install after changes as lots of headeronly functionality 

    sy      
    om 
    u4
    om 
    c
    om 


    u4t
    ./U4TreeCreateTest.sh   ## Create stree from gdml
    ct
    ./CSGImportTest.sh      ## import stree into CSGFoundary and save 

    ## TODO: combine the above two steps
    ct
    ./CSGFoundryAB.sh       ## compare A:old and B:new CSGFoundry 



Missing itra tran and inst in B::


  : A.SSim                                             :                 None : 4 days, 3:38:40.838511 
  : A.solid                                            :           (10, 3, 4) : 4 days, 3:39:36.056485 
  : A.prim                                             :         (3259, 4, 4) : 4 days, 3:39:36.057583 
  : A.node                                             :        (23547, 4, 4) : 4 days, 3:39:36.441330 

  : A.mmlabel                                          :                   10 : 4 days, 3:39:37.611860 
  : A.primname                                         :                 3259 : 4 days, 3:39:36.056862 
  : A.meshname                                         :                  152 : 4 days, 3:39:37.612941 
  : A.meta                                             :                    8 : 4 days, 3:39:37.612404 

  : A.itra                                             :         (8179, 4, 4) : 4 days, 3:39:37.613551 
  : A.tran                                             :         (8179, 4, 4) : 4 days, 3:39:35.423639 

  : A.inst                                             :        (48477, 4, 4) : 4 days, 3:39:37.973285 




Where to do balancing and positivization in new workflow ?
-------------------------------------------------------------

Old::

    X4PhysicalVolume::ConvertSolid
    X4PhysicalVolume::ConvertSolid_ 
    X4PhysicalVolume::ConvertSolid_FromRawNode
    NTreeProcess::init
    NTreePositive::init 


CSG transforms : stree/scsg f.csg.xform only 240 items vs CSGFoundry A.tran with 8179 ? 
-----------------------------------------------------------------------------------------

CSGFoundry has thousands of CSG level tran,itra::

  : A.itra                                             :         (8179, 4, 4) : 4 days, 3:39:37.613551 
  : A.tran                                             :         (8179, 4, 4) : 4 days, 3:39:35.423639 

scsg only has 240 xform (thats a repetition factor of 34)::

    In [12]: f.csg 
    CMDLINE:/Users/blyth/opticks/sysrap/tests/stree_load_test.py
    csg.base:/Users/blyth/.opticks/GEOM/J007/CSGFoundry/SSim/stree/csg

      : csg.node                                           :            (637, 16) : 4 days, 21:34:31.826544 
      : csg.aabb                                           :             (387, 6) : 4 days, 21:34:31.827683 
      : csg.xform                                          :            (240, 16) : 4 days, 21:34:31.825574 
      : csg.NPFold_index                                   :                    4 : 4 days, 21:34:31.828374 
      : csg.param                                          :             (387, 6) : 4 days, 21:34:31.826033 


* presumably some kind of repetition in CSGFoundry, but elaborate on that 
* tracing in CSGFoundry provides the explanation

  * because CSGFoundry::addTran gets called from CSGFoundry::addNode are getting 
    significant repetition of CSG level transforms due to node repetition eg from the globals 

  * POTENTIAL FOR ENHANCEMENT HERE : BUT SOME RELOCATING OF GLOBALS IS DONE SOMEWHERE, SO NON-TRIVIAL  

::

    1366 CSGNode* CSGFoundry::addNode(CSGNode nd, const std::vector<float4>* pl, const Tran<double>* tr  )
    1367 {
    ...
    1371     unsigned globalNodeIdx = node.size() ;
    ...
    1404     if(tr)
    1405     {
    1406         unsigned trIdx = 1u + addTran(tr);  // 1-based idx, 0 meaning None
    1407         nd.setTransform(trIdx);
    1408     }
    1409 
    1410     node.push_back(nd);
    1411     last_added_node = node.data() + globalNodeIdx ;
    1412     return last_added_node ;
    1413 }


HMM actually a lower level CSG_GGeo_Convert::convertNode is used doing much the same::

     674 CSGNode* CSG_GGeo_Convert::convertNode(const GParts* comp, unsigned primIdx, unsigned partIdxRel )
     675 {
     ...
     677     unsigned partOffset = comp->getPartOffset(primIdx) ;
     678     unsigned partIdx = partOffset + partIdxRel ;
     ...
     691     const Tran<float>* tv = nullptr ; 
     692     unsigned gtran = comp->getGTransform(partIdx);  // 1-based index, 0 means None
     693     if( gtran > 0 )
     694     {
     695         glm::mat4 t = comp->getTran(gtran-1,0) ;
     696         glm::mat4 v = comp->getTran(gtran-1,1); 
     697         tv = new Tran<float>(t, v); 
     698     }
     699 
     700     unsigned tranIdx = tv ?  1 + foundry->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms
     701 
     702     // HMM: this is not using the higher level 
     703     // CSGFoundry::addNode with transform pointer argumnent 
     704 


Need to do something similar in CSGImport::importNode 
BUT first need the gtransforms, snd/scsg only has local transforms so far. 

::

     740 /**
     741 nnode::global_transform
     742 ------------------------
     743 
     744 NB parent links are needed
     745 
     746 Is invoked by nnode::update_gtransforms_r from each primitive, 
     747 whence parent links are followed up the tree until reaching root
     748 which has no parent. Along the way transforms are collected
     749 into the tvq vector in reverse hierarchical order from 
     750 leaf back up to root
     751 
     752 If a placement transform is present on the root node, that 
     753 is also collected. 
     754 
     755 * NB these are the CSG nodes, not structure nodes
     756 
     757 **/
     759 const nmat4triple* nnode::global_transform(nnode* n)
     760 {
     761     std::vector<const nmat4triple*> tvq ;
     762     nnode* r = NULL ;
     763     while(n)
     764     {
     765         if(n->transform) tvq.push_back(n->transform);
     766         r = n ;            // keep hold of the last non-NULL 
     767         n = n->parent ;
     768     }
     769 
     770     if(r->placement) tvq.push_back(r->placement);
     771 
     772 
     773     bool reverse = true ;
     774     const nmat4triple* gtransform= tvq.size() == 0 ? NULL : nmat4triple::product(tvq, reverse) ;
     775 
     776     if(gtransform == NULL )  // make sure all primitives have a gtransform 
     777     {
     778         gtransform = nmat4triple::make_identity() ;
     779     }
     780     return gtransform ;
     781 }



More modern transform handling (for structure) in stree::get_m2w_product

* need something similar for CSG snd starting with get_ancestors following parent links 


* HMM is G4Ellipsoid scale Xform added ? YEP snd::SetNodeXForm(root, scale ); 


::

    In [15]: f.csg.node.shape
    Out[15]: (624, 16)

    In [12]: f.csg.node[:,snd.xf].min(), f.csg.node[:,snd.xf].max()   # the snd refs the xform 
    Out[12]: (-1, 239)

    In [9]: f.csg.xform.shape
    Out[9]: (240, 16)

    In [7]: np.unique( f.csg.node[:,snd.xf], return_counts=True )  # many -1 "null" but only one of 0 to 239
    Out[7]: 
    (array([ -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36, ...
            232, 233, 234, 235, 236, 237, 238, 239], dtype=int32),
     array([389,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
              1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
              1,   1,   1,   1,   1,   1,   1,   1]))






persisted stree.h looks to have lots of debugging extras not in CSGFoundry
------------------------------------------------------------------------------ 

* TODO: review U4Tree creation of the stree
* TODO: document stree.h arrays in constexpr notes
* TODO: make non-essentials optional in the persisted folder
* TODO: comparing transforms with CSGFoundry ones, work out how to CSGImport 

::

    f.base:/Users/blyth/.opticks/GEOM/J007/CSGFoundry/SSim/stree

      : f.inst                                             :        (48477, 4, 4) : 4 days, 19:50:27.711199 
      : f.iinst                                            :        (48477, 4, 4) : 4 days, 19:51:11.766482 
      : f.inst_f4                                          :        (48477, 4, 4) : 4 days, 19:50:15.080793 
      : f.iinst_f4                                         :        (48477, 4, 4) : 4 days, 19:51:03.021160 
      : f.inst_nidx                                        :             (48477,) : 4 days, 19:50:14.668612 

      : f.nds                                              :         (422092, 14) : 4 days, 19:49:13.288836 
      : f.gtd                                              :       (422092, 4, 4) : 4 days, 19:51:24.749130 
      : f.m2w                                              :       (422092, 4, 4) : 4 days, 19:49:56.715706 
      : f.w2m                                              :       (422092, 4, 4) : 4 days, 19:48:49.592172 
      : f.subs                                             :               422092 : 4 days, 19:49:10.533278 
      : f.digs                                             :               422092 : 4 days, 19:51:53.722598 

      : f.rem                                              :           (3089, 14) : 4 days, 19:49:12.776762 
      : f.factor                                           :              (9, 12) : 4 days, 19:51:53.722111 

      : f.sensor_id                                        :             (46116,) : 4 days, 19:49:11.631723 

      : f.csg                                              :                 None : 4 days, 19:48:49.587808 
      : f.soname                                           :                  150 : 4 days, 19:49:11.630989 

      : f.bd                                               :              (54, 4) : 4 days, 19:52:00.140095 
      : f.bd_names                                         :                   54 : 4 days, 19:52:00.139822 

      : f.surface                                          :                 None : 4 days, 19:48:49.258335 
      : f.suname                                           :                   46 : 4 days, 19:49:10.532493 
      : f.suindex                                          :                (46,) : 4 days, 19:49:10.532860 

      : f.material                                         :                 None : 4 days, 19:48:49.587773 
      : f.mtline                                           :                (20,) : 4 days, 19:49:56.714553 
      : f.mtname                                           :                   20 : 4 days, 19:49:56.714169 
      : f.mtindex                                          :                (20,) : 4 days, 19:49:56.714992 



Back to ct:CSGFoundryAB.sh 
---------------------------

Extra meshname in A from the unbalanced alt (names appear twice)::

    In [15]: A.meshname[149:]
    Out[15]: ['sWorld', 'solidSJReceiverFastern', 'uni1']

    In [16]: B.meshname[149:]
    Out[16]: ['sWorld0x59dfbe0']


* DONE: trim the 0x ref in B 


Comparing CSGFoundry
----------------------

* lvid 93, 99 are very different : A balanced, B not  
* A also positivized, B not 

::

    In [12]: A.base
    Out[12]: '/Users/blyth/.opticks/GEOM/J007/CSGFoundry'

    In [13]: B.base
    Out[13]: '/tmp/blyth/opticks/CSGImportTest/CSGFoundry'

    In [15]: A.prim.shape, B.prim.shape
    Out[15]: ((3259, 4, 4), (3259, 4, 4))

    In [16]: A.node.shape, B.node.shape
    Out[16]: ((23547, 4, 4), (25435, 4, 4))

    In [11]: np.c_[A.prim.view(np.int32)[:,0,:2],B.prim.view(np.int32)[:,0,:2]][:2200]
    Out[11]: 
    array([[    1,     0,     1,     0],
           [    1,     1,     1,     1],
           [    1,     2,     1,     2],
           [    3,     3,     3,     3],
           [    3,     6,     3,     6],
           ...,
           [    7, 14149,     7, 14149],
           [    7, 14156,     7, 14156],
           [    7, 14163,     7, 14163],
           [    7, 14170,     7, 14170],
           [    7, 14177,     7, 14177]], dtype=int32)


    In [18]: np.c_[A.prim.view(np.int32)[:,1],B.prim.view(np.int32)[:,1]][:1000]
    Out[18]: 
    array([[  0, 149,   0,   0,   0,  -1,   0,   0],
           [  1,  17,   0,   1,   1,  -1,   0,   0],
           [  2,   2,   0,   2,   2,  -1,   0,   0],
           [  3,   1,   0,   3,   3,  -1,   0,   0],
           [  4,   0,   0,   4,   4,  -1,   0,   0],
           ...,
           [995,  45,   0, 995, 995,  -1,   0,   0],
           [996,  45,   0, 996, 996,  -1,   0,   0],
           [997,  45,   0, 997, 997,  -1,   0,   0],
           [998,  45,   0, 998, 998,  -1,   0,   0],
           [999,  45,   0, 999, 999,  -1,   0,   0]], dtype=int32)


9 prim have mismatched numNode (8 are contiguous primIdx from ridx 0)::

    In [12]: mi = np.where( A.prim.view(np.int32)[:,0,0]  != B.prim.view(np.int32)[:,0,0] ) ; mi 
    Out[13]: (array([2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 3126]),)

    In [22]: np.c_[A.prim.view(np.int32)[mi,0],B.prim.view(np.int32)[mi,0]]
    Out[22]: 
    array([[[   15, 15209,  6672,     0,   127, 15209,     0,     0],
            [   15, 15224,  6680,     0,   127, 15336,     0,     0],
            [   15, 15239,  6688,     0,   127, 15463,     0,     0],
            [   15, 15254,  6696,     0,   127, 15590,     0,     0],
            [   15, 15269,  6704,     0,   127, 15717,     0,     0],
            [   15, 15284,  6712,     0,   127, 15844,     0,     0],
            [   15, 15299,  6720,     0,   127, 15971,     0,     0],
            [   15, 15314,  6728,     0,   127, 16098,     0,     0],
            [   31, 23372,  8032,     0,  1023, 24268,     0,     0]]], dtype=int32)

    ## numNode much bigger for B  (is A using CSG_CONTIGUOUS?)
    ## Looks like A is balanced but B is not. 
    ##
    ## B misses the tranOffset                                                          


    In [18]: A.meshname[93]
    Out[18]: 'solidSJReceiverFastern'

    In [19]: B.meshname[93]
    Out[19]: 'solidSJReceiverFastern0x5bc98c0'

    In [21]: A.meshname[99]
    Out[21]: 'uni1'

    In [20]: B.meshname[99]
    Out[20]: 'uni10x5a93440'


    In [17]: np.c_[A.prim.view(np.int32)[mi,1],B.prim.view(np.int32)[mi,1]]
    Out[17]: 
    array([[[2375,   93,    0, 2375, 2375,   93,    0, 2375],
            [2376,   93,    0, 2376, 2376,   93,    0, 2376],
            [2377,   93,    0, 2377, 2377,   93,    0, 2377],
            [2378,   93,    0, 2378, 2378,   93,    0, 2378],
            [2379,   93,    0, 2379, 2379,   93,    0, 2379],
            [2380,   93,    0, 2380, 2380,   93,    0, 2380],
            [2381,   93,    0, 2381, 2381,   93,    0, 2381],
            [2382,   93,    0, 2382, 2382,   93,    0, 2382],
            [   0,   99,    6,    0,    0,   99,    6,    0]]], dtype=int32)

    ## matched: sbtIndexOffset, meshIdx, repeatIdx, primIdx  


    In [25]: A.node[15209:15209+15].view(np.int32)[:,3,2:]
    Out[25]: 
    array([[          1,           0],
           [          1,           0],
           [          1,           0],
           [          1,           0],
           [          1,           0],
           [          2,           0],
           [          1,           0],
           [        110,        6673],
           [        110,        6674],
           [        110,        6675],
           [        110,        6676],
           [        105,        6677],
           [        105, -2147476970],
           [        110,        6679],
           [        110,        6680]], dtype=int32)

           110:box3 105:cyl 1:uni 2:intersect


    In [26]: B.node[15209:15209+127].view(np.int32)[:,3,2:]
    Out[26]: 
    array([[  1,   0],
           [  1,   0],
           [  1,   0],
           [  1,   0],
           [110,   0],
           [110,   0],
           [110,   0],
           [  1,   0],
           [110,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  1,   0],
           [110,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           ...




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



