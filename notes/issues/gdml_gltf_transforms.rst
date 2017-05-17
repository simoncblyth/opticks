GDML-glTF Geometry Route Transform Debug
===========================================



Review Transforms
------------------

Transform heirarchies are used at both the 
local distinct CSG solid(mesh) level and at the structural level.  
Instancing collects the structural transforms of meshes that pass
the instancing criteria (eg based on number of repeats). 

Meshes that do not pass instancing criteria (as not enough of them) 
are clumped together into a global mesh, this means that the 
structural transform heirarchy down to the mesh (placement transform)
needs to be applied directly to the CSG geometry. 

CSG and Structural Transform workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSG transforms are collected from the source GDML boolean secondtransform
and set as transforms of ncsg nodes within python. csg.serialize
collects the transforms (still only local level transforms) into 
the transforms buffer and index references to the transforms are written 
into the part buffer.
A similar workflow happens for planes referenced from convex polyhedra.

Structural transforms are collected from source GDML PhysVol by gdml.py 
into the wrapped GDML model.  These are rearranged into homogenous node
tree by treebase.py which is just the structural tree, not decending into the 
CSG solids. The recursive sc.py sc.add_tree_gdml collects node structure 
and level transforms into the Nd tree matrix attributes. The Nd matrix 
are persisted into the gltf node tree.  

The CSG parts, transforms and planes are treated as GLTF extras saved as
sidecars to the gltf json file that holds the structural node tree with 
matrix attributes.

npy-/NScene/NGLTF reads in the gltf and reconstructs the nd tree, also the 
CSG extras are loaded with NScene::load_mesh_extras and stored as
NCSG objects 

GScene::importMeshes(NScene* scene)
    uses the polygonized CSG triangles to form GMesh

GSolid* GScene::createVolumeTree_r(nd* n, GSolid* parent)
    creates GNode/GSolid tree with associated GMesh  

Currently the structural global transforms for all nodes are computed in NScene::import_r, 
thats too soon as need a structural transform on the top for geometry 
that does not meet the instancing criteria.

::

    086 nd* NScene::import_r(int idx,  nd* parent, int depth)
     87 {
     88     ygltf::node_t* node = getNode(idx);
     89     auto extras = node->extras ;
     90     std::string boundary = extras["boundary"] ;
     91 
     92     nd* n = new nd ;
     93 
     94     n->idx = idx ;
     95     n->mesh = node->mesh ;
     96     n->parent = parent ;
     97     n->depth = depth ;
     98     n->boundary = boundary ;
     99     n->transform = new nmat4triple( node->matrix.data() );
    100     n->gtransform = nd::make_global_transform(n) ;
    101 
    102     for(auto child : node->children) n->children.push_back(import_r(child, n, depth+1));  // recursive call
    103 
    104     m_nd[idx] = n ;
    105 
    106     return n ;
    107 }


The nd tree transforms are currently passed thru unchanged into the GNode/GSolid tree

* this may be a good place to disperse placement transforms for global geometry 
  ... but then need to move the repeat index decision/labelling earlier down into NScene/nd,
  then GScene::createVolume can use the repeat index from nd tree  

Note that it may become necessary to manually set ridx, so dont assume 
some criteria in more than one place ... apply the criteria and write ridx into the tree and then
act upon the ridx in the tree.


::

    268 GSolid* GScene::createVolume(nd* n)
    269 {
    270     assert(n);
    271 
    272     unsigned node_idx = n->idx ;
    273     unsigned mesh_idx = n->mesh ;
    274     std::string bnd = n->boundary ;
    275 
    276     const char* spec = bnd.c_str();
    277 
    278     LOG(debug) << "GScene::createVolume"
    279               << " node_idx " << std::setw(5) << node_idx
    280               << " mesh_idx " << std::setw(3) << mesh_idx
    281               << " bnd " << bnd
    282               ;
    283 
    284     assert(!bnd.empty());
    285 
    286     GMesh* mesh = getMesh(mesh_idx);
    287 
    288     NCSG* csg = getCSG(mesh_idx);
    289 
    290 
    291     glm::mat4 xf_global = n->gtransform->t ;
    292 
    293     glm::mat4 xf_local  = n->transform->t ;
    294 
    295     GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));
    296 
    297     GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(xf_local));
    298 
    299 
    300     GSolid* solid = new GSolid(node_idx, gtransform, mesh, UINT_MAX, NULL );
    301 
    302     solid->setLevelTransform(ltransform);




GScene::labelTree_r(GNode* node)  
     applies the instancing criteria setting ridx RepeatIndex for all nodes of the tree

GScene::makeMergedMeshAndInstancedBuffers
      acts on the ridx labels, creating the merged mesh and instancing buffers 


The NCSG GTransformsBuffer of CSG "global" transforms (global to the CSG tree, not to the full geometry)
for each node of the CSG tree lives on inside the GParts instance that is associated to every GNode::

    NPY<float>* tranbuf = tree->getGTransformBuffer();


Triggered by GScene::makeMergedMeshAndInstancedBuffers the separate GParts are combined within
GMergedMesh::mergeSolid providing the transform offsets for each primitive, allowing lookup of 
transforms from GPU.

Need to apply placement transforms to all the nodes in the CSG trees for the un-instanced 
geometry. This doesnt change referencing, just all transforms for each node tree.



4 nodes, three meshes::

    delta:issues blyth$ tgltf-;tgltf-gdml-
    args: 
    [2017-05-17 10:18:23,278] p23902 {/Users/blyth/opticks/ana/pmt/gdml.py:948} INFO - wrapping gdml element  
    sc.py:add_node_gdml nodeIdx:   0 lvIdx: 2 soName:                  oil0xbf5ed48 lvName:/dd/Geometry/AD/lvOIL0xbf5e0b8 
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]], dtype=float32)
    sc.py:add_node_gdml nodeIdx:   1 lvIdx: 0 soName:             pmt-hemi0xc0fed90 lvName:/dd/Geometry/PMT/lvPmtHemi0xc133740 
    array([[    0.    ,    -0.    ,     1.    ,     0.    ],
           [    0.1305,    -0.9914,    -0.    ,     0.    ],
           [    0.9914,     0.1305,     0.    ,     0.    ],
           [-2304.6135,  -303.4081, -1750.    ,     1.    ]], dtype=float32)
    sc.py:add_node_gdml nodeIdx:   2 lvIdx: 1 soName:          AdPmtCollar0xc2c5260 lvName:/dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0 
    array([[    0.    ,    -0.    ,     1.    ,     0.    ],
           [    0.1305,    -0.9914,    -0.    ,     0.    ],
           [    0.9914,     0.1305,     0.    ,     0.    ],
           [-2249.0928,  -296.0987, -1750.    ,     1.    ]], dtype=float32)
    sc.py:add_node_gdml nodeIdx:   3 lvIdx: 0 soName:             pmt-hemi0xc0fed90 lvName:/dd/Geometry/PMT/lvPmtHemi0xc133740 
    array([[    0.    ,    -0.    ,     1.    ,     0.    ],
           [    0.3827,    -0.9239,    -0.    ,     0.    ],
           [    0.9239,     0.3827,     0.    ,     0.    ],
           [-2147.5579,  -889.5477, -1750.    ,     1.    ]], dtype=float32)
    [2017-05-17 10:18:23,283] p23902 {/Users/blyth/opticks/dev/csg/sc.py:230} INFO - saving to /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf 
    [2017-05-17 10:18:23,286] p23902 {/Users/blyth/opticks/dev/csg/sc.py:225} INFO - save_extras /tmp/blyth/opticks/tgltf/extras  : saved 3 
    /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf


All three CSG trees have at least one transform::

    delta:issues blyth$ head -1 /tmp/blyth/opticks/tgltf/extras/0/transforms.npy 
    ?NUMPYF{'descr': '<f4', 'fortran_order': False, 'shape': (4, 4, 4), }       
    delta:issues blyth$ 
    delta:issues blyth$ head -1 /tmp/blyth/opticks/tgltf/extras/*/transforms.npy 
    ==> /tmp/blyth/opticks/tgltf/extras/0/transforms.npy <==
    ?NUMPYF{'descr': '<f4', 'fortran_order': False, 'shape': (4, 4, 4), }       

    ==> /tmp/blyth/opticks/tgltf/extras/1/transforms.npy <==
    ?NUMPYF{'descr': '<f4', 'fortran_order': False, 'shape': (1, 4, 4), }       

    ==> /tmp/blyth/opticks/tgltf/extras/2/transforms.npy <==
    ?NUMPYF{'descr': '<f4', 'fortran_order': False, 'shape': (1, 4, 4), }       
    delta:issues blyth$ 



Perhaps simpler not to attempt to change transforms, rather get 
them correct first time via providing a top transform input to the 
NCSG::import_r which is driven by NCSG::LoadTree.

But, NCSG::LoadTree is done at mesh level(not node level) by NScene::load_mesh_extras
... so looks like need to stay with node level modification.

When a mesh fails the instancing criteria every primitive needs to 
have a gtransform (ie must not suppress identity) as need somewhere to 
apply the placement transform ... unless arrange another place for this, hmm 
maybe simpler than diddling in the node tree. BUT must diddle there to avoid more
work on GPU. 

So can I know that before NScene::load_mesh_extras and ensure not to 
suppress gtransform slots for all primitive in that case ?


::

    480 nnode* NCSG::import_r(unsigned idx, nnode* parent)
    481 {
    482     if(idx >= m_num_nodes) return NULL ;
    483 
    484     OpticksCSG_t typecode = (OpticksCSG_t)getTypeCode(idx);     
    485     int transform_idx = getTransformIndex(idx) ;
    486     bool complement = isComplement(idx) ;
    487 
    488     LOG(debug) << "NCSG::import_r"
    489               << " idx " << idx
    490               << " transform_idx " << transform_idx
    491               << " complement " << complement
    492               ;
    493 
    494 
    495     nnode* node = NULL ;
    496 
    497     if(typecode == CSG_UNION || typecode == CSG_INTERSECTION || typecode == CSG_DIFFERENCE)
    498     {
    499         node = import_operator( idx, typecode ) ;
    500         node->parent = parent ;
    501 
    502         node->transform = import_transform_triple( transform_idx ) ;
    503 
    504         node->left = import_r(idx*2+1, node );
    505         node->right = import_r(idx*2+2, node );
    506 
    507         // recursive calls after "visit" as full ancestry needed for transform collection once reach primitives
    508     }
    509     else
    510     {
    511         node = import_primitive( idx, typecode );
    512         node->parent = parent ;                // <-- parent hookup needed prior to gtransform collection 
    513 
    514         node->transform = import_transform_triple( transform_idx ) ;
    515 
    516         nmat4triple* gtransform = node->global_transform();

                ^^^^^^^^^^^^^^^^ this often gets identity suppressed stymie-ing the placement transform approach ^^^^^^^^^^^^^

    517         unsigned gtransform_idx = gtransform ? addUniqueTransform(gtransform) : 0 ;
    518         node->gtransform = gtransform ;
    519         node->gtransform_idx = gtransform_idx ; // 1-based, 0 for None
    520     }
    521     assert(node);
    522     node->idx = idx ;
    523     node->complement = complement ;
    524 
    525     return node ;
    526 }



Hmm getting no gtransforms for two of the meshes, so nowhere to apply the placement transform::

    NScene::load_mesh_extras num_meshes 3
     mid    0 prm    1 nam                                    /dd/Geometry/AD/lvOIL0xbf5e0b8 smry  ht  0 nn    1 tri  17884 tmsg             nd 1,4,4  tr 1,3,4,4 gtr 0,3,4,4 pln NULL
     mid    1 prm    1 nam                               /dd/Geometry/PMT/lvPmtHemi0xc133740 smry  ht  3 nn   15 tri   8308 tmsg             nd 15,4,4 tr 4,3,4,4 gtr 3,3,4,4 pln NULL
     mid    2 prm    1 nam                           /dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0 smry  ht  1 nn    3 tri     12 tmsg PLACEHOLDER nd 3,4,4  tr 1,3,4,4 gtr 0,3,4,4 pln NULL
    2017-05-17 12:05:31.607 INFO  [1806476] [NScene::dumpNdTree@120] NScene::NScene


After arrange to always have gtransforms slots for meshes that are used globally::

    NScene::load_mesh_extras num_meshes 3
     mid    0 prm    1 nam                                    /dd/Geometry/AD/lvOIL0xbf5e0b8 iug 1 smry  ht  0 nn    1 tri  17884 tmsg  iug 1 nd 1,4,4 tr 1,3,4,4 gtr 1,3,4,4 pln NULL
     mid    1 prm    1 nam                               /dd/Geometry/PMT/lvPmtHemi0xc133740 iug 1 smry  ht  3 nn   15 tri   8308 tmsg  iug 1 nd 15,4,4 tr 4,3,4,4 gtr 4,3,4,4 pln NULL
     mid    2 prm    1 nam                           /dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0 iug 1 smry  ht  1 nn    3 tri     12 tmsg PLACEHOLDER iug 1 nd 3,4,4 tr 1,3,4,4 gtr 1,3,4,4 pln NULL
    2017-05-17 13:43:43.718 INFO  [1838644] [GScene::importMeshes@60] GScene::importMeshes num_meshes 3




Issue : top level (non-instanced) transforms ignored by raytrace
------------------------------------------------------------------


* ray trace not handling transforms applied to global geometry ie non-instanced


Using gdml_builder to make a partial geometry with just the oil 
and 2 PMTs (using 2 to be beneath the instancing cut in)

* other than PMT reverse direction, rasterized looks OK : with the 2 PMTs located near the edge of the oil cylinder
* raytrace shows single PMT at origin (presumably 2 on top of each other) ignoring the top level transform

* increasing the number beyond instancing cut in (at 4) the PMTs and Collars adopt their positions, 
  (still reverse pointing)

::

    tgltf-;tgltf-gdml 




FIXED : PMTs pointing in reverse direction !
------------------------------------------------

* fixed by transposing the rotation matrix relative to that obtained from 
  the numpy translation of glm::rotate in glm.py 

* TODO: motivate the transform better that it looks right :
  ie look for documentation of rotation matrix conventions used in Geant4 and OpenGL/GLM


::

    tgltf-
    tgltf--

See:

* npy-/tests/NGLMTest.cc:test_axisAngle
* dev/csg/sc_transform_check.py 


Red Herring
~~~~~~~~~~~~~


Permuting axes (X,Y,Z)->(Y,Z,X) leads to much more reasonable interpretation 
of the txf transforms.  This is suggestive that a PMT orienting 
transform (to adjust from model frame with +Z in PMT pointing direction)
is being applied after PMT ring rotatations. 

::

     76         glm::mat4 trs2(1.f) ;
     77         trs2[0] = trs[1] ;  //  Y->X
     78         trs2[1] = trs[2] ;  //  Z->Y
     79         trs2[2] = trs[0] ;  //  X->Z
     80         trs2[3] = trs[3] ;
     81 
     82         //  ( X,Y,Z ) -> ( Y,Z,X )
     83         


Take axes for a spin::

    In [28]: from glm import rotate

    In [30]: rot = rotate([1,1,1,360./3.] )

    In [31]: rot
    Out[31]: 
    array([[-0.,  1., -0.,  0.],
           [-0., -0.,  1.,  0.],
           [ 1., -0., -0.,  0.],
           [ 0.,  0.,  0.,  1.]], dtype=float32)


    In [32]: rot = rotate([1,1,1,-360./3.] )

    In [33]: rot
    Out[33]: 
    array([[-0., -0.,  1.,  0.],      // Z->X
           [ 1., -0., -0.,  0.],      // X->Y
           [-0.,  1., -0.,  0.],      // Y->Z
           [ 0.,  0.,  0.,  1.]], dtype=float32)



::

    * txf: 8,24,4,4
    ( 0, 0) {    0.0000    0.0000    1.0000} 1.7017 (  {   -0.13    0.99    0.00    0.00} {   -0.99   -0.13    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 1) {    0.0000    0.0000    1.0000} 1.9635 (  {   -0.38    0.92    0.00    0.00} {   -0.92   -0.38    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 2) {    0.0000    0.0000    1.0000} 2.2253 (  {   -0.61    0.79    0.00    0.00} {   -0.79   -0.61    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 3) {    0.0000    0.0000    1.0000} 2.4871 (  {   -0.79    0.61    0.00    0.00} {   -0.61   -0.79    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 4) {    0.0000    0.0000    1.0000} 2.7489 (  {   -0.92    0.38    0.00    0.00} {   -0.38   -0.92    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 5) {    0.0000    0.0000    1.0000} 3.0107 (  {   -0.99    0.13    0.00    0.00} {   -0.13   -0.99    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 6) {   -0.0000   -0.0000   -1.0000} 3.0107 (  {   -0.99   -0.13    0.00    0.00} {    0.13   -0.99    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 7) {   -0.0000   -0.0000   -1.0000} 2.7489 (  {   -0.92   -0.38    0.00    0.00} {    0.38   -0.92    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 8) {   -0.0000   -0.0000   -1.0000} 2.4871 (  {   -0.79   -0.61    0.00    0.00} {    0.61   -0.79    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 9) {   -0.0000   -0.0000   -1.0000} 2.2253 (  {   -0.61   -0.79    0.00    0.00} {    0.79   -0.61    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,10) {   -0.0000   -0.0000   -1.0000} 1.9635 (  {   -0.38   -0.92    0.00    0.00} {    0.92   -0.38    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,11) {   -0.0000   -0.0000   -1.0000} 1.7017 (  {   -0.13   -0.99    0.00    0.00} {    0.99   -0.13    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,12) {    0.0000    0.0000   -1.0000} 1.4399 (  {    0.13   -0.99   -0.00    0.00} {    0.99    0.13    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,13) {    0.0000    0.0000   -1.0000} 1.1781 (  {    0.38   -0.92   -0.00    0.00} {    0.92    0.38    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,14) {    0.0000    0.0000   -1.0000} 0.9163 (  {    0.61   -0.79   -0.00    0.00} {    0.79    0.61    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,15) {    0.0000    0.0000   -1.0000} 0.6545 (  {    0.79   -0.61   -0.00    0.00} {    0.61    0.79    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,16) {    0.0000    0.0000   -1.0000} 0.3927 (  {    0.92   -0.38   -0.00    0.00} {    0.38    0.92    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,17) {    0.0000    0.0000   -1.0000} 0.1309 (  {    0.99   -0.13   -0.00    0.00} {    0.13    0.99    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,18) {    0.0000    0.0000    1.0000} 0.1309 (  {    0.99    0.13   -0.00    0.00} {   -0.13    0.99    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,19) {    0.0000    0.0000    1.0000} 0.3927 (  {    0.92    0.38   -0.00    0.00} {   -0.38    0.92    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,20) {    0.0000    0.0000    1.0000} 0.6545 (  {    0.79    0.61   -0.00    0.00} {   -0.61    0.79    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,21) {    0.0000    0.0000    1.0000} 0.9163 (  {    0.61    0.79   -0.00    0.00} {   -0.79    0.61    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,22) {    0.0000    0.0000    1.0000} 1.1781 (  {    0.38    0.92   -0.00    0.00} {   -0.92    0.38    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,23) {    0.0000    0.0000    1.0000} 1.4399 (  {    0.13    0.99   -0.00    0.00} {   -0.99    0.13    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )


Y-Z swap::

    | 1 0 0 0 |
    | 0 0 1 0 |
    | 0 1 0 0 |
    | 0 0 0 1 |


    
GDML/glTF route
----------------

opticks.ana.pmt.gdml:GDML
    parse GDML input file into wrapped element object model, no structural manipulations : just wrapping 

opticks.ana.pmt.treebase:Tree
    restructures stripped LV/PV/LV/... volume tree into homogenous node tree (LV,PV)/(LV,PV)/...


GDML Stage
~~~~~~~~~~~~

::

    191
    Position mm -2304.61358026 303.408133816 1750.0 
    Rotation deg -90.0 -82.5 -90.0 

    <position xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:8#pvAdPmtInRing:24#pvAdPmtUnit#pvAdPmt0xc110bd8_pos" unit="mm" x="-2304.61358026342" y="303.408133815512" z="1750"/>
            
    <rotation xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:8#pvAdPmtInRing:24#pvAdPmtUnit#pvAdPmt0xc110bd8_rot" unit="deg" x="-90" y="-82.4999999999999" z="-90"/>
          
    [[    0.        -0.         1.         0.    ]
     [    0.1305     0.9914    -0.         0.    ]
     [   -0.9914     0.1305     0.         0.    ]
     [-2304.6135   303.4081  1750.         1.    ]]



In [24]: pmts[191].transform    ## with modified zyx order in glm.py:rotate_three_axis 
Out[24]: 
array([[    0.    ,    -0.1305,     0.9914,     0.    ],
       [    0.    ,    -0.9914,    -0.1305,     0.    ],
       [    1.    ,     0.    ,     0.    ,     0.    ],
       [-2304.6135,   303.4081,  1750.    ,     1.    ]], dtype=float32)

In [2]: pmts[191].transform     ## with the longstanding xyz order  
Out[2]: 
array([[    0.    ,    -0.    ,     1.    ,     0.    ],
       [    0.1305,     0.9914,    -0.    ,     0.    ],
       [   -0.9914,     0.1305,     0.    ,     0.    ],
       [-2304.6135,   303.4081,  1750.    ,     1.    ]], dtype=float32)



    In [18]: eulerAngleXYZ([-90.0,-82.5,-90.0])
    Out[18]: 
    array([[-0.    ,  0.    ,  1.    ,  0.    ],
           [ 0.1305,  0.9914,  0.    ,  0.    ],
           [-0.9914,  0.1305, -0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  1.    ]], dtype=float32)


    In [17]: eulerAngleXYZ([90.0,-82.5,90.0])
    Out[17]: 
    array([[-0.    , -0.    ,  1.    ,  0.    ],
           [-0.1305,  0.9914, -0.    ,  0.    ],
           [-0.9914, -0.1305, -0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  1.    ]], dtype=float32)




Probably the 3-axis rotation interpretation I am using to 
convert this into a transform : doesnt match the GDML intention ?


::

     00                      Rotation deg 90.0 -82.5 90.0  Position mm -2304.61358026 -303.408133816 -1750.0  
      1                      Rotation deg 90.0 -67.5 90.0  Position mm -2147.55797332 -889.547638533 -1750.0  
      2                      Rotation deg 90.0 -52.5 90.0  Position mm -1844.14983951 -1415.06594173 -1750.0  
      3                      Rotation deg 90.0 -37.5 90.0  Position mm -1415.06594173 -1844.14983951 -1750.0  
      4                      Rotation deg 90.0 -22.5 90.0  Position mm -889.547638533 -2147.55797332 -1750.0  
      5                       Rotation deg 90.0 -7.5 90.0  Position mm -303.408133816 -2304.61358026 -1750.0  
      6                        Rotation deg 90.0 7.5 90.0  Position mm 303.408133816 -2304.61358026 -1750.0  
      7                       Rotation deg 90.0 22.5 90.0  Position mm 889.547638533 -2147.55797332 -1750.0  
      8                       Rotation deg 90.0 37.5 90.0  Position mm 1415.06594173 -1844.14983951 -1750.0  
      9                       Rotation deg 90.0 52.5 90.0  Position mm 1844.14983951 -1415.06594173 -1750.0  
     10                       Rotation deg 90.0 67.5 90.0  Position mm 2147.55797332 -889.547638533 -1750.0  
     11                       Rotation deg 90.0 82.5 90.0  Position mm 2304.61358026 -303.408133816 -1750.0  
     12                     Rotation deg -90.0 82.5 -90.0  Position mm 2304.61358026 303.408133816 -1750.0  
     13                     Rotation deg -90.0 67.5 -90.0  Position mm 2147.55797332 889.547638533 -1750.0  
     14                     Rotation deg -90.0 52.5 -90.0  Position mm 1844.14983951 1415.06594173 -1750.0  
     15                     Rotation deg -90.0 37.5 -90.0  Position mm 1415.06594173 1844.14983951 -1750.0  
     16                     Rotation deg -90.0 22.5 -90.0  Position mm 889.547638533 2147.55797332 -1750.0  
     17                      Rotation deg -90.0 7.5 -90.0  Position mm 303.408133816 2304.61358026 -1750.0  
     18                     Rotation deg -90.0 -7.5 -90.0  Position mm -303.408133816 2304.61358026 -1750.0  
     19                    Rotation deg -90.0 -22.5 -90.0  Position mm -889.547638533 2147.55797332 -1750.0  
     20                    Rotation deg -90.0 -37.5 -90.0  Position mm -1415.06594173 1844.14983951 -1750.0  
     21                    Rotation deg -90.0 -52.5 -90.0  Position mm -1844.14983951 1415.06594173 -1750.0  
     22                    Rotation deg -90.0 -67.5 -90.0  Position mm -2147.55797332 889.547638533 -1750.0  
     23                    Rotation deg -90.0 -82.5 -90.0  Position mm -2304.61358026 303.408133816 -1750.0  
     24                      Rotation deg 90.0 -82.5 90.0  Position mm -2304.61358026 -303.408133816 -1250.0  
     25                      Rotation deg 90.0 -67.5 90.0  Position mm -2147.55797332 -889.547638533 -1250.0  
     26                      Rotation deg 90.0 -52.5 90.0  Position mm -1844.14983951 -1415.06594173 -1250.0  
     27                      Rotation deg 90.0 -37.5 90.0  Position mm -1415.06594173 -1844.14983951 -1250.0  
     28                      Rotation deg 90.0 -22.5 90.0  Position mm -889.547638533 -2147.55797332 -1250.0  
     29                       Rotation deg 90.0 -7.5 90.0  Position mm -303.408133816 -2304.61358026 -1250.0  
     30                        Rotation deg 90.0 7.5 90.0  Position mm 303.408133816 -2304.61358026 -1250.0  
     31                       Rotation deg 90.0 22.5 90.0  Position mm 889.547638533 -2147.55797332 -1250.0  
     32                       Rotation deg 90.0 37.5 90.0  Position mm 1415.06594173 -1844.14983951 -1250.0  
     33                       Rotation deg 90.0 52.5 90.0  Position mm 1844.14983951 -1415.06594173 -1250.0  
     34                       Rotation deg 90.0 67.5 90.0  Position mm 2147.55797332 -889.547638533 -1250.0  
     35                       Rotation deg 90.0 82.5 90.0  Position mm 2304.61358026 -303.408133816 -1250.0  
     36                     Rotation deg -90.0 82.5 -90.0  Position mm 2304.61358026 303.408133816 -1250.0  
     37                     Rotation deg -90.0 67.5 -90.0  Position mm 2147.55797332 889.547638533 -1250.0  
     38                     Rotation deg -90.0 52.5 -90.0  Position mm 1844.14983951 1415.06594173 -1250.0  



/usr/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/gdml/include/G4GDMLWriteDefine.hh::

     36 // History:
     37 // - Created.                                  Zoltan Torzsok, November 2007
     38 // -------------------------------------------------------------------------
     39 
     40 #ifndef _G4GDMLWRITEDEFINE_INCLUDED_
     41 #define _G4GDMLWRITEDEFINE_INCLUDED_
     42 
     43 #include "G4Types.hh"
     44 #include "G4ThreeVector.hh"
     45 #include "G4RotationMatrix.hh"
     46 
     47 #include "G4GDMLWrite.hh"
     48 
     49 class G4GDMLWriteDefine : public G4GDMLWrite
     50 {
     51 
     52   public:
     53 
     54     G4ThreeVector GetAngles(const G4RotationMatrix&);
     55     void ScaleWrite(xercesc::DOMElement* element,
     56                     const G4String& name, const G4ThreeVector& scl)
     57          { Scale_vectorWrite(element,"scale",name,scl); }
     58     void RotationWrite(xercesc::DOMElement* element,
     59                     const G4String& name, const G4ThreeVector& rot)
     60          { Rotation_vectorWrite(element,"rotation",name,rot); }
     61     void PositionWrite(xercesc::DOMElement* element,
     62                     const G4String& name, const G4ThreeVector& pos)
     63          { Position_vectorWrite(element,"position",name,pos); }
     64     void FirstrotationWrite(xercesc::DOMElement* element,
     65                     const G4String& name, const G4ThreeVector& rot)
     66          { Rotation_vectorWrite(element,"firstrotation",name,rot); }
     67     void FirstpositionWrite(xercesc::DOMElement* element,
     68                     const G4String& name, const G4ThreeVector& pos)


::

    simon:gdml blyth$ find . -type f -exec grep -H Rotation_vectorWrite {} \;
    ./include/G4GDMLWriteDefine.hh:         { Rotation_vectorWrite(element,"rotation",name,rot); }
    ./include/G4GDMLWriteDefine.hh:         { Rotation_vectorWrite(element,"firstrotation",name,rot); }
    ./include/G4GDMLWriteDefine.hh:    void Rotation_vectorWrite(xercesc::DOMElement*, const G4String&,
    ./src/G4GDMLWriteDefine.cc:Rotation_vectorWrite(xercesc::DOMElement* element, const G4String& tag,
    simon:gdml blyth$ 


::

    097 void G4GDMLWriteDefine::
     98 Rotation_vectorWrite(xercesc::DOMElement* element, const G4String& tag,
     99                      const G4String& name, const G4ThreeVector& rot)
    100 {
    101    const G4double x = (std::fabs(rot.x()) < kAngularPrecision) ? 0.0 : rot.x();
    102    const G4double y = (std::fabs(rot.y()) < kAngularPrecision) ? 0.0 : rot.y();
    103    const G4double z = (std::fabs(rot.z()) < kAngularPrecision) ? 0.0 : rot.z();
    104 
    105    xercesc::DOMElement* rotationElement = NewElement(tag);
    106    rotationElement->setAttributeNode(NewAttribute("name",name));
    107    rotationElement->setAttributeNode(NewAttribute("x",x/degree));
    108    rotationElement->setAttributeNode(NewAttribute("y",y/degree));
    109    rotationElement->setAttributeNode(NewAttribute("z",z/degree));
    110    rotationElement->setAttributeNode(NewAttribute("unit","deg"));
    111    element->appendChild(rotationElement);
    112 }


::

     51 G4ThreeVector G4GDMLWriteDefine::GetAngles(const G4RotationMatrix& mtx)
     52 {
     53    G4double x,y,z;
     54    G4RotationMatrix mat = mtx;
     55    mat.rectify();   // Rectify matrix from possible roundoff errors
     56 
     57    // Direction of rotation given by left-hand rule; clockwise rotation
     58 
     59    static const G4double kMatrixPrecision = 10E-10;
     60    const G4double cosb = std::sqrt(mtx.xx()*mtx.xx()+mtx.yx()*mtx.yx());
     ..                                       r11^2 + r21^2
     61 
     62    if (cosb > kMatrixPrecision)
     63    {
     64       x = std::atan2(mtx.zy(),mtx.zz());   
     ..                         r32      r33   
     65       y = std::atan2(-mtx.zx(),cosb);
     ..                        -r31 
     66       z = std::atan2(mtx.yx(),mtx.xx());
     ..                         r21     r11
     67    }
     68    else
     69    {
     70       x = std::atan2(-mtx.yz(),mtx.yy());
     71       y = std::atan2(-mtx.zx(),cosb);
     72       z = 0.0;
     73    }
     74 
     75    return G4ThreeVector(x,y,z);
     76 }



Decomposing Euler Angles

* http://nghiaho.com/?page_id=846


::

    simon:gdml blyth$ find . -type f -exec grep -H GetAngles {} \;
    ./include/G4GDMLWriteDefine.hh:    G4ThreeVector GetAngles(const G4RotationMatrix&);
    ./src/G4GDMLWriteDefine.cc:G4ThreeVector G4GDMLWriteDefine::GetAngles(const G4RotationMatrix& mtx)
    ./src/G4GDMLWriteParamvol.cc:   Angles=GetAngles(paramvol->GetObjectRotationValue());
    ./src/G4GDMLWriteParamvol.cc:                   GetAngles(paramvol->GetObjectRotationValue()));
    ./src/G4GDMLWriteSolids.cc:      G4ThreeVector rot = GetAngles(rotm);
    ./src/G4GDMLWriteSolids.cc:         firstrot += GetAngles(disp->GetObjectRotation());
    ./src/G4GDMLWriteSolids.cc:         rot += GetAngles(disp->GetObjectRotation());
    ./src/G4GDMLWriteStructure.cc:   const G4ThreeVector rot = GetAngles(rotate.getRotation());
    simon:gdml blyth$ 



::

    107 void G4GDMLWriteStructure::PhysvolWrite(xercesc::DOMElement* volumeElement,
    108                                         const G4VPhysicalVolume* const physvol,
    109                                         const G4Transform3D& T,
    110                                         const G4String& ModuleName)
    111 {
    112    HepGeom::Scale3D scale;
    113    HepGeom::Rotate3D rotate;
    114    HepGeom::Translate3D translate;
    115 
    116    T.getDecomposition(scale,rotate,translate);
    117 
    118    const G4ThreeVector scl(scale(0,0),scale(1,1),scale(2,2));
    119    const G4ThreeVector rot = GetAngles(rotate.getRotation());
    120    const G4ThreeVector pos = T.getTranslation();
    121 
    122    const G4String name = GenerateName(physvol->GetName(),physvol);
    123    const G4int copynumber = physvol->GetCopyNo();
    124 
    125    xercesc::DOMElement* physvolElement = NewElement("physvol");
    126    physvolElement->setAttributeNode(NewAttribute("name",name));
    127    if (copynumber) physvolElement->setAttributeNode(NewAttribute("copynumber", copynumber));
    128 
    129    volumeElement->appendChild(physvolElement);
    130 



GDML Manual Unhelpful
~~~~~~~~~~~~~~~~~~~~~~~~


3.2.5 Rotations

Rotations are usually defined in the beginning of the GDML file (in the define
section). Once defined, they can be referenced in place where rotations are
expected. Positive rotations are expected to be right-handed. A rotation can be
defined as in the following example:

::

   <rotation name="RotateZ" z=="30" unit="deg"/>


GLM Euler
-----------

/usr/local/opticks/externals/glm/glm-0.9.6.3/glm/gtx/euler_angles.inl

Translated  eulerAngleX, eulerAngleY, eulerAngleZ into my glm.py 





GDML/glTF Tracing Transforms
--------------------------------






 
