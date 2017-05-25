Subtree Instances Missing Transform
======================================

FIX/WORKAROUND not yet sure which
-----------------------------------

Marking all meshes to be "global" gives
them guaranteed gtransform slots GPU side 
(see NCSG::import_r)  which then allows placement transforms to have 
an effect (see GParts::applyPlacementTransform).


::

    615 void NScene::markGloballyUsedMeshes_r(nd* n)
    616 {
    617     assert( n->repeatIdx > -1 );
    618     
    619     //if(n->repeatIdx == 0) setIsUsedGlobally(n->mesh, true );
    620     setIsUsedGlobally(n->mesh, true );
    621     
    622     for(nd* c : n->children) markGloballyUsedMeshes_r(c) ;
    623 }
    624 


ISSUE : GMergedMesh subtree assemblies missing transform ?
------------------------------------------------------------

PMT model frame z-shift transforms are not correctly applied when 
doing GMergedMesh of the PMT assembly of 5 solids.

When instancing a subtree assembly need to bake the subtree-root relative
transform into the geometry, that aint happening correctly.

Where is the subtree relative transform being baked into the analytic
CSG list of ~5 trees ?

::

    tgltf-;tgltf-gdml --restrictmesh 0   # crazy too big placeholder bbox to fix

    tgltf-;tgltf-gdml --restrictmesh 2

    tboolean-;tboolean-pmt 



A similar issue for global ridx 0 solids was handled with isUsedGlobally
---------------------------------------------------------------------------

::

    036 NScene::NScene(const char* base, const char* name, const char* config, int scene_idx)
     37    :
     38     NGLTF(base, name, config, scene_idx),
     39     m_verbosity(0),
     40     m_num_global(0),
     41     m_num_csgskip(0),
     42     m_node_count(0),
     43     m_label_count(0),
     44     m_digest_count(new Counts<unsigned>("progenyDigest"))
     45 {
     46     load_asset_extras();
     47     load_csg_metadata();
     48 
     49     m_root = import_r(0, NULL, 0);
     50 
     51     if(m_verbosity > 1)
     52     dumpNdTree("NScene::NScene");
     53 
     54     compare_trees();
     55 
     56     count_progeny_digests();
     57 
     58     find_repeat_candidates();
     59 
     60     dump_repeat_candidates();
     61 
     62     labelTree();
     63 
     64     if(m_verbosity > 1)
     65     dumpRepeatCount();
     66 
     67     markGloballyUsedMeshes_r(m_root);
     68 
     69     // move load_mesh_extras later so can know which meshes are non-instanced needing 
     70     // gtransform slots for all primitives
     71     load_mesh_extras();
     72 
     73 }



    614 
    615 void NScene::markGloballyUsedMeshes_r(nd* n)
    616 {
    617     assert( n->repeatIdx > -1 );
    618     if(n->repeatIdx == 0) setIsUsedGlobally(n->mesh, true );
    619 
    620     for(nd* c : n->children) markGloballyUsedMeshes_r(c) ;
    621 }
    622 


The upshot of the marking is to always have a gtransform slot for all primitives, 
so they can be transformed later by changing the transforms::

    503 nnode* NCSG::import_r(unsigned idx, nnode* parent)
    504 {
    505     if(idx >= m_num_nodes) return NULL ;
    506 
    507     OpticksCSG_t typecode = (OpticksCSG_t)getTypeCode(idx);
    508     int transform_idx = getTransformIndex(idx) ;
    509     bool complement = isComplement(idx) ;
    510 
    511     LOG(debug) << "NCSG::import_r"
    512               << " idx " << idx
    513               << " transform_idx " << transform_idx
    514               << " complement " << complement
    515               ;
    516 
    517 
    518     nnode* node = NULL ;
    519 
    520     if(typecode == CSG_UNION || typecode == CSG_INTERSECTION || typecode == CSG_DIFFERENCE)
    521     {
    522         node = import_operator( idx, typecode ) ;
    523         node->parent = parent ;
    524 
    525         node->transform = import_transform_triple( transform_idx ) ;
    526 
    527         node->left = import_r(idx*2+1, node );
    528         node->right = import_r(idx*2+2, node );
    529 
    530         // recursive calls after "visit" as full ancestry needed for transform collection once reach primitives
    531     }
    532     else
    533     {
    534         node = import_primitive( idx, typecode );
    535         node->parent = parent ;                // <-- parent hookup needed prior to gtransform collection 
    536 
    537         node->transform = import_transform_triple( transform_idx ) ;
    538 
    539         nmat4triple* gtransform = node->global_transform();
    540         if(gtransform == NULL && m_usedglobally)
    541         {
    542             gtransform = nmat4triple::make_identity() ;
    543         }
    544 
    545         unsigned gtransform_idx = gtransform ? addUniqueTransform(gtransform) : 0 ;
    546 
    547         node->gtransform = gtransform ;
    548         node->gtransform_idx = gtransform_idx ; // 1-based, 0 for None
    549     }
    550     assert(node);
    551     node->idx = idx ;
    552     node->complement = complement ;
    553 
    554     return node ;
    555 }

    114 // meshes that are used globally need to have gtransform slots for all primitives
    115 bool NGLTF::isUsedGlobally(unsigned mesh_idx)
    116 {
    117     assert( m_mesh_used_globally.count(mesh_idx) == 1 );
    118     return m_mesh_used_globally[mesh_idx] ;
    119 }
    120 
    121 void NGLTF::setIsUsedGlobally(unsigned mesh_idx, bool iug)
    122 {
    123     m_mesh_used_globally[mesh_idx] = iug ;
    124 }

::

    simon:opticksnpy blyth$ grep setIsUsedGlobally *.*
    NCSG.cpp:void NCSG::setIsUsedGlobally(bool usedglobally )
    NCSG.cpp:     tree->setIsUsedGlobally(usedglobally);
    NCSG.hpp:        void setIsUsedGlobally(bool usedglobally);
    NGLTF.cpp:void NGLTF::setIsUsedGlobally(unsigned mesh_idx, bool iug)
    NGLTF.hpp:        void                         setIsUsedGlobally(unsigned mesh_idx, bool iug);
    NScene.cpp:    if(n->repeatIdx == 0) setIsUsedGlobally(n->mesh, true );
    simon:opticksnpy blyth$ 
