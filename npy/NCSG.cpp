/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cstring>
#include <csignal>
#include <algorithm>
#include <sstream>

#include "SSys.hh"

#include "BStr.hh"
#include "BFile.hh"
#include "BMeta.hh"

#include "NGLMExt.hpp"
#include "nmat4triple.hpp"
#include "GLMFormat.hpp"


#include "NTrianglesNPY.hpp"


#ifdef WITH_OPENMESH
#include "NPolygonizer.hpp"
#endif

#include "NPrimitives.hpp"

#include "NSceneConfig.hpp"
#include "NScan.hpp"
#include "NNode.hpp"
#include "NNodePoints.hpp"
#include "NNodeUncoincide.hpp"
#include "NNodeNudger.hpp"


#include "NBBox.hpp"
#include "NPY.hpp"
#include "NCSG.hpp"
#include "NCSGData.hpp"
#include "NPYMeta.hpp"

#include "PLOG.hh"

const plog::Severity NCSG::LEVEL = PLOG::EnvLevel("NCSG", "DEBUG") ; 

const float NCSG::SURFACE_EPSILON = 1e-5f ; 

const unsigned NCSG::MAX_EXPORT_HEIGHT = 16 ; 


/////////////////////////////////////////////////////////////////////////////////////////
/**
NCSG::Load
-----------

Loads a single CSG tree from a tree directory, as written 
by python opticks.analytic.csg:CSG.Serialize


**/

NCSG* NCSG::Load(const char* treedir)
{
    const char* config_ = "polygonize=0" ; 
    NSceneConfig* config = new NSceneConfig(config_); 
    return NCSG::Load(treedir, config); 
}

NCSG* NCSG::Load(const char* treedir, const char* gltfconfig)
{
    if(!NCSGData::Exists(treedir))
    {
         LOG(warning) << "NCSG::Load no such treedir " << treedir ;
         return NULL ; 
    }
    NSceneConfig* config = new NSceneConfig(gltfconfig) ; 
    NCSG* csg = NCSG::Load(treedir, config);
    assert(csg);
    return csg ; 
}

NCSG* NCSG::Load(const char* treedir, const NSceneConfig* config  )
{
    if(!NCSGData::Exists(treedir) )
    {
         LOG(debug) << "NCSG::Load no such treedir " << treedir ;
         return NULL ; 
    }

    NCSG* tree = new NCSG(treedir) ; 

    tree->setConfig(config);
    tree->setVerbosity(config->verbosity);
    tree->setIsUsedGlobally(true);

    tree->loadsrc();     // populate the src* buffers 
    tree->import();      // complete binary tree m_nodes buffer -> nnode tree
    tree->postchange();  // update result buffers  

    return tree ; 
}



/**
/////////////////////////////////////////////////////////////////////////////////////////
NCSG::Adopt
------------

Canonical usage in direct route is from X4PhysicalVolume::convertSolid

Note that as this starts from an in memory nnode tree there 
is no *loadsrc* or *import* only an *export* into the GPU ready 
complete binary tree buffers.

**/

NCSG* NCSG::Adopt(nnode* root)
{
    const char* config = "" ; 
    unsigned soIdx = 0 ; 
    unsigned lvIdx = 0 ; 
    return Adopt( root, config, soIdx , lvIdx ); 
}

NCSG* NCSG::Adopt(nnode* root, const char* config_, unsigned soIdx, unsigned lvIdx)
{
    LOG(LEVEL) << "[" ; 
    std::cout << "NCSG::Adopt" << std::endl ; 

    NSceneConfig* config = new NSceneConfig(config_); 
    NCSG* csg = Adopt( root, config, soIdx , lvIdx ); 
    LOG(LEVEL) << "]" ; 
    return csg ; 
}

/**
NCSG::Adopt
------------

NCSG instanciation via ctor taking nnode argument 

**/

NCSG* NCSG::Adopt(nnode* root, const NSceneConfig* config, unsigned soIdx, unsigned lvIdx )
{
    root->prepare();  // prepareTree or prepareList depending on type 

    LOG(LEVEL) 
        << " [ "
        << " soIdx " << soIdx  
        << " lvIdx " << lvIdx 
        ; 

    // XCSG test generation giving lots of these Adopts with zero idx
    // if( soIdx == 0 && lvIdx == 0) std::raise(SIGINT); 


    int treeidx = root->get_treeidx() ; 
    if( treeidx > -1 ) 
    {
        assert( unsigned(treeidx) == lvIdx ); 
    }

    root->set_treeidx(lvIdx) ;  // without this no nudging is done

    NCSG* tree = new NCSG(root);

    tree->setConfig(config);
    tree->setSOIdx(soIdx); 
    tree->setLVIdx(lvIdx); 
    tree->setIndex(lvIdx); 

    tree->postchange();  // collect global transforms and exports

    LOG(LEVEL) 
        << " ] "
        << " soIdx " << soIdx  
        << " lvIdx " << lvIdx 
        ; 

    return tree ; 
}


/**
NCSG::postchange
-----------------

Updates result buffers. Needs to be called following 
changes to the geometry such as centering by the 
setting of placement transforms or translations.

**/

void NCSG::postchange() 
{
    collect_global_transforms();  // also sets the gtransform_idx onto the tree
    export_();                    // node tree or list serialized into m_nodes buffer
    export_srcidx();              // identity indices into srcidx buffer  : formerly was not done by NCSG::Load only NCSG::Adopt

    if(m_config)
    {
        if(m_config->verbosity > 1) dump("NCSG::postchange");
        if(m_config->polygonize) polygonize();
    }

    assert( getGTransformBuffer() );

    bool pointskip = m_root->is_pointskip() ;  
    if(!pointskip) collect_surface_points();
}



/////////////////////////////////////////////////////////////////////////////////////////

/**
NCSG::NCSG(const char* treedir) 
---------------------------------

Private constructor used by NCSG::Load deserialization of a tree directory 

**/
NCSG::NCSG(const char* treedir) 
    :
    m_meta(new NPYMeta),
    m_treedir(treedir ? strdup(treedir) : NULL),
    m_index(0),
    m_surface_epsilon(SURFACE_EPSILON),
    m_verbosity(0),
    m_usedglobally(true),   // changed to true : June 2018, see notes/issues/subtree_instances_missing_transform.rst
    m_root(NULL),
    m_points(NULL),
    m_uncoincide(NULL),
    m_nudger(NULL),
    m_csgdata(new NCSGData),
    m_adopted(false), 
    m_boundary(NULL),
    m_config(NULL),
    m_gpuoffset(0,0,0),
    m_proxylv(-1),
    m_container(0),
    m_containerscale(2.f),
    m_containerautosize(-1),
    m_tris(NULL),
    m_soIdx(0),
    m_lvIdx(0),
    m_other(NULL),
    m_height(-1)
{
}



/**
NCSG::MakeNudger
----------------

static helper used from ctor. 
A static method is is appropriate as most member variables are 
not yet initialized when this is called.

**/

NNodeNudger* NCSG::MakeNudger(const char* msg, nnode* root, float surface_epsilon )   // static  
{
    int treeidx = root->get_treeidx(); 
    bool nudgeskip = root->is_nudgeskip() ; 

    LOG(LEVEL) 
        << " treeidx " << treeidx 
        << " nudgeskip " << nudgeskip 
         ; 

    NNodeNudger* nudger = nudgeskip ? nullptr : new NNodeNudger(root, surface_epsilon, root->verbosity);
    return nudger ; 
}






/**
NCSG::NCSG(nnode* root )
-----------------------------

Private constructor used by NCSG::Adopt booting from in memory nnode tree

* the root nnode cannot be const because of the nudger 
* TODO: nudger should know the lvIdx for identification of problem solids

**/

NCSG::NCSG(nnode* root ) 
    :
    m_meta(new NPYMeta),
    m_treedir(NULL),
    m_index(0),
    m_surface_epsilon(SURFACE_EPSILON),
    m_verbosity(root->verbosity),
    m_usedglobally(true),   // changed to true : June 2018, see notes/issues/subtree_instances_missing_transform.rst
    m_root(root),
    m_points(NULL),
    m_uncoincide(make_uncoincide()),
    m_nudger(MakeNudger("Adopt root ctor", root, SURFACE_EPSILON)),
    m_csgdata(new NCSGData),
    m_adopted(true), 
    m_boundary(NULL),
    m_config(NULL),
    m_gpuoffset(0,0,0),
    m_proxylv(-1),
    m_container(0),
    m_containerscale(2.f),
    m_containerautosize(-1),
    m_tris(NULL),
    m_soIdx(0),
    m_lvIdx(0),
    m_other(NULL),
    m_height(root->is_tree() ? root->maxdepth() : -1)
{
    init();
}


void NCSG::init()
{
    setBoundary( m_root->boundary );  // boundary spec

    unsigned num_serialization_nodes = m_root->num_serialization_nodes(); 
    LOG(LEVEL) << "[ init csgdata : num_serialization_nodes " << num_serialization_nodes << " height " << m_height << "(-1 for lists)"   ; 
    m_csgdata->init_node_buffers(num_serialization_nodes) ;  
    LOG(LEVEL) << "] init csgdata " ; 
}


void NCSG::savesrc(const char* idpath, const char* rela, const char* relb ) const 
{
    std::string treedir_ = BFile::FormPath( idpath, rela, relb ); 
    const char* treedir = treedir_.c_str(); 

    savesrc(treedir);  
}


void NCSG::savesrc(const char* treedir_ ) const 
{
    bool same_dir = m_treedir && strcmp( treedir_, m_treedir) == 0  ;
    if( same_dir ) LOG(fatal) << "saving back into the same dir as loaded from is not allowed " ; 
    assert( !same_dir) ; 
    assert( treedir_ ) ; 

    LOG(LEVEL) << "[ treedir_ " << treedir_ ; 

    LOG(LEVEL) << "m_csgdata" ; 
    m_csgdata->savesrc( treedir_ ) ;  

    LOG(LEVEL) << "[ m_meta" ; 
    m_meta->save( treedir_ ); 
    LOG(LEVEL) << "] m_meta" ; 

    LOG(LEVEL) << "] treedir_ " << treedir_ ; 
}

void NCSG::loadsrc()
{
    if(m_verbosity > 1) 
         LOG(info) 
             << " verbosity(>1) " << m_verbosity 
             << " index " << m_index 
             << " treedir " << m_treedir 
             ; 

    m_csgdata->loadsrc( m_treedir ) ; 
    m_meta->load( m_treedir ); 

    postload();
    LOG(debug) << "NCSG::load DONE " ; 
}


void NCSG::postload()
{
    m_soIdx = m_csgdata->getSrcSOIdx(); 
    m_lvIdx = m_csgdata->getSrcLVIdx(); 

    if( m_verbosity > 2) 
    LOG(error) 
        << " verbosity(>2) " << m_verbosity
        << " soIdx " << m_soIdx 
        << " lvIdx " << m_lvIdx 
        ; 

    m_proxylv         = m_meta->getValue<int>("proxylv", "-1");
    m_container       = m_meta->getValue<int>("container", "-1");
    m_containerscale  = m_meta->getValue<float>("containerscale", "2.");
    m_containerautosize = m_meta->getValue<int>("containerautosize", "0");

    if( m_proxylv > -1 )
    {
        LOG(LEVEL) 
            << " proxylv " << m_proxylv
            ;
    }

    std::string gpuoffset = m_meta->getValue<std::string>("gpuoffset", "0,0,0" );
    m_gpuoffset = gvec3(gpuoffset);  

    int verbosity = m_meta->getValue<int>("verbosity", "0") ;

    increaseVerbosity(verbosity);
}

bool NCSG::isProxy() const 
{
    return m_proxylv > -1 ; 
}
unsigned NCSG::getProxyLV() const 
{
    return m_proxylv ;  
}
bool NCSG::isContainerAutoSize() const 
{
    return m_containerautosize == 1 ; 
}
bool NCSG::isContainer() const 
{
    return m_container > 0  ; 
}
float NCSG::getContainerScale() const 
{
    return m_containerscale  ; 
}




/** 
NCSG::import : from complete binary tree buffers into nnode tree
--------------------------------------------------------------------

1. prepareForImport : tripletize the m_srctransforms into m_transforms
2. import_tree : collects m_gtransforms by multiplication down the tree

**/

void NCSG::import()
{
    if(m_verbosity > 1)
        LOG(info) 
                  << " verbosity(>1) " << m_verbosity
                  << " treedir " << m_treedir
                  << " smry " << smry()
                  ; 

    if(m_verbosity > 1)
    {
        LOG(info) 
                  << " verbosity(>0) " << m_verbosity 
                  << " importing buffer into CSG node tree "
                  << " num_nodes " << getNumNodes()
                  ;
    }

    m_csgdata->prepareForImport() ;  // from m_srctransforms to m_transforms, and get m_gtransforms ready to collect

    // need type of first node to distinguish tree from list   
    OpticksCSG_t root_type  = (OpticksCSG_t)m_csgdata->getTypeCode(0);      
    LOG(LEVEL) << " root_type " << CSG::Name(root_type) ; 

    if(CSG::IsTree(root_type))
    {
        import_tree(); 
    }
    else if(CSG::IsList(root_type))
    {
        import_list(); 
    }
    else
    {
        assert(0) ; // unexpected root_type 
    }


    m_root->set_treedir(m_treedir) ; 
    m_root->set_treeidx(getTreeNameIdx()) ;  //

    postimport();  // create nudger
    checkroot(); 

    if(m_verbosity > 5) check();  // recursive transform dumping 
    if(m_verbosity > 1) LOG(info) << "]" ; 
}

void NCSG::import_tree()
{
    m_root = import_tree_r(0, NULL) ;  

    m_root->check_tree( FEATURE_PARENT_LINKS );  
    m_root->update_gtransforms(); 
    m_root->check_tree( FEATURE_PARENT_LINKS | FEATURE_GTRANSFORMS );  
}


void NCSG::import_list()
{
    m_root = import_list_node(0); 

    std::vector<nnode*>& subs = m_root->subs ; 

    assert( subs.size() == 0 ); 

    unsigned num = m_root->subNum(); 

    LOG(LEVEL)
        << " root.type " << CSG::Name(m_root->type) 
        << " num " << num 
        ; 
    assert( num > 0 ); 

    for(unsigned isub=0 ; isub < num ; isub++)
    {
        nnode* sub = import_list_node( 1+isub );
        subs.push_back(sub);  
    }
}

nnode* NCSG::import_list_node( unsigned idx ) 
{
    OpticksCSG_t type = (OpticksCSG_t)m_csgdata->getTypeCode(idx);      
    int transform_idx = m_csgdata->getTransformIndex(idx) ;
    bool complement = m_csgdata->isComplement(idx) ; 
    bool is_list = CSG::IsList(type) ; 
    bool is_list_expect = idx == 0 ? true : false ; 

    LOG(LEVEL) 
        << " idx " << idx
        << " type " << CSG::Name(type)
        << " transform_idx " << transform_idx
        << " complement " << complement 
        ;
    assert( is_list == is_list_expect ); 
    assert( complement == false ); 

    nnode* node = import_primitive( idx, type ); 

    node->transform = m_csgdata->import_transform_triple( transform_idx ) ;  
    // from m_transforms, expecting (-1,3,4,4)

    return node ; 
}



void NCSG::postimport()
{
    m_nudger = MakeNudger("postimport", m_root, SURFACE_EPSILON) ; 
}

/**
NCSG::import_tree_r
--------------------

Importing : constructs the in memory nnode tree from 
the src buffers loaded by loadsrc ( which were written by analytic/csg.py ) 
Prior to importing the NCSGData::prepareForImport must 
be called to triplet-ize the srctransforms making the 
transforms buffer.

Formerly:

    On import the gtransforms (**for primitives only**) are constructed 
    by multiplication down the tree, and uniquely collected into m_gtransforms 
    with the 1-based gtransforms_idx being set on the node.

But now thats done from NCSG::import 

**/

nnode* NCSG::import_tree_r(unsigned idx, nnode* parent)
{
    if(idx >= getNumNodes()) return NULL ; 
    
    // from m_srcnodes     
    OpticksCSG_t typecode = (OpticksCSG_t)m_csgdata->getTypeCode(idx);      
    int transform_idx = m_csgdata->getTransformIndex(idx) ; 
    bool complement = m_csgdata->isComplement(idx) ; 

    LOG(debug) 
        << " idx " << idx
        << " transform_idx " << transform_idx
        << " complement " << complement 
        ;

    nnode* node = NULL ;   
 
    if(typecode == CSG_UNION || typecode == CSG_INTERSECTION || typecode == CSG_DIFFERENCE)
    {
        node = import_tree_operator( idx, typecode ) ; 

        node->parent = parent ; 
        node->idx = idx ; 
        node->complement = complement ; 

        node->transform = m_csgdata->import_transform_triple( transform_idx ) ;  // from m_transforms, expecting (-1,3,4,4)

        node->left = import_tree_r(idx*2+1, node );  
        node->right = import_tree_r(idx*2+2, node );

        node->left->other = node->right ;   // used by NOpenMesh 
        node->right->other = node->left ; 

        // recursive calls after "visit" as full ancestry needed for transform collection once reach primitives
    }
    else 
    {
        node = import_primitive( idx, typecode ); 

        node->parent = parent ;                // <-- parent hookup needed prior to gtransform collection 
        node->idx = idx ; 
        node->complement = complement ; 
        node->transform = m_csgdata->import_transform_triple( transform_idx ) ;  // from m_transforms, expecting (-1,3,4,4)
    }
    assert(node); 

    BMeta* nodemeta = m_meta->getMeta(idx);

    if(nodemeta) node->meta = nodemeta ; 

    // Avoiding duplication between the operator and primitive branches 
    // in the above is not sufficient reason to put things here, so very late.
    return node ; 
} 



nnode* NCSG::import_tree_operator( unsigned idx, OpticksCSG_t typecode )
{
    if(m_verbosity > 2)
    {
    LOG(info) << "NCSG::import_operator " 
              << " idx " << idx 
              << " typecode " << typecode 
              << " csgname " << CSG::Name(typecode) 
              ;
    }

    nnode* node = NULL ;   
    switch(typecode)
    {
       case CSG_UNION:        node = nunion::make_union(NULL, NULL ) ; break ; 
       case CSG_INTERSECTION: node = nintersection::make_intersection(NULL, NULL ) ; break ; 
       case CSG_DIFFERENCE:   node = ndifference::make_difference(NULL, NULL )   ; break ; 
       default:               node = NULL                                 ; break ; 
    }
    assert(node);
    return node ; 
}

nnode* NCSG::import_primitive( unsigned idx, OpticksCSG_t typecode )
{
    // from srcnodes buffer
    nquad p0 = m_csgdata->getQuad(idx, 0);
    nquad p1 = m_csgdata->getQuad(idx, 1);
    nquad p2 = m_csgdata->getQuad(idx, 2);
    nquad p3 = m_csgdata->getQuad(idx, 3);

    if(m_verbosity > 2)
    {
    LOG(info) << "NCSG::import_primitive  " 
              << " idx " << idx 
              << " typecode " << typecode 
              << " csgname " << CSG::Name(typecode) 
              ;
    }

    nnode* node = NULL ;   
    switch(typecode)
    {
       case CSG_SPHERE:         node = nsphere::Create(p0)      ; break ; 
       case CSG_ZSPHERE:        node = make_zsphere(p0,p1,p2)   ; break ; 
       case CSG_BOX:            node = make_box(p0)             ; break ; 
       case CSG_BOX3:           node = make_box3(p0)            ; break ; 
       case CSG_SLAB:           node = make_slab(p0, p1)        ; break ; 
       case CSG_PLANE:          node = make_plane(p0)           ; break ; 
       case CSG_CYLINDER:       node = make_cylinder(p0, p1)    ; break ; 
       case CSG_DISC:           node = make_disc(p0, p1)        ; break ; 
       case CSG_CONE:           node = make_cone(p0)            ; break ; 
       case CSG_TORUS:          node = make_torus(p0)           ; break ; 
       case CSG_CUBIC:          node = make_cubic(p0,p1)        ; break ; 
       case CSG_HYPERBOLOID:    node = make_hyperboloid(p0)     ; break ; 
       case CSG_TRAPEZOID:  
       case CSG_SEGMENT:  
       case CSG_CONVEXPOLYHEDRON:  
                                node = make_convexpolyhedron(p0,p1,p2,p3)   ; break ; 
       case CSG_CONTIGUOUS:     
       case CSG_DISCONTIGUOUS:  
                                node = nmultiunion::Create(typecode, p0)  ; break ; 

       case CSG_ELLIPSOID: assert(0 && "ellipsoid should be zsphere at this level" )   ; break ; 
       default:           node = NULL ; break ; 
    }       


    if(node == NULL) 
    {
        LOG(fatal) 
            << " TYPECODE NOT IMPLEMENTED " 
            << " idx " << idx 
            << " typecode " << typecode
            << " csgname " << CSG::Name(typecode)
            ;
    } 

    assert(node); 

    if(CSG::HasPlanes(typecode)) 
    {
        import_srcplanes( node );
        import_srcvertsfaces( node );
    }

    if(m_verbosity > 3)
    {
        LOG(info) 
            << " idx " << idx 
            << " typecode " << typecode 
            << " csgname " << CSG::Name(typecode) 
            << " DONE " 
            ;
    } 
    return node ; 
}

/**
NCSG::import_srcvertsfaces
----------------------------

Import from SrcVerts and SrcFaces buffers into the nconvexpolyhedron instance

**/


void NCSG::import_srcvertsfaces(nnode* node)
{
    assert( node->has_planes() );
    
    NPY<float>* srcverts = m_csgdata->getSrcVertsBuffer() ; 
    NPY<int>*   srcfaces = m_csgdata->getSrcFacesBuffer() ; 

    if(!srcverts || !srcfaces) 
    {
        LOG(debug) << "NCSG::import_srcvertsfaces no srcverts  srcfaces " ; 
        return ; 
    }

    nconvexpolyhedron* cpol = dynamic_cast<nconvexpolyhedron*>(node);
    assert(cpol);

    std::vector<glm::vec3> _verts ;  
    std::vector<glm::ivec4> _faces ;  

    srcverts->copyTo(_verts);
    srcfaces->copyTo(_faces);

    cpol->set_srcvertsfaces(_verts, _faces);     
}

void NCSG::import_srcplanes(nnode* node)
{
    assert( node->has_planes() );

    nconvexpolyhedron* cpol = dynamic_cast<nconvexpolyhedron*>(node);
    assert(cpol);

    unsigned iplane = node->planeIdx() ;   // 1-based idx ?
    unsigned num_plane = node->planeNum() ;
    unsigned idx = iplane - 1 ;     

    if(m_verbosity > 3)
    {
    LOG(info) << "NCSG::import_planes"
              << " iplane " << iplane
              << " num_plane " << num_plane
              ;
    }

    assert( node->planes.size() == 0u );

    std::vector<glm::vec4> _planes ;  
    m_csgdata->getSrcPlanes(_planes, idx, num_plane ); 
    assert( _planes.size() == num_plane ) ; 

    cpol->set_planes(_planes);     
    assert( cpol->planes.size() == num_plane );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void NCSG::checkroot() const 
{
    assert( m_root ); 
    //if( m_root->is_primitive() )  m_root->dump(); 
}


void NCSG::check() const 
{
    check_r( m_root );
    m_csgdata->dump_gtransforms(); 
}

void NCSG::check_r(const nnode* node) const 
{
    check_node(node);
    if(node->left && node->right)
    {
        check_r(node->left);
        check_r(node->right);
    }
}
void NCSG::check_node(const nnode* node ) const 
{
    if(m_verbosity > 2)
    {
        if(node->gtransform)
        {
            std::cout << "NCSG::check_r"
                      << " gtransform_idx " << node->gtransform_idx
                      << std::endl 
                      ;
        }
        if(node->transform)
        {
            std::cout << "NCSG::check_r"
                      << " transform " << *node->transform
                      << std::endl 
                      ;
        }
    }
}







///////////////////////////////////////////////////////////////////////////////////////////////////
/**
NCSG::collect_global_transforms
-------------------------------

The setting of global transforms was formally done both on
import and export : which was confusing.  To avoid the
confusion split it off into these focussed methods.

Effects:

1. collects global transforms into m_gtransforms buffer
2. sets node->gtransform_idx for primitives in the tree

   * formerly it set node->gtransform : but thats too late for eg NNodeNudger
     so now it just checks it gets the same transform

   

Related:

opticks/notes/issues/subtree_instances_missing_transform.rst

On GPU only gtransform_idx on primitives are used
(there being no multiplying up the tree). 
Any gtransform_idx on operator nodes are ignored.

see notes/issues/OKX4Test_partBuffer_difference.rst

using 0 (meaning None) for identity 

**/

void NCSG::collect_global_transforms() 
{
    bool locked(false) ;  // with locked=true m_csgdata asserts if called more than once
    m_csgdata->prepareForGTransforms(locked);

    if(m_root->is_tree())
    {
        collect_global_transforms_r( m_root ) ; 
    }
    else if(m_root->is_list())
    {
        collect_global_transforms_list() ; 
    }
}

void NCSG::collect_global_transforms_list() 
{
    check_subs(); 
    collect_global_transforms_node(m_root);

    unsigned sub_num = m_root->subNum(); 
    for(unsigned isub=0 ; isub < sub_num ; isub++)
    {
        nnode* sub = m_root->subs[isub];    
        // sub cannot be const, as the export writes things like indices into the node

        collect_global_transforms_node(sub);
    }
}

void NCSG::collect_global_transforms_r(nnode* node) 
{
    collect_global_transforms_node(node);

    if(node->left && node->right)
    {
        collect_global_transforms_r(node->left);
        collect_global_transforms_r(node->right);
    }
}
void NCSG::collect_global_transforms_node(nnode* node)
{
    if(node->is_primitive())
    {
        const nmat4triple* gtransform = node->global_transform();   
        assert( gtransform ); 
        assert( node->gtransform ) ; 
        assert( node->gtransform->is_equal_to( gtransform )); 

        unsigned gtransform_idx = addUniqueTransform(gtransform) ;   // collect into m_gtransforms

        node->gtransform_idx = gtransform_idx ; // 1-based, 0 for None (which should not happen now)
    }
} 




unsigned NCSG::addUniqueTransform( const nmat4triple* gtransform_ )
{
    bool no_offset = m_gpuoffset.x == 0.f && m_gpuoffset.y == 0.f && m_gpuoffset.z == 0.f ;

    bool reverse = true ; // <-- apply transfrom at root of transform hierarchy (rather than leaf)

    bool match = true ; 

    const nmat4triple* gtransform = no_offset ? gtransform_ : gtransform_->make_translated(m_gpuoffset, reverse, "NCSG::addUniqueTransform", match ) ;

    if(!match)
    {
        LOG(error) << "matrix inversion precision issue ?" ; 
    }

    /*
    std::cout << "NCSG::addUniqueTransform"
              << " orig " << *gtransform_
              << " tlated " << *gtransform
              << " gpuoffset " << m_gpuoffset 
              << std::endl 
              ;
    */
    return m_csgdata->addUniqueTransform( gtransform );   // add to m_gtransforms
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/**
NCSG::export_
-----------------

Writing node tree into transport buffers using 
complete binary tree indexing. 

1. prepareForExport : init NODES, PLANES 


NCSG::export_node
~~~~~~~~~~~~~~~~~~

export_gtransform
   collects gtransform into the tran buffer and sets gtransform_idx 
   on the node tree 

export_planes
   collects convex polyhedron planes into plan buffer

NB the part writing into part buffer has to be after 
these as idxs get into that buffer


**/

void NCSG::export_()
{
    m_csgdata->prepareForExport() ;  //  create node buffer 

    NPY<float>* _nodes = m_csgdata->getNodeBuffer() ; 
    assert(_nodes);

    export_idx();  

    if( m_root->is_tree() )
    {
        export_tree_();
    }
    else if( m_root->is_list() )
    {
        export_list_();  
    }
    else if( m_root->is_leaf() )
    {
        export_leaf_();  
    }
    else
    {
        assert(0) ;  // unexpected m_root type  
    }
}



/**
NCSG::export_idx
-------------------

Set identity indices into the idx buffer

**/

void NCSG::export_idx()   // only tree level
{
    bool src = false ; 
    m_csgdata->setIdx( m_index, m_soIdx, m_lvIdx, m_height, src ); 
}

void NCSG::export_srcidx()   // only tree level
{
    bool src = true  ; 
    m_csgdata->setIdx( m_index, m_soIdx, m_lvIdx, m_height, src ); 
}


void NCSG::export_tree_()
{
    LOG(LEVEL) 
        << " exporting CSG node tree into nodes buffer "
        << " num_nodes " << getNumNodes()
        ;


    m_root->check_tree( FEATURE_PARENT_LINKS | FEATURE_GTRANSFORMS ) ; 

    export_tree_r(m_root, 0);
}

/**
NCSG::export_tree_r
----------------------

Somewhat unually this is using 0-based level indices for complete binary tree indexing::

             0
       1            2 
     3   4      5      6 
    7 8 9 10  11 12  13 14

1-based indexing is more intuitive, expecially when express in binary::

                     1
            10                    11 
      100       101         110         111
  1000 1001  1010 1011   1100 1101   1110 1111 

**/

void NCSG::export_tree_r(nnode* node, unsigned idx)
{
    export_node( node, idx) ; 

    if(node->left && node->right)
    {
        export_tree_r(node->left,  2*idx + 1);  // 0-based complete binary tree indexing 
        export_tree_r(node->right, 2*idx + 2);
    }  
}

void NCSG::check_subs() const 
{
    bool is_list = CSG::IsList(m_root->type); 
    assert( is_list ); 

    unsigned sub_num_0 = m_root->subs.size(); 
    unsigned sub_num_1 = m_root->subNum(); 
    assert( sub_num_0 == sub_num_1 ); 
}

void NCSG::export_leaf_()
{
    export_node( m_root, 0) ; 
}

void NCSG::export_list_()
{
    unsigned idx = 0 ; 
    export_node( m_root,  idx) ; 

    check_subs(); 
    unsigned sub_num = m_root->subNum(); 

    for(unsigned isub=0 ; isub < sub_num ; isub++)
    {
        idx = 1 + isub ; 
        nnode* sub = m_root->subs[isub];    
        // sub cannot be const, as the export writes things like indices into the node
        export_node( sub,  idx) ; 
    }
}


/**
NCSG::export_node
--------------------

NB this now writes nodes to the transport nodes buffer, which 
is separate from the srcnodes buffer 

Note the critical importance of the NNode::part method
which converts the node into npart instance 
of which 4 quads get written to the transport nodes buffer.   

In the case of adopted nnode the src buffers are also populated, 
to enable subsequent loading to be just like loading from python written trees 

**/

void NCSG::export_node(nnode* node, unsigned idx)
{
    assert(idx < getNumNodes() ); 
    LOG(verbose) 
        << " idx " << idx 
        << node->desc()
        ;

    if(m_adopted)   
    {
        export_srcnode( node, idx) ; 
    }

    NPY<float>* _planes = m_csgdata->getPlaneBuffer() ; 
    export_planes(node, _planes);
  
    npart pt = node->part();  // node->type node->gtransform_idx node->complement written into pt 

    bool root_is_list = m_root->is_list(); 
    if(root_is_list)
    {
        LOG(LEVEL) 
           << " idx " << idx 
           << " root_is_list exporting node  pt.q0.u.x " << pt.q0.u.x
           ; 
    }

    pt.check_bb_is_zero(node->type); 

    NPY<float>* _nodes = m_csgdata->getNodeBuffer(); 

    _nodes->setPart( pt, idx);  // writes 4 quads to buffer
}


/**
NCSG::export_planes
----------------------

Adds the planes from the node into the _planes buffer argument
and documents the 1-based planeIdx and number of planes into the 
node param.  

NCSG::export_planes is invoked from NCSG::export_node which 
is called from the recursive NCSG::export_r 

NCSG::export_node gets the _planes buffer with m_csgdata->getPlaneBuffer

With CSGFoundry a single global planes (and transforms) buffer is 
used unlike with the old pre-7 model where separate buffers (and indexing)
for each GMergedMesh are used.  

**/

void NCSG::export_planes(nnode* node, NPY<float>* _planes)
{
    assert( _planes && "export_planes requires _planes buffer "); 

    if(!node->has_planes()) return ;

    nconvexpolyhedron* cpol = dynamic_cast<nconvexpolyhedron*>(node);
    assert(cpol);

    assert(node->planes.size() > 0);  

    unsigned planeNum = node->planes.size() ;
    unsigned planeIdx0 = _planes->getNumItems();   // 0-based idx
    unsigned planeIdx1 = planeIdx0 + 1 ;           // 1-based idx

    if( node->verbosity > 2 )
    LOG(error) 
         << " export_planes "
         << " node.verbosity " << node->verbosity
         << " planeIdx1 " << planeIdx1 
         << " planeNum " << planeNum
         << " _planes.shape " << _planes->getShapeString() 
         ;

    node->setPlaneIdx( planeIdx1 );  
    node->setPlaneNum( planeNum );   // directly sets into node param

    assert( planeNum > 3); 

    for(unsigned i=0 ; i < planeNum ; i++)
    {
        const glm::vec4& pln = node->planes[i] ; 
        _planes->add(pln); 
    } 
    unsigned planeIdxN = _planes->getNumItems(); 
    assert( planeIdxN == planeIdx0 + planeNum );
}


/**
NCSG::export_srcnode
----------------------

In the adopted from nnode case (as opposed to loaded from python written buffers) 
there are no src buffers so remedy that here : 
allowing loading to proceed just like from python
 
**/

void NCSG::export_srcnode(nnode* node, unsigned idx)
{
    LOG(LEVEL) << "[" ; 

    assert( m_adopted ) ; 

    NPY<float>* _srcnodes = m_csgdata->getSrcNodeBuffer(); 
    assert( _srcnodes ) ; 

    NPY<float>* _srctransforms = m_csgdata->getSrcTransformBuffer(); 
    export_itransform( node, _srctransforms );   // sets node->itransform_idx 

    NPY<float>* _srcplanes = m_csgdata->getSrcPlaneBuffer() ; 
    export_planes(node, _srcplanes);  // directly sets planeIdx, planeNum into nnode param
 
    npart pt = node->srcpart();
    _srcnodes->setPart( pt, idx );   // writes 4 quads to buffer 

    LOG(LEVEL) << "]" ; 
}


/**
NCSG::export_itransform
------------------------

Note this writes just the transforms, not the full triplet as
that is what analytic/csg.py does

**/

void NCSG::export_itransform( nnode* node, NPY<float>* _dest )
{
    assert( _dest ) ; 

    const nmat4triple* trs = node->transform ; 
    const glm::mat4 identity(1.f) ;  
    const glm::mat4& t = trs ? trs->t : identity ; 

    unsigned itransform1 = _dest->getNumItems() + 1 ;  // 1-based transform idx
 
    _dest->add(t) ;   // <-- nothing fancy straight collection just like analytic/csg.py:serialize 

    node->itransform_idx = itransform1  ;    
}


//////////////////////////////////////////////////////////////////////////////////////////////


void NCSG::dump(const char* msg) const 
{
    LOG(info) << msg  ; 
    std::cout << brief() << std::endl ; 

    if(!m_root) return ;

    unsigned nsp = getNumSurfacePoints();
    if( nsp > 0)
    {
        nbbox bbsp = bbox_surface_points();
        std::cout << " bbsp " << bbsp.desc() << std::endl ; 
    }

    m_root->dump("NCSG::dump");   

    BMeta* _meta = m_meta->getMeta(-1) ;

    if(_meta) _meta->dump(); 

}

void NCSG::dump_surface_points(const char* msg, unsigned dmax) const 
{
    if(!m_points) return ;
    m_points->dump(msg, dmax );
}


std::string NCSG::brief() const 
{
    std::stringstream ss ; 
    ss << " NCSG " 
       << " ix " << std::setw(4) << m_index
       << " surfpoints " << std::setw(4) << getNumSurfacePoints() 
       << " so " << std::setw(40) << std::left << get_soname()
       << " lv " << std::setw(40) << std::left << get_lvname()
       << " . " 
       ;

    return ss.str();  
}

std::string NCSG::desc()
{
    std::stringstream ss ; 
    ss << "NCSG " 
       << " index " << m_index
       << " treedir " << ( m_treedir ? m_treedir : "NULL" ) 
       << " boundary " << ( m_boundary ? m_boundary : "NULL" ) 
       << m_csgdata->desc()
       ;
    return ss.str();  
}


NTrianglesNPY* NCSG::polygonize_bbox_placeholder()
{
    nbbox bb = bbox(); 
    NTrianglesNPY* tris = NTrianglesNPY::box(bb) ;
    return tris ; 
}


NTrianglesNPY* NCSG::polygonize()
{
    assert( m_config ) ; 
    //m_config->dump("NCSG::polygonize"); 

    if(m_tris == NULL)
    {
#ifdef WITH_OPENMESH
        NPolygonizer pg(this);
        m_tris = pg.polygonize();
#else
        LOG(fatal) << "NCSG::polygonize requires compilation with the optional OpenMesh : using bbox triangles placeholder " ;  
        m_tris = polygonize_bbox_placeholder() ; 
#endif
    }
    return m_tris ; 
}

NTrianglesNPY* NCSG::getTris() const 
{
    return m_tris ; 
}
unsigned NCSG::getNumTriangles() const 
{
    return m_tris ? m_tris->getNumTriangles() : 0 ; 
}

/**
NCSG::collect_surface_points
------------------------------

For some solids this seems to hang.

**/

glm::uvec4 NCSG::collect_surface_points() 
{
    if(!m_points) 
    {
        m_points = new NNodePoints(m_root, m_config );
    } 
    glm::uvec4 tots = m_points->collect_surface_points();
    return tots ; 
}





nbbox NCSG::bbox() const 
{
    assert(m_root);
    return m_root->bbox();
}

glm::vec4 NCSG::bbox_center_extent() const 
{
    assert(m_root);
    return m_root->bbox_center_extent() ; 
}


bool NCSG::has_placement_translation() const 
{
    assert(m_root);
    return m_root->has_placement_translation() ; 
}
glm::vec3 NCSG::get_placement_translation() const 
{
    assert(m_root);
    return m_root->get_placement_translation() ; 
}

bool NCSG::has_placement_transform() const 
{
    assert(m_root);
    return m_root->has_placement_transform() ; 
}
glm::mat4 NCSG::get_placement_transform() const 
{
    assert(m_root);
    return m_root->get_placement_transform() ; 
}

bool NCSG::has_root_transform() const 
{
    assert(m_root);
    return m_root->has_root_transform() ; 
}
glm::mat4 NCSG::get_root_transform() const 
{
    assert(m_root);
    return m_root->get_root_transform() ; 
}











/**
NCSG::set_translation
------------------------

invoked from GMesh::applyCentering for GGeoTest 

**/

void NCSG::set_translation(float x, float y, float z) 
{
    assert(m_root);
    m_root->set_translation(x, y, z) ; 

    postchange();  // update buffers following geometry change
}

void NCSG::set_centering()  
{
    assert(m_root);
    m_root->set_centering() ; 

    postchange();  // update buffers following geometry change
}


nbbox NCSG::bbox_surface_points() const 
{
    assert(m_points); 
    return m_points->bbox_surface_points();
}


const std::vector<glm::vec3>& NCSG::getSurfacePoints() const 
{
    assert(m_points); 
    return m_points->getCompositePoints();
}
unsigned NCSG::getNumSurfacePoints() const 
{
    return m_points ? m_points->getNumCompositePoints() : 0 ;
}
float NCSG::getSurfaceEpsilon() const 
{
    return m_points ? m_points->getEpsilon() : -1.f ;
}


/**
NCSG::resizeToFit
------------------

Changes extent of analytic geometry to be that of the container argument
with scale and delta applied.
Only implemented for CSG_BOX, CSG_BOX3 and CSG_SPHERE.

**/

void NCSG::resizeToFit( const nbbox& fit_bb, float scale, float delta ) const 
{
    LOG(debug) << "[" ; 

    bool empty_bb = fit_bb.is_empty(); 

    if(empty_bb)
        LOG(fatal) << " EMPTY fit_bb " << fit_bb.desc()
        ;

    assert( !empty_bb );     


    nnode* root = getRoot();

    nbbox root_bb = root->bbox();
 
    nnode::ResizeToFit(root, fit_bb, scale, delta );         

    LOG(LEVEL) << "]"
               << " scale " << scale 
               << " delta " << delta
               << " root_bb " << root_bb.desc()
               << " fit_bb " << fit_bb.desc()
               ;
}



// huh aint these persisted ?? in csgdata 
void NCSG::setSOIdx(unsigned soIdx) { m_soIdx = soIdx ; }
void NCSG::setLVIdx(unsigned lvIdx) { m_lvIdx = lvIdx ; }
unsigned NCSG::getSOIdx() const { return m_soIdx ; } 
unsigned NCSG::getLVIdx() const { return m_lvIdx ; }



int NCSG::get_num_coincidence() const 
{
   return m_nudger ? m_nudger->get_num_coincidence() : -1 ; 
}
std::string NCSG::desc_coincidence() const 
{
   return m_nudger ? m_nudger->desc_coincidence() : "" ; 
}

void NCSG::postimport_autoscan()
{

   float mmstep = 0.1f ; 
    NScan scan(*m_root, m_verbosity);
    unsigned nzero = scan.autoscan(mmstep);
    const std::string& msg = scan.get_message();

    bool even_crossings = nzero % 2 == 0 ; 

    if( !msg.empty() )
    {
        LOG(warning) << "NCSG::postimport_autoscan"
                     << " autoscan message " << msg 
                     ;
    }


    if( !even_crossings )
    {
        LOG(warning) << "NCSG::postimport_autoscan"
                     << " autoscan odd crossings "
                     << " nzero " << nzero 
                     ;
    }

    std::cout 
         << "NCSG::postimport_autoscan"
         << " nzero " << std::setw(4) << nzero 
         << " NScanTest " << std::left << std::setw(40) << getTreeDir()  << std::right
         << " soname " << std::setw(40) << get_soname()  
         << " tag " << std::setw(10) << m_root->tag()
         << " nprim " << std::setw(4) << m_root->get_num_prim()
         << " typ " << std::setw(20)  << m_root->get_type_mask_string()
         << " msg " << scan.get_message()
         << std::endl 
         ;
}


BMeta* NCSG::LoadMetadata( const char* treedir, int item )
{
    return NPYMeta::LoadMetadata(treedir, item); 
} 

NCSGData* NCSG::getCSGData() const 
{
   return m_csgdata ; 
}

NPYList* NCSG::getNPYList() const
{
   return m_csgdata->getNPYList() ;
}

NNodeUncoincide* NCSG::make_uncoincide() const 
{
    return NULL ;  
    //return new NNodeUncoincide(m_root, m_surface_epsilon, m_root->verbosity);
}
NNodeNudger* NCSG::get_nudger() const   // seems unused, TODO: remove
{
    return m_nudger ; 
}


std::string NCSG::TestVolumeName(const char* shapename, const char* suffix, int idx) // static
{
    std::stringstream ss ; 
    ss << shapename << "_" << suffix ; 
    if(idx > -1) ss << idx ; 
    ss << "_" ; 
    return ss.str();
}

std::string NCSG::getTestPVName() const 
{
    OpticksCSG_t type = getRootType() ;
    const char* shapename = CSG::Name(type); 
    unsigned idx = getIndex();
    return TestVolumeName( shapename, "pv", idx);     
}
std::string NCSG::getTestLVName() const 
{
    OpticksCSG_t type = getRootType() ;
    const char* shapename = CSG::Name(type); 
    unsigned idx = getIndex();
    return TestVolumeName( shapename, "lv", idx);     
}


// m_meta is NPYMeta instance, all the below get_* set_* are accessing default item=-1 
// corresponding to the tree, rather than individual nodes with index 0,1,... etc

NPYMeta* NCSG::getMeta() const { return m_meta ;  }
BMeta*   NCSG::getMeta(int idx) const { return m_meta->getMeta(idx); }

void NCSG::set_lvname(const char* name) { m_meta->setValue<std::string>("lvname", name) ; }
void NCSG::set_soname(const char* name) { m_meta->setValue<std::string>("soname", name) ; }
void NCSG::set_balanced(bool balanced ){  m_meta->setValue<int>("balanced", balanced ? 1 : 0 ); }
void NCSG::set_altindex(int altindex ){   m_meta->setValue<int>("altindex", altindex ); }   // used from GMeshLib 
void NCSG::set_emit(int emit){                      m_meta->setValue<int>("emit", emit) ; }   // used by --testauto
void NCSG::set_emitconfig(const char* emitconfig){  m_meta->setValue<std::string>("emitconfig", emitconfig ); }

std::string NCSG::get_lvname() const {        return m_meta->getValue<std::string>("lvname","-") ; }
std::string NCSG::get_soname() const {        return m_meta->getValue<std::string>("soname","-") ; }
bool        NCSG::is_balanced() const  {      return m_meta->getValue<int>("balanced","0") == 1 ; }
int         NCSG::get_altindex() const  {     return m_meta->getValue<int>("altindex","-1") ; }  // used from GMeshLib 
int         NCSG::get_emit() const {          return m_meta->getValue<int>("emit","0") ;  }
bool        NCSG::is_emitter() const {        int emit = get_emit() ; return emit == 1 || emit == -1 ; }
const char* NCSG::get_emitconfig() const { 
    std::string ec = m_meta->getValue<std::string>("emitconfig","") ;
    return ec.empty() ? NULL : strdup(ec.c_str()) ; 
}

int         NCSG::get_treeindex() const {     return m_meta->getValue<int>("treeindex","-1") ; }
int         NCSG::get_depth() const {         return m_meta->getValue<int>("depth","-1") ; }
int         NCSG::get_nchild() const {        return m_meta->getValue<int>("nchild","-1") ; }
bool        NCSG::is_skip() const {           return m_meta->getValue<int>("skip","0") == 1 ; }
bool        NCSG::is_uncoincide() const {     return m_meta->getValue<int>("uncoincide","1") == 1 ; }



unsigned NCSG::getHeight() const 
{ 
    return m_height ; 
    //return m_csgdata->getHeight(); 
}

// from m_csgdata
unsigned NCSG::getNumNodes() const { return m_csgdata->getNumNodes() ; } 

NPY<float>*    NCSG::getNodeBuffer() const {       return m_csgdata->getNodeBuffer() ; }
NPY<unsigned>* NCSG::getIdxBuffer() const {        return m_csgdata->getIdxBuffer() ; } 
NPY<float>*    NCSG::getTransformBuffer() const {  return m_csgdata->getTransformBuffer() ; } 
NPY<float>*    NCSG::getGTransformBuffer() const { return m_csgdata->getGTransformBuffer() ; } 
NPY<float>*    NCSG::getPlaneBuffer() const {      return m_csgdata->getPlaneBuffer() ; } 


// ordinary members

const char*         NCSG::getTreeDir() const {  return m_treedir ; }
const char*         NCSG::getTreeName() const { std::string name = BFile::Name(m_treedir ? m_treedir : "-1") ; return strdup(name.c_str()); }
int                 NCSG::getTreeNameIdx() const { const char* name = getTreeName(); return BStr::atoi(name, -1); } 



/**
NCSG::setIsUsedGlobally
------------------------

This formerly was used to indicate if a solid was to be used in the 
non-instanced "global" GMergedMesh requiring slightly different 
handling of transforms.   Nowadays I suspect that handling 
has been standardized and all solids can now be regarded as 
being "usedglobally".

**/

void NCSG::setIsUsedGlobally(bool usedglobally ){  m_usedglobally = usedglobally ;  }
void NCSG::setVerbosity(int verbosity) {           m_verbosity = verbosity ;  }
void NCSG::setBoundary(const char* boundary){      m_boundary = boundary ? strdup(boundary) : NULL ;  }
void NCSG::setConfig(const NSceneConfig* config) { m_config = config ;  }

bool                NCSG::isUsedGlobally() const { return m_usedglobally ;  }
int                 NCSG::getVerbosity() const { return m_verbosity ; }
const char*         NCSG::getBoundary() const { return m_boundary ; }
const NSceneConfig* NCSG::getConfig() const { return m_config ;  }

// from m_root 

nnode*       NCSG::getRoot() const {     return m_root ;  }
OpticksCSG_t NCSG::getRootType() const { assert(m_root); return m_root->type ; } 
const char*  NCSG::getRootCSGName() const { return m_root ? m_root->csgname() : NULL ; }
bool         NCSG::isBox() const {       assert( m_root ); return m_root->is_box() ;  }
bool         NCSG::isBox3() const  {     assert( m_root ); return m_root->is_box3() ;  }

std::string NCSG::get_type_mask_string() const { assert(m_root); return m_root->get_type_mask_string() ; }
unsigned    NCSG::get_type_mask() const  {          assert(m_root); return m_root->get_type_mask() ; }
unsigned    NCSG::get_oper_mask() const {           assert(m_root); return m_root->get_oper_mask() ; }
unsigned    NCSG::get_prim_mask() const {           assert(m_root); return m_root->get_prim_mask() ; } 



void NCSG::setOther(NCSG* other) { m_other = other ;   }
NCSG* NCSG::getOther() const   {   return m_other ;  }

void NCSG::setIndex(unsigned index){ m_index = index ;  }
unsigned NCSG::getIndex() const {    return m_index ;  }



std::string NCSG::meta() const 
{
    std::stringstream ss ; 
    ss << " get_treeindex "  << get_treeindex()
       << " get_depth "      << get_depth()
       << " get_nchild "     << get_nchild()
       << " get_lvname "     << get_lvname() 
       << " get_soname "     << get_soname() 
       << " is_skip "        << is_skip()
       << " is_uncoincide "  << is_uncoincide()
       << " get_emit "       << get_emit()
       ;

    return ss.str();
}

std::string NCSG::smry() const 
{
    std::stringstream ss ; 
    ss 
       << " ht " << std::setw(2) << getHeight() 
       << " nn " << std::setw(4) << getNumNodes()
       << " tri " << std::setw(6) << getNumTriangles()
       << " tmsg " << ( m_tris ? m_tris->getMessage() : "NULL-tris" ) 
       << " iug " << m_usedglobally 
       << m_csgdata->smry() 
      ;

    return ss.str();
}

void NCSG::increaseVerbosity(int verbosity)
{
    if(verbosity > -1) 
    {
        if(verbosity > m_verbosity)
        {
            LOG(debug)
                << " treedir " << m_treedir
                << " old " << m_verbosity 
                << " new " << verbosity 
                ; 
        }
        else
        {
            LOG(debug) 
                << "IGNORING REQUEST TO DECREASE verbosity " 
                << " treedir " << m_treedir
                << " current verbosity " << m_verbosity 
                << " requested : " << verbosity 
                ; 
        }
    }
}


