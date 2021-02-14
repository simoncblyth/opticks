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

#include <cstdio>
#include <cassert>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <set>

#include "BFile.hh"
#include "BRng.hh"

#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"

#include "NCSG.hpp"
#include "NNode.hpp"
#include "NPart.hpp"
#include "NQuad.hpp"
#include "Nuv.hpp"
#include "NBBox.hpp"

#include "NNodeDump2.hpp"
#include "NNodePoints.hpp"
#include "NNodeUncoincide.hpp"
#include "NTreeAnalyse.hpp"

#include "NPrimitives.hpp"

#include "PLOG.hh"


unsigned nnode::bb_count = 0 ; 

// see NNodeUncoincide::uncoincide_union
float nnode::z1() const { assert(0 && "nnode::z1 needs override "); return 0 ; } 
float nnode::z2() const { assert(0 && "nnode::z2 needs override "); return 0 ; } 
float nnode::r1() const { assert(0 && "nnode::r1 needs override "); return 0 ; } 
float nnode::r2() const { assert(0 && "nnode::r2 needs override "); return 0 ; } 
void  nnode::increase_z2(float /*dz*/){ assert(0 && "nnode::increase_z2 needs override "); }
void  nnode::decrease_z1(float /*dz*/){ assert(0 && "nnode::decrease_z1 needs override "); }



bool nnode::has_planes() const 
{
    return CSGHasPlanes(type) ;
}
 
unsigned nnode::planeIdx() const 
{
    return param.u.x ; 
}
void nnode::setPlaneIdx(unsigned idx)
{
    param.u.x = idx ; 
}

unsigned nnode::planeNum() const 
{
    return param.u.y ; 
}
void nnode::setPlaneNum(unsigned num)
{
    param.u.y = num ; 
}


std::string nnode::ana_desc() const
{
    return NTreeAnalyse<nnode>::Desc(this) ; 
}
std::string nnode::ana_brief() const
{
    return NTreeAnalyse<nnode>::Brief(this) ; 
}




unsigned nnode::desc_indent = 10 ; 

std::string nnode::desc() const 
{
    std::stringstream ss ; 
    ss << std::setw(desc_indent) << std::left  << tag() ; 
    return ss.str();
}


std::string nnode::id() const 
{
    std::stringstream ss ; 

    //ss << subdepth ; 
    ss   << ( complement ? "!" : "" ) << CSGTag(type)  ;     

    return ss.str();
}

std::string nnode::tag() const 
{
    std::stringstream ss ; 
    ss  
        << "[" 
        << std::setw(2) << idx 
        << ":"
        << id()
        << "]"
        << " " << ( is_primitive() ? "P" : "C" ) 
        ;     
    return ss.str();
}


void nnode::pdump(const char* msg) const 
{
    LOG(info) << msg << " verbosity " << verbosity ; 
    assert(0 && "override nnode::pdump in primitives" );
}


void nnode::nudge(unsigned /*uv_surf*/, float /*delta*/ ) 
{
    assert(0 && "override nnode::nudge in primitives" );
}


bool nnode::is_zero() const 
{
    return type == CSG_ZERO ; 
}
bool nnode::is_lrzero() const 
{
    return is_operator() && left->is_zero() && right->is_zero() ; 
}
bool nnode::is_rzero() const 
{
    return is_operator() && !left->is_zero() && right->is_zero() ; 
}
bool nnode::is_lzero() const 
{
    return is_operator() && left->is_zero() && !right->is_zero() ; 
}




bool nnode::is_operator() const 
{
    return type == CSG_UNION || type == CSG_INTERSECTION || type == CSG_DIFFERENCE ; 
}
bool nnode::is_primitive() const 
{
    return left == NULL && right == NULL ; 
}
bool nnode::is_unbounded() const 
{
    return type == CSG_SLAB || type == CSG_PLANE ; 
}


bool nnode::is_root() const 
{
    return parent == NULL ; 
}
bool nnode::is_bileaf() const 
{
    return !is_primitive() && left->is_primitive() && right->is_primitive() ; 
}

bool nnode::is_znudge_capable() const 
{
    return type == CSG_CYLINDER || type == CSG_CONE || type == CSG_DISC ; 
    //return type == CSG_CYLINDER || type == CSG_CONE || type == CSG_ZSPHERE || type == CSG_DISC ;    // ZSPHERE has radius constraints 
}

void nnode::set_treedir( const char* treedir_)
{
    treedir = treedir_ ? strdup(treedir_) : NULL ; 
}
void nnode::set_treeidx(int treeidx_)
{
    treeidx = treeidx_ ; 
}



void nnode::set_boundary( const char* boundary_)
{
    boundary = boundary_ ? strdup(boundary_) : NULL ; 
}



void nnode::Init( nnode* n , OpticksCSG_t type, nnode* left, nnode* right )
{
    n->idx = 0 ; 
    n->type = type ; 

    n->left = left ; 
    n->right = right ; 
    n->parent = NULL ; 
    n->other  = NULL ;   // used by NOpenMesh 

    std::string tag = CSGTag(type) ; // 2-char 
    n->label = strdup(tag.c_str()); 
 
    n->treedir = NULL ; 
    n->treeidx = -1 ; 
    n->depth = 0 ; 
    n->subdepth = 0 ; 
    n->boundary = NULL ;  

    n->placement = NULL ; 
    n->transform = NULL ; 
    n->gtransform = NULL ; 
    n->gtransform_idx = 0 ; 
    n->itransform_idx = 0 ; 

    n->complement = false ; 
    n->verbosity = 0 ; 

    n->param.u  = {0u,0u,0u,0u};
    n->param1.u = {0u,0u,0u,0u};
    n->param2.u = {0u,0u,0u,0u};
    n->param3.u = {0u,0u,0u,0u};

    n->planes = {} ;
    n->par_points = {};
    n->par_coords = {};

    n->meta = NULL ; 
    n->_dump = new NNodeDump2(n) ; 
    n->_bbox_model = NULL ; 
    n->g4code = NULL ;  
    n->g4name = NULL ;  
    n->g4args = NULL ;  
}




const char* nnode::csgname() const
{
   return CSGName(type);
}
unsigned nnode::maxdepth() const 
{
    return _maxdepth(0);
}
unsigned nnode::_maxdepth(unsigned depth) const   // recursive 
{
    return left && right ? nmaxu( left->_maxdepth(depth+1), right->_maxdepth(depth+1)) : depth ;  
}


const nmat4triple* nnode::global_transform()
{
    return global_transform(this);
}

/**
nnode::global_transform
------------------------

Is invoked by nnode::update_gtransforms_r from each primitive, 
whence parent links are followed up the tree until reaching root
which has no parent. Along the way transforms are collected
into the tvq vector in reverse hierarchical order from 
leaf back up to root

If a placement transform is present on the root node, that 
is also collected. 


* NB these are the CSG nodes, not structure nodes

**/

const nmat4triple* nnode::global_transform(nnode* n)
{
    std::vector<const nmat4triple*> tvq ; 
    nnode* r = NULL ;  
    while(n)
    {
        if(n->transform) tvq.push_back(n->transform);
        r = n ;            // keep hold of the last non-NULL 
        n = n->parent ; 
    }

    if(r->placement) tvq.push_back(r->placement); 


    bool reverse = true ; 
    const nmat4triple* gtransform= tvq.size() == 0 ? NULL : nmat4triple::product(tvq, reverse) ; 

    if(gtransform == NULL )  // make sure all primitives have a gtransform 
    {
        gtransform = nmat4triple::make_identity() ;
    }
    return gtransform ; 
}


/**
nnode::set_centering
-----------------------

Obtains the center extent from the global analytic bounding box, 
and then contructs a translation moving the center to the origin.
After this the bounding box is expected to be centered. 

**/

void nnode::set_centering()
{
    nbbox bb0 = bbox(); // <-- global
    nvec4 ce0 = bb0.center_extent(); 
    nvec3 c0 = nvec3::from_vec4(ce0) ; 
    bool centered0 = c0.is_zero(1e-5) ;  

    if(verbosity > 0)
    LOG(info) 
        << " bb0 " << bb0.description()
        << " ce0 " << ce0.desc()
        << ( centered0 ? " CENTERED " : " NOT-CENTERED " )
        ; 

    if(centered0) return ; 

    glm::vec3 tla( -ce0.x, -ce0.y, -ce0.z );  
    set_translation( tla.x, tla.y, tla.z ); 

    nbbox bb1 = bbox(); // <-- global
    nvec4 ce1 = bb1.center_extent(); 
    nvec3 c1 = nvec3::from_vec4(ce1) ; 


    bool centered1 = c1.is_zero(2e-5) ;  

    if(!centered1)
    {
        LOG(info) << "c1 " << c1.descg() ;   
    }

    if(verbosity > 0)
    LOG(info) 
        << " bb1 " << bb1.description()
        << " ce1 " << ce1.desc()
        << ( centered1 ? " CENTERED " : " NOT-CENTERED " )
        ; 

    assert( centered1 ); 
}

bool nnode::has_placement() const
{
    return placement != NULL ;  
}
bool nnode::has_placement_translation() const
{
    return placement != NULL && placement->is_translation_only() ;  
}
glm::vec3 nnode::get_placement_translation() const 
{
    assert( has_placement_translation() ); 
    return placement->get_translation() ; 
}


bool nnode::has_placement_transform() const
{
    return placement != NULL && !placement->is_translation_only() ;  
}

glm::mat4 nnode::get_placement_transform() const 
{
    return placement->t  ; 
}




bool nnode::has_root_transform() const
{
    return transform != NULL  ;  
}
glm::mat4 nnode::get_root_transform() const 
{
    return transform->t  ; 
}









/**
nnode::set_translation
-----------------------

Invoked from NCSG::set_translation/GMesh::applyCentering from GGeoTest

**/

void nnode::set_translation( float x, float y, float z )
{
    const nmat4triple* plc = nmat4triple::make_translate( x, y, z );  
    set_placement( plc ); 
}


/**
nnode::set_placement
-----------------------

Formerly when this was called apply_placement
the placement was nullified after update_gtransforms()
but it appears that methods such as NCSG::collect_global_transforms 
are going to invoke node->global_transform() again on all 
primitives ... so need to leave the placement in place.

**/

void nnode::set_placement( const nmat4triple* plc )
{
    assert( is_root() ) ; 
    placement = plc ; 
    update_gtransforms(); 

    // placement = NULL ;   
}



void nnode::check_tree(unsigned mask) const 
{
    check_tree_r(this, NULL, 0, mask); 
}
void nnode::check_tree_r(const nnode* node, const nnode* parent, unsigned depth, unsigned mask) // static
{
    if( mask & FEATURE_PARENT_LINKS )
    {
        if(depth == 0) assert( node->parent == NULL ); 
        if(depth > 0)  assert( node->parent != NULL ); 
        assert( node->parent == parent ) ; 
    }

    if(node->is_primitive())
    {
        if( mask & FEATURE_GTRANSFORMS )
        {
            assert( node->gtransform != NULL ); 
        }

        if( mask & FEATURE_GTRANSFORM_IDX )
        {
            assert( node->gtransform_idx > 0 ); 
        }
    } 
    else
    {
        if( mask & FEATURE_GTRANSFORMS )
        {
            assert( node->gtransform == NULL ); 
        }
        if( mask & FEATURE_GTRANSFORM_IDX )
        {
            assert( node->gtransform_idx == 0 ); 
        }
    }

    if(node->left && node->right)
    {
        check_tree_r(node->left, node, depth+1, mask);
        check_tree_r(node->right, node, depth+1, mask);
    }
}




void nnode::collect_ancestors( std::vector<const nnode*>& ancestors, OpticksCSG_t qtyp) const 
{
    if(!parent) return ;   // start from parent to avoid collecting self
    collect_ancestors_(parent, ancestors, qtyp);
}
void nnode::collect_ancestors_( const nnode* n, std::vector<const nnode*>& ancestors, OpticksCSG_t qtyp) // static
{
    while(n)
    {
        if(n->type == qtyp || qtyp == CSG_ZERO) ancestors.push_back(n);
        n = n->parent ; 
    }
}



void nnode::collect_connectedtype_ancestors( std::vector<const nnode*>& ancestors) const 
{
    if(!parent) return ;   // start from parent to avoid collecting self
    collect_connectedtype_ancestors_(parent, ancestors, parent->type);
}
void nnode::collect_connectedtype_ancestors_( const nnode* n, std::vector<const nnode*>& ancestors, OpticksCSG_t qtyp) // static
{
    while(n && n->type == qtyp)
    {
        ancestors.push_back(n);
        n = n->parent ; 
    }
}





const nnode* nnode::find_one( OpticksCSG_t qtyp ) const   // returns NULL if exactly one node is not found
{
    std::vector<OpticksCSG_t> typ = { qtyp } ; 
    return find_one(typ);
}
const nnode* nnode::find_one( OpticksCSG_t qtyp1, OpticksCSG_t qtyp2 ) const   // returns NULL if exactly one node is not found
{
    std::vector<OpticksCSG_t> typ = { qtyp1, qtyp2 } ; 
    return find_one(typ);
}
const nnode* nnode::find_one( std::vector<OpticksCSG_t>& qtyp ) const   // returns NULL if exactly one node is not found
{
    std::vector<const nnode*> nodes ; 
    collect_nodes( nodes, qtyp );
    return nodes.size() == 1 ? nodes[0] : NULL  ; 
}
void nnode::collect_nodes( std::vector<const nnode*>& nodes, std::vector<OpticksCSG_t>& qtyp ) const 
{
    collect_nodes_r(this, nodes, qtyp );
}
void nnode::collect_nodes_r( const nnode* n, std::vector<const nnode*>& nodes, std::vector<OpticksCSG_t>& qtyp  ) // static
{
    if(std::find(qtyp.begin(), qtyp.end(), n->type ) != qtyp.end() )
    { 
        nodes.push_back(n) ; 
    }

    if(n->left && n->right)
    {
        collect_nodes_r(n->left, nodes, qtyp);
        collect_nodes_r(n->right, nodes, qtyp);
    }   
}








void nnode::collect_progeny( std::vector<const nnode*>& progeny, OpticksCSG_t xtyp ) const 
{
    if(left && right)  // start from left/right to avoid collecting self
    {
        collect_progeny_r(left, progeny, xtyp);
        collect_progeny_r(right, progeny, xtyp);
    }
}
void nnode::collect_progeny_r( const nnode* n, std::vector<const nnode*>& progeny, OpticksCSG_t xtyp ) // static
{
    if(n->type != xtyp || xtyp == CSG_ZERO)  // huh why != xtyp,  excluded type perhaps ?
    {
        if(std::find(progeny.begin(), progeny.end(), n) == progeny.end()) progeny.push_back(n);
    }

    if(n->left && n->right)
    {
        collect_progeny_r(n->left, progeny, xtyp);
        collect_progeny_r(n->right, progeny, xtyp);
    }   
}


void nnode::collect_monogroup( std::vector<const nnode*>& monogroup ) const 
{
   if(!parent) return ;    

   // follow parent links collecting ancestors until reach ancestor of another CSG type
   // eg on starting with a primitive of CSG_UNION parent finds 
   // direct lineage ancestors that are also CSG_UNION

   std::vector<const nnode*> connectedtype ; 
   collect_connectedtype_ancestors(connectedtype);  

   // then for each of those same type ancestors collect
   // all progeny... eg for CSG_UNION parent this
   // will collect all operators and primitives in the union

   OpticksCSG_t xtyp = parent->type ;  // exclude the operators from the monogroup

   for(unsigned c=0 ; c < connectedtype.size() ; c++)
   {
       const nnode* cu = connectedtype[c];
       cu->collect_progeny( monogroup, xtyp ); 
   }  
}
bool nnode::is_same_monogroup(const nnode* a, const nnode* b, OpticksCSG_t op)  // static
{
   if(!a->parent || !b->parent || a->parent->type != op || b->parent->type != op) return false ;  

   std::vector<const nnode*> monogroup ; 
   a->collect_monogroup(monogroup);

   return std::find(monogroup.begin(), monogroup.end(), b ) != monogroup.end() ;
}

bool nnode::is_same_union(const nnode* a, const nnode* b) // static
{
   return is_same_monogroup(a,b, CSG_UNION ); 
} 




/**
nnode::update_gtransforms
--------------------------

Sets global transforms for all primitives, by multiplying the
level transforms for parent nodes.  If no level transforms are collected
the gtransform is set to the identity matrix. 

**/

void nnode::update_gtransforms()
{
    update_gtransforms_r(this);
}
void nnode::update_gtransforms_r(nnode* node)
{
    // NB this downwards traversal doesnt need parent links, 
    // but the ancestor collection of global_transforms that it uses does...

    if( node->is_primitive() )  // this was setting gtransform on all nodes until Aug 1 2018
    {
        node->gtransform = node->global_transform();
    }

    if(node->left && node->right)
    {
        update_gtransforms_r(node->left);
        update_gtransforms_r(node->right);
    }
}






unsigned nnode::get_type_mask() const { return get_mask(NODE_ALL) ; }
unsigned nnode::get_prim_mask() const { return get_mask(NODE_PRIMITIVE) ; }
unsigned nnode::get_oper_mask() const { return get_mask(NODE_OPERATOR) ; }

std::string nnode::get_type_mask_string() const { return get_mask_string(NODE_ALL) ; }
std::string nnode::get_prim_mask_string() const { return get_mask_string(NODE_PRIMITIVE) ; }
std::string nnode::get_oper_mask_string() const { return get_mask_string(NODE_OPERATOR) ; }



/**
nnode::get_mask
----------------

returns a mask integer holding the types of all nodes in the tree
that meet the NNodeType criteria allowing selection of all/operators/primitives

**/
unsigned nnode::get_mask(NNodeType ntyp) const 
{
    unsigned msk = 0 ;   
    get_mask_r(this, ntyp, msk );
    return msk ; 
}
void nnode::get_mask_r(const nnode* node, NNodeType ntyp, unsigned& msk ) // static
{
    bool collect = false ;  
    if(node->type < 32)
    {
       if(     ntyp == NODE_ALL) collect = true ; 
       else if(ntyp == NODE_PRIMITIVE && node->type >= CSG_SPHERE ) collect = true  ;
       else if(ntyp == NODE_OPERATOR  && (node->type == CSG_UNION || node->type == CSG_INTERSECTION || node->type == CSG_DIFFERENCE )) collect = true  ;
    }
    if(collect)  msk |= (0x1 << node->type) ;


    if(node->left && node->right)
    {
        get_mask_r(node->left, ntyp, msk);
        get_mask_r(node->right, ntyp, msk);
    }
} 
std::string nnode::get_mask_string(NNodeType ntyp) const 
{
    unsigned msk = get_mask(ntyp);
    std::stringstream ss ; 
    for(unsigned i=0 ; i < 32 ; i++) if(msk & (0x1 << i)) ss << CSGName((OpticksCSG_t)i) << " " ;   
    return ss.str();
}



bool nnode::has_torus() const
{
    return get_count(CSG_TORUS) > 0 ; 
}
bool nnode::is_box() const
{
    return is_primitive() && type == CSG_BOX ; 
}
bool nnode::is_box3() const
{
    return is_primitive() && type == CSG_BOX3 ; 
}





unsigned nnode::get_count(OpticksCSG_t typ) const 
{
    unsigned count = 0 ;
    get_count_r(this, typ, count );  
    return count ;  
}
void nnode::get_count_r(const nnode* node, OpticksCSG_t typ, unsigned& count) // static
{
    if( node->type == typ ) count += 1 ; 

    if(node->left && node->right)
    {
        get_count_r(node->left, typ, count);
        get_count_r(node->right, typ, count);
    }
}
 






void nnode::dump_g4code() const 
{
    std::ostream& out = std::cout ;
    to_g4code( this, out, 0); 
}
void nnode::write_g4code(const char* path_) const 
{
    std::string path = BFile::FormPath(path_) ;
    std::ofstream out(path.c_str());
    to_g4code( this, out, 0 ); 
}

void nnode::to_g4code(const nnode* root, std::ostream& out, unsigned depth )   // static
{
    out << "// start portion generated by nnode::to_g4code " << std::endl ; 
    out << "G4VSolid* make_solid()" << std::endl ; 
    out << "{ " << std::endl ; 

    to_g4code_r( root, out, 0 ); 

    out << "    return a ; " << std::endl ; 
    out << "} " << std::endl ; 
    out << "// end portion generated by nnode::to_g4code " << std::endl ; 
}

void nnode::to_g4code_r(const nnode* node, std::ostream& out, unsigned depth )  // static
{
    if(node->left && node->right)
    {
        // hmm : this doesnt follow for polycone, as its an nnode union tree, but a primitive in G4 
        //assert( node->left->g4code && node->right->g4code );
        if( node->left->g4code && node->right->g4code )
        {
            to_g4code_r( node->left, out, depth+1 ) ; 
            to_g4code_r( node->right, out, depth+1 ) ; 
        }
        else
        {
            LOG(error) << "no g4code on left/right :  prim in G4, but tree in Opticks perhaps ? " ;             
        }
    } 
    assert( node->g4code );   
    out 
        << node->g4code      // no indent : its already indented
        << " // " 
        << depth 
        << std::endl ;  
}




glm::vec3 nnode::apply_gtransform(const glm::vec4& v_) const 
{
    glm::vec4 v(v_) ; 
    if(gtransform) v = gtransform->t * v ; 
    return glm::vec3(v) ; 
}


glm::vec3 nnode::center() const    // override in shapes if needed
{
    glm::vec3 c(0,0,0); 
    return c ;
}
glm::vec3 nnode::direction() const    // override in shapes if needed
{
    glm::vec3 d(1,1,1); 
    return d ;
}


glm::vec3 nnode::gseeddir() const    
{
    glm::vec4 d(direction(), 0.f);
    return apply_gtransform(d);
}
glm::vec3 nnode::gseedcenter() const 
{
    glm::vec4 c(center(),1.f);
    return apply_gtransform(c);
}



npart nnode::srcpart() const 
{
    npart pt ; 
    pt.zero();
    pt.setParam(  param );
    pt.setParam1( param1 );
    pt.setParam2( param2 );
    pt.setParam3( param3 );

    pt.setTypeCode( type );
    pt.setITransform( itransform_idx, complement ); // hmm complemented sources ?
    return pt ; 
}

npart nnode::part() const 
{
    // this is invoked by NCSG::export_r to totally re-write the nodes buffer 
    // BUT: is it being used by partlist approach, am assuming not by not setting bbox

    npart pt ; 
    pt.zero();
    pt.setParam(  param );
    pt.setParam1( param1 );
    pt.setParam2( param2 );
    pt.setParam3( param3 );

    pt.setTypeCode( type );
    pt.setGTransform( gtransform_idx, complement );

    // gtransform_idx is index into a buffer of the distinct compound transforms for the tree

    if(npart::VERSION == 0u)
    {
        nbbox bb = bbox();
        pt.setBBox( bb );  
    }

    return pt ; 
}


void nnode::check_primitive_bb( const nbbox& bb) const 
{
    bool invert_is_complement = bb.invert == complement ;

    if(!invert_is_complement)
    {
        LOG(fatal) << "nnode::check_primitive_bb"
                      << " invert_is_complement FAIL "
                      << " bb  " << bb.desc()
                      << " node  " << desc()
                      ;
    }
                     
    assert( invert_is_complement );
}

/**
nnode::get_composite_bbox
---------------------------

Is this pre-cache only ?

**/

void nnode::get_composite_bbox( nbbox& bb ) const 
{
    assert( left && right );

    bool l_unbound = left->is_unbounded();
    bool r_unbound = right->is_unbounded();

    bool lr_unbound = l_unbound && r_unbound ;
    if(lr_unbound)
    {
        LOG(warning) << "nnode::get_composite_bbox lr_unbound leave bb as is " ; 
        return ; 
    }   
    //assert( !lr_unbound  && " combination of two unbounded prmitives is not allowed " );


    if(verbosity > 4 )
    {
         LOG(info) 
             << " l_unbound " << l_unbound  
             << " r_unbound " << r_unbound  
             << " lr_unbound " << lr_unbound  
             << " this " << this
             << " left " << left
             << " right " << right
             << " ll " << left->left 
             << " lr " << left->right
             << " rl " << right->left 
             << " rr " << right->right
             ;
    }

    nbbox l_bb = left->bbox();
    if( verbosity > 4 ) LOG(info) << "l_bb " << l_bb.desc() ; 

    nbbox r_bb = right->bbox();
    if( verbosity > 4 ) LOG(info) << "r_bb " << r_bb.desc() ; 


    if( left->is_unbounded() )
    {
        assert(l_bb.is_empty());
        bb = r_bb ; 
    }
    else if( right->is_unbounded() )
    {
        assert(r_bb.is_empty());
        bb = l_bb ; 
    }
    else
    {
        if(left->is_primitive()) left->check_primitive_bb(l_bb);
        if(right->is_primitive()) right->check_primitive_bb(r_bb);

        nbbox::CombineCSG(bb, l_bb, r_bb, type, verbosity  );
    }

    if(verbosity > 0)
    std::cout << "nnode::composite_bbox "
              << " left " << left->desc()
              << " right " << right->desc()
              << " bb " << bb.desc()
              << std::endl 
              ;

}  



/**
nnode::get_primitive_bbox
--------------------------

Relies on different action for base classes and subclasses
to prevent infinite recursion.

**/

void nnode::get_primitive_bbox(nbbox& bb) const 
{
    assert(is_primitive());

    const nnode* node = this ;  

    if(node->is_unbounded())
    {
        LOG(error) << "nnode::get_primitive_bbox not providing bbox for unbounded primitive " ; 
    }
    else if(node->type == CSG_SPHERE)
    { 
        const nsphere* n = dynamic_cast<const nsphere*>(node) ;
        if(!n) LOG(fatal) << "failed to dynamic_cast a node of type CSG_SPHERE to const nsphere " ;  assert(n) ; 
        //nsphere* n = (nsphere*)node ;
        //if(!n) LOG(fatal) << "failed to cast a node of type CSG_SPHERE to nsphere " ;  assert(n) ; 
        nbbox pp = n->bbox() ;
        bb.copy_from(pp) ; 
    }
    else if(node->type == CSG_ZSPHERE)
    {
        nzsphere* n = (nzsphere*)node ;
        nbbox pp = n->bbox() ;
        bb.copy_from(pp) ; 
    }
    else if(node->type == CSG_BOX || node->type == CSG_BOX3)
    {
        if(verbosity > 4 ) LOG(info) << " CSG_BOX/CSG_BOX3 " ; 
        nbox* n = (nbox*)node ;
        nbbox pp = n->bbox() ;   // <-- if the node is not a real nbox 
        bb.copy_from(pp) ; 
    }
    else if(node->type == CSG_CYLINDER)
    {
        ncylinder* n = (ncylinder*)node ;
        nbbox pp = n->bbox() ;
        bb.copy_from(pp) ; 
    } 
    else if(node->type == CSG_DISC)
    {
        ndisc* n = (ndisc*)node ;
        nbbox pp = n->bbox() ;
        bb.copy_from(pp) ; 
    }
    else if(node->type == CSG_CONE)
    {
        ncone* n = (ncone*)node ;
        nbbox pp = n->bbox() ;
        bb.copy_from(pp) ; 
    }
    else if(node->type == CSG_CONVEXPOLYHEDRON)
    {
        nconvexpolyhedron* n = (nconvexpolyhedron*)node ; 
        nbbox pp = n->bbox() ;
        bb.copy_from(pp) ; 
    }
    else
    {
        LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSGName(node->type) ;  
        assert(0);
    }
}


/**
nnode::bbox()
---------------

The gtransforms are applied at the leaves, ie the bbox returned
from primitives already uses the full heirarchy of transforms 
collected from the tree by *update_gtransforms()*.  

Due to this it would be incorrect to apply gtransforms 
of composite nodes to their bbox as those gtransforms 
together with those of their progeny have already been 
applied down at the leaves.

Indeed without subnode bbox being in the same CSG tree top frame
it would not be possible to combine them.

**/

nbbox nnode::bbox() const 
{
    nnode::bb_count += 1 ; 

    if(verbosity > 0)
    {
        LOG(info) << "nnode::bbox " << desc() << " " << this << " bb_count " << nnode::bb_count ; 
    }

    //assert( nnode::bb_count < 10 );  // just for debug some infinite recursion  

    nbbox bb = make_bbox() ; 

    if(is_primitive())
    {
        get_primitive_bbox(bb);
    } 
    else 
    {
        get_composite_bbox(bb);
    }
    return bb ; 
}



glm::vec4 nnode::bbox_center_extent() const 
{
   nbbox bb = bbox(); 
   return bb.ce(); 
}



nnode* nnode::Load(const char* treedir, const NSceneConfig* config)
{
    NCSG* tree = NCSG::Load(treedir, config);
    nnode* root = tree->getRoot();
    return root ; 
}



/**
To translate or rotate a surface modeled as an SDF, you can apply the inverse
transformation to the point before evaluating the SDF.

**/


float nunion::operator()(float x, float y, float z) const 
{
    assert( left && right );
    float l = (*left)(x, y, z) ;
    float r = (*right)(x, y, z) ;
    return fminf( l, r );
}
float nintersection::operator()(float x, float y, float z) const 
{
    assert( left && right );
    float l = (*left)(x, y, z) ;
    float r = (*right)(x, y, z) ;
    return fmaxf( l, r );
}
float ndifference::operator()(float x, float y, float z) const 
{
    assert( left && right );
    float l = (*left)(x, y, z) ;
    float r = (*right)(x, y, z) ;
    return fmaxf( l, -r);    // difference is intersection with complement, complement negates signed distance function
}





void nnode::ResizeToFit(nnode* root, const nbbox& container, float scale, float delta ) 
{
    if( root->type == CSG_BOX || root->type == CSG_BOX3)
    {
        nbox* box = dynamic_cast<nbox*>(root)  ;
        assert(box) ; 
        box->resizeToFit(container, scale, delta );
    }
    else if( root->type == CSG_SPHERE )
    {
        nsphere* sph = dynamic_cast<nsphere*>(root)  ;
        assert(sph) ; 
        sph->resizeToFit(container, scale, delta );
    }
    else
    {
        LOG(fatal)
            << " auto-containement only implemented for BOX and SPHERE"
            << " root: " << root->desc()
            ; 
        assert(0 && "nnode::ResizeToFit" ); 
    }
}


/**
nnode::copy
-------------

See test/NTreeBuilderTest.cc:test_bbox for the motivation 
for node instance copying that passes along the vtable.
Essentially::

    //nnode* b = new nnode(*a) ;       // <-- culprit : CAUSING THE INFINITE RECURSION
    //nnode* b = new nbox(*(nbox*)a) ;   // <--- fix
    nnode* b = a->make_copy();   
   
**/
nnode* nnode::copy( const nnode* node )  // static
{
    nnode* c = NULL ; 
    typedef nconvexpolyhedron ncxpol ; 
    switch(node->type)   
    {
        case CSG_UNION:           { nunion* n        = (nunion*)node         ; c = new nunion(*n)        ; } break ;
        case CSG_INTERSECTION:    { nintersection* n = (nintersection*)node  ; c = new nintersection(*n) ; } break ;
        case CSG_DIFFERENCE:      { ndifference* n   = (ndifference*)node    ; c = new ndifference(*n)   ; } break ;
        case CSG_SPHERE:          { nsphere* n       = (nsphere*)node        ; c = new nsphere(*n)       ; } break ;
        case CSG_ZSPHERE:         { nzsphere* n      = (nzsphere*)node       ; c = new nzsphere(*n)      ; } break ;
        case CSG_BOX:             { nbox* n          = (nbox*)node           ; c = new nbox(*n)          ; } break ;
        case CSG_BOX3:            { nbox* n          = (nbox*)node           ; c = new nbox(*n)          ; } break ;
        case CSG_SLAB:            { nslab* n         = (nslab*)node          ; c = new nslab(*n)         ; } break ; 
        case CSG_PLANE:           { nplane* n        = (nplane*)node         ; c = new nplane(*n)        ; } break ; 
        case CSG_CYLINDER:        { ncylinder* n     = (ncylinder*)node      ; c = new ncylinder(*n)     ; } break ; 
        case CSG_DISC:            { ndisc* n         = (ndisc*)node          ; c = new ndisc(*n)         ; } break ; 
        case CSG_CONE:            { ncone* n         = (ncone*)node          ; c = new ncone(*n)         ; } break ; 
        case CSG_CONVEXPOLYHEDRON:{ ncxpol* n        = (ncxpol*)node         ; c = new ncxpol(*n)        ; } break ; 
        case CSG_TORUS:           { ntorus* n        = (ntorus*)node         ; c = new ntorus(*n)        ; } break ; 
        case CSG_CUBIC:           { ncubic* n        = (ncubic*)node         ; c = new ncubic(*n)        ; } break ; 
        case CSG_HYPERBOLOID:     { nhyperboloid* n  = (nhyperboloid*)node   ; c = new nhyperboloid(*n)  ; } break ; 
        default:
            LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSGName(node->type) ;  
            assert(0);
    }
    return c ;
}

nnode* nnode::make_copy() const 
{
    return nnode::copy(this); 
}




nnode* nnode::deepcopy_r( const nnode* n ) // static 
{
    if( n == NULL ) return NULL ; 
    nnode* c = nnode::copy(n) ;       // ARE TRANSFORMS COPIED  ?
    c->left = deepcopy_r(n->left) ; 
    c->right = deepcopy_r(n->right) ;  
    return c ; 
}
nnode* nnode::make_deepcopy() const  
{
    return nnode::deepcopy_r(this); 
}




std::function<float(float,float,float)> nnode::sdf() const 
{
    //  return node types operator() callable as function
    //  somehow the object parameter member data goes along for the ride

    const nnode* node = this ; 
    std::function<float(float,float,float)> f ; 

    switch(node->type)   
    {
        case CSG_UNION:          { nunion* n        = (nunion*)node         ; f = *n ; } break ;
        case CSG_INTERSECTION:   { nintersection* n = (nintersection*)node  ; f = *n ; } break ;
        case CSG_DIFFERENCE:     { ndifference* n   = (ndifference*)node    ; f = *n ; } break ;
        case CSG_SPHERE:         { nsphere* n       = (nsphere*)node        ; f = *n ; } break ;
        case CSG_ZSPHERE:        { nzsphere* n      = (nzsphere*)node       ; f = *n ; } break ;
        case CSG_BOX:            { nbox* n          = (nbox*)node           ; f = *n ; } break ;
        case CSG_BOX3:           { nbox* n          = (nbox*)node           ; f = *n ; } break ;
        case CSG_SLAB:           { nslab* n         = (nslab*)node          ; f = *n ; } break ; 
        case CSG_PLANE:          { nplane* n        = (nplane*)node         ; f = *n ; } break ; 
        case CSG_CYLINDER:       { ncylinder* n     = (ncylinder*)node      ; f = *n ; } break ; 
        case CSG_DISC:           { ndisc* n         = (ndisc*)node          ; f = *n ; } break ; 
        case CSG_CONE:           { ncone* n         = (ncone*)node          ; f = *n ; } break ; 
        case CSG_CONVEXPOLYHEDRON:{ nconvexpolyhedron* n = (nconvexpolyhedron*)node ; f = *n ; } break ; 
        case CSG_TORUS:          { ntorus* n = (ntorus*)node ; f = *n ; } break ; 
        case CSG_CUBIC:          { ncubic* n = (ncubic*)node ; f = *n ; } break ; 
        case CSG_HYPERBOLOID:    { nhyperboloid* n = (nhyperboloid*)node ; f = *n ; } break ; 
        default:
            LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSGName(node->type) ;  
            assert(0);
    }
    return f ;
}









/*
float nnode::operator()(const glm::vec3& p) const 
{
   // presumably more efficient to grab the sdf once and use that, rather than doinf this for every point ?
    float f = 0.f ; 
    const nnode* node = this ; 
    switch(node->type)   
    {
        case CSG_UNION:          { nunion* n        = (nunion*)node         ; f = (*n)(p.x,p.y,p.z) ; } break ;
        case CSG_INTERSECTION:   { nintersection* n = (nintersection*)node  ; f = (*n)(p.x,p.y,p.z) ; } break ;
        case CSG_DIFFERENCE:     { ndifference* n   = (ndifference*)node    ; f = (*n)(p.x,p.y,p.z) ; } break ;
        case CSG_SPHERE:         { nsphere* n       = (nsphere*)node        ; f = (*n)(p.x,p.y,p.z) ; } break ;
        case CSG_ZSPHERE:        { nzsphere* n      = (nzsphere*)node       ; f = (*n)(p.x,p.y,p.z) ; } break ;
        case CSG_BOX:            { nbox* n          = (nbox*)node           ; f = (*n)(p.x,p.y,p.z) ; } break ;
        case CSG_BOX3:           { nbox* n          = (nbox*)node           ; f = (*n)(p.x,p.y,p.z) ; } break ;
        case CSG_SLAB:           { nslab* n         = (nslab*)node          ; f = (*n)(p.x,p.y,p.z) ; } break ; 
        case CSG_PLANE:          { nplane* n        = (nplane*)node         ; f = (*n)(p.x,p.y,p.z) ; } break ; 
        case CSG_CYLINDER:       { ncylinder* n     = (ncylinder*)node      ; f = (*n)(p.x,p.y,p.z) ; } break ; 
        case CSG_DISC:           { ndisc* n         = (ndisc*)node          ; f = (*n)(p.x,p.y,p.z) ; } break ; 
        case CSG_CONE:           { ncone* n         = (ncone*)node          ; f = (*n)(p.x,p.y,p.z) ; } break ; 
        case CSG_CONVEXPOLYHEDRON:{ nconvexpolyhedron* n = (nconvexpolyhedron*)node ; f = (*n)(p.x,p.y,p.z) ; } break ; 
        default:
            LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSGName(node->type) ;  
            assert(0);
    }
    return f; 
} 
*/


float nnode::operator()(float , float , float ) const 
{
    assert(0 && "nnode::operator() needs override ");
    return 0.f ; 
}

float nnode::sdf_(const glm::vec3& , NNodeFrameType ) const 
{
    assert(0 && "nnode::sdf_ needs override ");
    return 0.f ; 
}











glm::vec3 nnode::par_pos_(const nuv& uv, NNodeFrameType fty) const 
{
    glm::vec3 pos ; 
    switch(fty)
    {
        case FRAME_MODEL:  pos = par_pos_(uv, NULL)       ; break ; 
        case FRAME_LOCAL:  pos = par_pos_(uv, transform)  ; break ; 
        case FRAME_GLOBAL: pos = par_pos_(uv, gtransform) ; break ; 
    }
    return pos ; 
}

glm::vec3 nnode::par_pos_(const nuv& uv, const nmat4triple* triple) const 
{
    glm::vec3 mpos = par_pos_model(uv);
    glm::vec4 tpos(mpos, 1) ; 
    if(triple) tpos = triple->t * tpos ;   // <-- direct transform, not inverse
    return glm::vec3(tpos) ; 
}





glm::vec3 nnode::par_pos_local( const nuv& uv) const { return par_pos_(uv, transform) ; }
glm::vec3 nnode::par_pos_global(const nuv& uv) const { return par_pos_(uv, gtransform) ; }
//glm::vec3 nnode::par_pos(       const nuv& uv) const { return par_pos_(uv, gtransform) ; }


int nnode::par_euler() const 
{
    assert(0 && "this need to be overridden");
    return 0 ; 
}

unsigned nnode::par_nsurf() const 
{
    assert(0 && "this need to be overridden");
    return 0 ; 
}
unsigned nnode::par_nvertices(unsigned , unsigned ) const 
{
    assert(0 && "this need to be overridden");
    return 0 ; 
}

/*
glm::vec3 nnode::par_pos(const nuv& uv) const  // override in shapes 
{
    unsigned surf = uv.s();
    assert(0 && "this need to be overridden");
    assert( surf < par_nsurf());
    glm::vec3 pos ;
    return pos ;  
}
*/








void nnode::_par_pos_endcap(glm::vec3& pos,  const nuv& uv, const float r_, const float z_) // static 
{
    unsigned s  = uv.s(); 
    //unsigned u  = uv.u() ; 
    unsigned v  = uv.v() ; 

    assert( s == 1 || s == 2 );  // endcaps

    bool is_north_pole = v == 0 && s == 1 ; 
    bool is_south_pole = v == 0 && s == 2 ; 
    bool is_pole = is_north_pole || is_south_pole ;

    pos.z = z_ ;  

    float r = r_*uv.fv() ; 
    bool seamed = true ; 
    float azimuth = uv.fu2pi(seamed); 

    float ca = cosf(azimuth);
    float sa = sinf(azimuth);

    if(!is_pole)
    {
        pos += glm::vec3( r*ca, r*sa, 0.f );
    }
}


void nnode::collect_prim_centers(std::vector<glm::vec3>& centers, std::vector<glm::vec3>& dirs,  const nnode* p  )
{
     centers.push_back(p->gseedcenter()); 
     dirs.push_back(p->gseeddir());

/*
    switch(p->type)
    {
        case CSG_SPHERE: 
           {  
               nsphere* n = (nsphere*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  

        case CSG_ZSPHERE: 
           {  
               nzsphere* n = (nzsphere*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  
      
        case CSG_BOX: 
        case CSG_BOX3: 
           {  
               nbox* n = (nbox*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  

        case CSG_SLAB: 
           {  
               nslab* n = (nslab*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  

        case CSG_PLANE: 
           {  
               nplane* n = (nplane*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  

        case CSG_CYLINDER: 
           {  
               ncylinder* n = (ncylinder*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  

        case CSG_DISC: 
           {  
               ndisc* n = (ndisc*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  

        case CSG_CONE: 
           {  
               ncone* n = (ncone*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  

        case CSG_TORUS: 
           {  
               ntorus* n = (ntorus*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  

        case CSG_HYPERBOLOID: 
           {  
               nhyperboloid* n = (nhyperboloid*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  

        case CSG_CONVEXPOLYHEDRON: 
           {  
               nconvexpolyhedron* n = (nconvexpolyhedron*)p ;
               centers.push_back(n->gseedcenter()); 
               dirs.push_back(n->gseeddir());
           }
           break ;  

        default:
           {
               LOG(fatal) << "nnode::collect_prim_centers unhanded shape type " << p->type << " name " << CSGName(p->type) ;
               assert(0) ;
           }
    }
*/

}



void nnode::collect_prim_centers(std::vector<glm::vec3>& centers, std::vector<glm::vec3>& dirs, int verbosity_)
{
    std::vector<const nnode*> prim ; 
    collect_prim(prim);    // recursive collection of list of all primitives in tree
    unsigned nprim = prim.size();

    if(verbosity_ > 0)
    {
        LOG(info) << "nnode::collect_prim_centers"
                  << " verbosity " << verbosity_
                  << " nprim " << nprim 
                  ;
    } 

    for(unsigned i=0 ; i < nprim ; i++)
    {
        const nnode* p = prim[i] ; 

        if(verbosity_ > 1 )
        {
           LOG(info) << "nnode::collect_prim_centers"
                     << " i " << i 
                     << " type " << p->type 
                     << " name " << CSGName(p->type) 
                     ;
        }


        collect_prim_centers( centers, dirs, p );

    }
}




void nnode::collect_prim_for_edit(std::vector<nnode*>& prim) 
{
    collect_prim_for_edit_r(prim, this);   
}
void nnode::collect_prim_for_edit_r(std::vector<nnode*>& prim, nnode* node) // static
{
    bool internal = node->left && node->right ; 
    if(!internal)
    {
        prim.push_back(node);
    }
    else
    {
        collect_prim_for_edit_r(prim, node->left);
        collect_prim_for_edit_r(prim, node->right);
    }
}


unsigned nnode::get_num_prim() const 
{
    std::vector<const nnode*> prim ; 
    collect_prim(prim);
    return prim.size() ;
}
void nnode::collect_prim(std::vector<const nnode*>& prim) const 
{
    collect_prim_r(prim, this);   
}
void nnode::collect_prim_r(std::vector<const nnode*>& prim, const nnode* node) // static
{
    bool internal = node->left && node->right ; 
    if(!internal)
    {
        prim.push_back(node);
    }
    else
    {
        collect_prim_r(prim, node->left);
        collect_prim_r(prim, node->right);
    }
}




/** 
nnode::set_parent_links_r
--------------------------

Parent are setup on import NCSG::import_r
from the complete binary buffer.  But when 
Adoption there is no "import" so need a separate 
setup of parent links.
Parent links are necessary for calculating 
gtransforms.

**/
void nnode::Set_parent_links_r(nnode* node, nnode* parent) // static 
{
    if(node->parent == NULL && parent != NULL)
    {
        LOG(verbose) << " change parent link from NULL for  " << ( node->label ? node->label : "-")  ; 
        node->parent = parent ; 
    }
    else if( node->parent == parent )
    {
        LOG(verbose) << " unchanged parent link  " ; 
    }
    else if( node->parent != parent )
    {
        LOG(fatal) << " NOT changing parent links "
                   << "\n node         : " << node 
                   << "\n parent       : " << parent
                   << "\n node->parent : " << node->parent
                   ;
        assert( 0 && "not expecting to change parent links") ; 
    }

    if(node->left && node->right)
    {
        Set_parent_links_r(node->left,  node);
        Set_parent_links_r(node->right, node);
    }
}


void nnode::prepTree()  
{
    nnode* root = this ; 
    nnode::Set_parent_links_r(root, NULL);
    root->check_tree( FEATURE_PARENT_LINKS ); 
    root->update_gtransforms() ;  // sets node->gtransform (not gtransform_idx) parent links are required 
    root->check_tree( FEATURE_GTRANSFORMS ); 
    root->check_tree( FEATURE_PARENT_LINKS | FEATURE_GTRANSFORMS ); 
}





void nnode::dump(const char* msg) const 
{
    if(msg)
    { 
       LOG(info) << msg ; 
    } 
    _dump->dump();
}

void nnode::dump_prim() const 
{
    _dump->dump_prim();
}
void nnode::dump_gtransform() const 
{
    _dump->dump_gtransform();
}
void nnode::dump_transform() const 
{
    _dump->dump_transform();
}
void nnode::dump_planes() const 
{
    _dump->dump_planes();
}



/*
// move up to NCSG now that are operating from root..

unsigned nnode::uncoincide(unsigned verbosity)
{
    NNodeUncoincide unco(this, verbosity);
    return unco.uncoincide();
}
*/



/**
nnode::getCoincident
---------------------

Checking the disposition of parametric points of parametric surfaces of this node 
with respect to the implicit distance to the surface of the other node...  

Parametric uv coordinates of this node are collected into *coincident* 
when their positions are within epsilon of the surface of the other node.

The frame of this nodes parametric positions and the frame of the other nodes 
signed distance to surface are specified by the fr arguement, typically FRAME_LOCAL
is appropriate.

PROBLEM IS THIS WILL ONLY WORK AT BILEAF LEVEL JUST ABOVE PRIMITIVES

**/

void nnode::getCoincident(std::vector<nuv>& coincident, const nnode* other_, float epsilon, unsigned level, int margin, NNodeFrameType fr) const 
{
     unsigned ns = par_nsurf();
     for(unsigned s = 0 ; s < ns ; s++)
     {    
         getCoincidentSurfacePoints(coincident, s, level, margin, other_, epsilon, fr) ;
     }
}

void nnode::getCoincidentSurfacePoints(std::vector<nuv>& surf, int s, unsigned level, int margin, const nnode* other_, float epsilon, NNodeFrameType fr) const 
{
    assert( level < 16 );  
    int nu = 1 << level ; 
    int nv = 1 << level ; 
    assert( (nv - margin) > 0 );
    assert( (nu - margin) > 0 );

    //  level = 1, margin = 0   -> grid of 9 points on the surface
    //        nu,nv = 2,2     v = 0,1,2   u = 0,1,2      
    //
    //  level = 1, margin = 1   -> single central point on the surface
    //        nu,nv = 2,2     v = 1   u = 1      
    //

    unsigned p = 0 ; 

    for (int v = margin; v <= (nv-margin)  ; v++)
    {
        for (int u = margin; u <= (nu-margin) ; u++) 
        {
            nuv uv = make_uv(s,u,v,nu,nv, p);

            glm::vec3 pos = par_pos_(uv, fr);

            float other_sdf = other_->sdf_(pos, fr );

            if( fabs(other_sdf) < epsilon )  
            {
                surf.push_back(uv) ;
            }
        }
    }
}





const std::vector<glm::vec3>& nnode::get_par_points() const 
{
    return par_points ; 
}
const std::vector<nuv>& nnode::get_par_coords() const 
{
    return par_coords ; 
}




glm::vec3 nnode::par_pos_model(const nuv& ) const 
{
    assert(0 && "nnode::par_pos_model needs override in all primitives ");
    return glm::vec3(0);
}

void nnode::par_posnrm_model(glm::vec3& , glm::vec3& , unsigned, float, float ) const 
{
    assert(0 && "nnode::par_posnrm_model needs override in all primitives ");
}

/**
nnode::selectSheets
---------------------

Converts a mask integer into a vector of sheet indices.
For a box with 6 sheets, the indices will be from 0,1,2,3,4,5

**/

void nnode::selectSheets( std::vector<unsigned>& sheets, unsigned sheetmask ) const 
{
    unsigned nsa = par_nsurf();

    if( sheetmask == 0 ) // all sheets
    {
        for(unsigned sheet = 0 ; sheet < nsa ; sheet++) sheets.push_back(sheet) ;
    }
    else
    {
        for(unsigned sheet = 0 ; sheet < nsa ; sheet++) if((sheetmask & (0x1 << sheet)) != 0  ) sheets.push_back(sheet) ;
    } 

    LOG(error) 
        << " nsa " << nsa
        << " sheetmask " << std::hex << sheetmask << std::dec
        << " ns " << sheets.size() 
        ;
}


void nnode::generateParPoints(unsigned seed, const glm::vec4& uvdom, std::vector<glm::vec3>& points, std::vector<glm::vec3>& normals, unsigned num_total, unsigned sheetmask ) const  
{

    LOG(error) << " sheetmask " << std::hex << sheetmask ; 

    std::vector<unsigned> sheets ; 
    selectSheets(sheets, sheetmask);
    unsigned ns = sheets.size() ; 

    unsigned cumsum = 0 ; 
    BRng ugen(uvdom.x,uvdom.y, seed,   "U") ;  
    BRng vgen(uvdom.z,uvdom.w, seed+1, "V") ;  

    for(unsigned i = 0 ; i < ns  ; i++) 
    {
        unsigned num = i == ns - 1 ? num_total - cumsum : num_total/ns  ;  // divided total or remainder for last  
        unsigned sheet = sheets[i] ;    // <-- fixed buglet, was conflating the "i" with the "sheet" 
        generateParPointsSheet(points, normals, ugen, vgen,  sheet, num ) ;
        cumsum += num ; 
    }
}
void nnode::generateParPointsSheet(std::vector<glm::vec3>& points, std::vector<glm::vec3>& normals, BRng& ugen, BRng& vgen,  unsigned sheet, unsigned num ) const 
{

    glm::vec3 pos ; 
    glm::vec3 nrm ; 

    LOG(info) 
        << " sheet " << sheet 
        << " num " << num
        ;

    bool dump = false ; 

    for(unsigned i=0 ; i < num ; i++)
    {
        float fu = ugen.one() ;
        float fv = vgen.one() ;

        if(dump)
        std::cout 
             << " i " << std::setw(10) << i 
             << " fu " << std::setw(10) << fu  
             << " fv " << std::setw(10) << fv
             << std::endl 
             ;  

        par_posnrm_model( pos, nrm, sheet, fu, fv );        

        points.push_back(pos);
        normals.push_back(nrm);
    }
}





void nnode::collectParPoints(unsigned prim_idx, unsigned level, int margin, NNodeFrameType fr, unsigned verbosity_) 
{
    assert(is_primitive());
    par_points.clear();
    par_coords.clear();

    unsigned ns = par_nsurf();
    for(unsigned sheet = 0 ; sheet < ns ; sheet++) collectParPointsSheet(prim_idx, sheet, level, margin, fr, verbosity_) ;
}

void nnode::collectParPointsSheet(unsigned prim_idx, int sheet, unsigned level, int margin, NNodeFrameType fr, unsigned verbosity_) 
{
    /*

     prim_idx 
          gets persisted into the coordinates, to enable untangling which primitive easily 

     level 1  
          (1 << 1) = 2 divisions in u and v,  +1 for endpost -> 3 points in u and v 

     margin 1
          skip the start and end of uv range 
          3 points - 2*margin ->   1 point   at mid uv of the face 

          +---+---+
          |   |   |
          +---*---+
          |   |   |
          +---+---+
    */
    assert(is_primitive());
    int nu = 1 << level ; 
    int nv = 1 << level ; 
    assert( (nv - margin) > 0 );
    assert( (nu - margin) > 0 );

    int ndiv = nu + 1 - 2*margin ;
    unsigned expect = ndiv*ndiv  ;  

    unsigned n0 = par_points.size();


    //std::vector<const glm::vec3> uniq ; 

    for (int v = margin; v <= (nv-margin) ; v++)
    {
        for (int u = margin; u <= (nu-margin) ; u++) 
        {
            nuv uv = make_uv(sheet,u,v,nu,nv, prim_idx);
            glm::vec3 pos = par_pos_(uv, fr );

            par_points.push_back(pos) ;
            par_coords.push_back(uv) ;

            // tried std::set it didnt work
            //if(std::find(uniq.begin(),uniq.end(), pos) == uniq.end())  uniq.push_back(pos);
        }
    }
    unsigned n1 = par_points.size();
    unsigned n = n1 - n0 ; 
    assert( n == expect );


    if(verbosity_ > 5)
    {
        std::cout
                 << "nnode::collectParPointsSheet"
                 << " verbosity " << std::setw(3) << verbosity_
                 << " prim " << std::setw(3) << prim_idx 
                 << " sheet " << std::setw(3) << sheet 
                 << " nu " << std::setw(4) << nu 
                 << " nv " << std::setw(4) << nv
                 << " ndiv " << std::setw(5) << ndiv
                 << " expect " << std::setw(6) << expect
                 << " n0 " << std::setw(6) << n0
                 << " n1 " << std::setw(6) << n1
                 << " n " << std::setw(6) << n
            //     << " uniq " << std::setw(6) << uniq.size()
            //     << " uniq[0] " << ( uniq.size() > 0 ? gpresent(uniq[0]) : "" )
                 << std::endl 
                 ;
    } 
}


void nnode::dumpSurfacePointsAll(const char* msg, NNodeFrameType fr) const 
{
    LOG(info) << msg << " nnode::dumpSurfacePointsAll " ; 
    std::cout 
              << NNodeEnum::FrameType(fr)
              << " verbosity " << verbosity     
              << std::endl ;     

/*
    std::vector<glm::vec3> points ; 
    getParPoints( points, prim_idx, level, margin, fr, verbosity ); 

    float epsilon = 1e-5f ; 
    dumpPointsSDF(points, epsilon );
*/
}





/*
void nnode::dumpPointsSDF(const std::vector<glm::vec3>& points, float epsilon) const 
{
    LOG(info) << "nnode::dumpPointsSDF"
              << " points " << points.size()
              ;

    unsigned num_inside(0);
    unsigned num_outside(0);
    unsigned num_surface(0);

    std::function<float(float,float,float)> _sdf = sdf() ;

    for(unsigned i=0 ; i < points.size() ; i++) 
    {
          glm::vec3 p = points[i] ;

          float sd =  _sdf(p.x, p.y, p.z) ;

          if(fabsf(sd) < epsilon ) num_surface++ ; 
          else if( sd  <  0 )      num_inside++ ; 
          else if( sd  >  0 )      num_outside++ ; 
           
          std::cout
               << " i " << std::setw(4) << i 
               << " p " << gpresent(p) 
               << " sd(fx4) " << std::setw(10) << std::fixed << std::setprecision(4) << sd 
               << " sd(sci) " << std::setw(10) << std::scientific << sd 
               << " sd(def) " << std::setw(10) << std::defaultfloat  << sd 
               << std::endl
               ; 
    }

    LOG(info) << "nnode::dumpPointsSDF"
              << " points " << std::setw(6) << points.size()
              << " epsilon " << std::scientific << epsilon 
              << " num_inside " << std::setw(6) << num_inside
              << " num_surface " << std::setw(6) << num_surface
              << " num_outside " << std::setw(6) << num_outside
              ;

}
*/



/**
nnode::is_ellipsoid
---------------------

A CSG_SPHERE or CSG_ZSPHERE with non-identity scale is an ellipsoid.

**/

bool nnode::is_ellipsoid(bool verbose) const 
{
    if(!(type == CSG_SPHERE || type == CSG_ZSPHERE ))  return false ;
    if(transform == NULL) return false ; 

   
    ndeco d ;
    nglmext::polar_decomposition( transform->t, d );

    glm::vec3 dsca = nglmext::pluck_scale( d ); 

    float epsilon = 1e-3 ; 

    bool has_scale = nglmext::has_scale( dsca, epsilon ); 

    if(verbose)
    { 
        LOG(info) ; 
        std::cout << gpresent( "dsca", dsca ) << std::endl ;
        std::cout << " has_scale " << has_scale << std::endl ; 
    }
    return has_scale ; 
}


/**
nnode::reconstruct_ellipsoid
------------------------------

G4Ellipsoid gets translated into either an nzsphere or an nsphere 
(depending on there being zcuts) with a scaling transform associated

**/

void nnode::reconstruct_ellipsoid( glm::vec3& axes, glm::vec2& zcut, glm::mat4& trs_unscaled ) const 
{
    const nnode* n = this ; 
 
    const nsphere* sp = dynamic_cast<const nsphere*>(n) ;   
    const nzsphere* zs = dynamic_cast<const nzsphere*>(n) ;   

    bool is_sphere = sp != NULL ; 
    bool is_zsphere = zs != NULL ; 
    assert( is_sphere ^ is_zsphere ); 
    float radius = is_sphere ? sp->radius() : zs->radius() ; 

    const nnode* e = sp ? (nnode*)sp : (nnode*)zs ; 

    const nmat4triple* txf = e->transform ; 
    assert( txf );

    //LOG(info) << " txf " << txf ; 
    //print(txf->t , "t" ) ; 

    ndeco d ;
    nglmext::polar_decomposition( txf->t, d ); 
    //print(d.t , "t" ) ; 
    //print(d.r , "r" ) ; 
    //print(d.s , "s" ) ; 

    float sx = d.s[0][0] ; 
    float sy = d.s[1][1] ; 
    float sz = d.s[2][2] ;

    LOG(debug) 
         << " sx " << sx  
         << " sy " << sy  
         << " sz " << sz 
         << " radius " << radius 
         ;

    axes.x = sx*radius ; 
    axes.y = sy*radius ; 
    axes.z = sz*radius ; 

    zcut.x = is_sphere ? -axes.z : zs->z1() ;  
    zcut.y = is_sphere ?  axes.z : zs->z2() ; 

    trs_unscaled = d.tr ; 

    //print(axes, "axes" ); 
    //print(zcut, "zcut" ); 

}




