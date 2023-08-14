
#include <iostream>
#include <algorithm>

#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "spa.h"
#include "sbb.h"
#include "sxf.h"
#include "scanvas.h"

#include "OpticksCSG.h"
#include "scsg.hh"
#include "snd.hh"
#include "sndtree.h"  // using flexible sn.h 
#include "st.h"       // only st::complete_binary_tree_nodes
#include "stra.h"     // transform utilities based on glm  
#include "stv.h"

sn::POOL sn::pool = {} ; 
stv::POOL stv::pool = {} ; 


scsg* snd::POOL = nullptr  ; 
void snd::SetPOOL( scsg* pool ){ POOL = pool ; }  // static 
int snd::Level(){ return POOL ? POOL->level : -1 ; } // static

NPFold* snd::Serialize(){ return POOL ? POOL->serialize() : nullptr ; }  // static 
void    snd::Import(const NPFold* fold){ assert(POOL) ; POOL->import(fold) ; } // static 
std::string snd::Desc(){  return POOL ? POOL->desc() : "? NO POOL ?" ; } // static 



std::string snd::Brief(int idx) // static
{
    const snd* nd = Get(idx); 
    return nd ? nd->brief() : "-" ; 
}
std::string snd::Brief(const std::vector<int>& nodes) // static 
{
    int num_nodes = nodes.size();  
    std::stringstream ss ; 
    ss << "snd::Brief num_nodes " << num_nodes << std::endl ; 
    for(int i=0 ; i < num_nodes ; i++)
    {
        int idx = nodes[i]; 
        const snd* nd = Get(idx); 
        assert( nd ); 
        ss << std::setw(2) << i << " : " << nd->brief() << std::endl ;  
    }
    std::string str = ss.str(); 
    return str ; 
}

std::string snd::Brief_(const std::vector<snd>& nodes) // static 
{
    int num_nodes = nodes.size();  
    std::stringstream ss ; 
    ss << "snd::Brief_ num_nodes " << num_nodes << std::endl ; 
    for(int i=0 ; i < num_nodes ; i++)
    {
        const snd& nd = nodes[i] ; 
        ss << std::setw(2) << i << " : " << nd.brief() << std::endl ;  
    }
    std::string str = ss.str(); 
    return str ; 
}

const snd* snd::Get(int idx){ return POOL ? POOL->getND(idx) : nullptr ; } // static
      snd* snd::Get_(int idx){ return POOL ? POOL->getND_(idx) : nullptr ; } // static


int snd::GetMaxDepth( int idx)
{ 
    const snd* nd = Get(idx) ; 
    return nd ? nd->max_depth() : -1 ; 
}
int snd::GetNumNode( int idx)
{ 
    const snd* nd = Get(idx) ; 
    return nd ? nd->num_node() : -1 ; 
}

void snd::GetTypes(std::vector<int>& types, const std::vector<int>& idxs ) // static
{
    int num_idx = idxs.size(); 
    for(int i=0 ; i < num_idx ; i++)
    {
        int idx = idxs[i]; 
        const snd* nd = Get(idx) ; 
        types.push_back(nd->typecode) ;  
    }
    assert( idxs.size() == types.size() ); 
}


/**
snd::GetNodeXForm : xform "pointer" for a node
----------------------------------------------

Access the idx *snd*  and return the xform *idx*

This is used from U4Solid::init_BooleanSolid
to check that xforms are associated to the left, right 
nodes. 

*/

int snd::GetNodeXForm(int idx)   // static 
{ 
    const snd* n = Get(idx); 
    return n ? n->xform : -1 ; 
}


/**
snd::SetNodeXForm
------------------

Canonical usage is from U4Solid::init_DisplacedSolid collecting boolean rhs transforms 

**/

void snd::SetNodeXForm(int idx, const glm::tmat4x4<double>& t )
{
    snd* nd = Get_(idx); 
    nd->combineXF(t); 
}
void snd::SetNodeXForm(int idx, const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v )
{
    snd* nd = Get_(idx); 
    nd->combineXF(t, v); 
}




/**
snd::setXF
---------------

Calling setXF again will just adds another transform 
and updates the snd::xform integer reference, effectively leaking the old transform. 

HMM if there is a transform already present (eg ellipsoid scale transform)
the setXF actually needs to combine the transforms as was done in nnode::set_transform
so that with snd::combineXF


**/

void snd::setXF(const glm::tmat4x4<double>& t )
{
    glm::tmat4x4<double> v = glm::inverse(t) ; 
    setXF(t, v); 
}
void snd::combineXF(const glm::tmat4x4<double>& t )
{
    glm::tmat4x4<double> v = glm::inverse(t) ; 
    combineXF(t, v); 
}


void snd::setXF(const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v )
{
    if( xform > -1 )
    {
        if(Level()>-1) std::cout 
            << "snd::setXF STOMPING"
            << " xform " << xform 
            << std::endl 
            << stra<double>::Desc(t,v, "t", "v")
            << std::endl 
            ; 
    }


    sxf xf ; 
    xf.t = t ; 
    xf.v = v ; 

    CheckPOOL("snd::setXForm") ; 
    xform = POOL->addXF(xf) ; 
}

/**
snd::combineXF
----------------

Transform product ordering is an ad-hoc guess::

    tt = current->t * t
    vv = v * current->v 
     
**/

void snd::combineXF( const glm::tmat4x4<double>& t, const glm::tmat4x4<double>& v )
{
    sxf* current = getXF_(); 
    if(current == nullptr)
    {
        setXF(t,v); 
    }
    else
    {
        glm::tmat4x4<double> tt = current->t * t ;   
        glm::tmat4x4<double> vv = v * current->v ;   

        current->t = tt ; 
        current->v = vv ; 
    }
}


const sxf* snd::GetXF(int idx)  // static
{
    const snd* n = Get(idx); 
    return n ? n->getXF() : nullptr ; 
}
sxf* snd::GetXF_(int idx)  // static
{
    snd* n = Get_(idx); 
    return n ? n->getXF_() : nullptr ; 
}

const sxf* snd::getXF() const
{
    CheckPOOL("snd::getXF") ; 
    return POOL->getXF(xform) ; 
}
sxf* snd::getXF_()
{
    CheckPOOL("snd::getXF_") ; 
    return POOL->getXF_(xform) ; 
}

/**
snd::NodeTransformProduct
---------------------------

cf nmat4triple::product

**/

void snd::NodeTransformProduct(int root, glm::tmat4x4<double>& t, glm::tmat4x4<double>& v, bool reverse, std::ostream* out)  // static
{
    std::vector<int> nds ; 
    Ancestors(root, nds);  
    nds.push_back(root); 
    int num_nds = nds.size();

    if(out)
    {
        *out 
             << std::endl 
             << "snd::NodeTransformProduct" 
             << " root " << root 
             << " reverse " << reverse
             << " num_nds " << num_nds 
             << std::endl 
             ;
    }

    glm::tmat4x4<double> tp(1.); 
    glm::tmat4x4<double> vp(1.); 

    for(int i=0 ; i < num_nds ; i++ ) 
    {
        int j  = num_nds - 1 - i ;  
        int ii = nds[reverse ? j : i] ; 
        int jj = nds[reverse ? i : j] ; 

        const sxf* ixf = GetXF(ii) ; 
        const sxf* jxf = GetXF(jj) ; 

        if(out)
        {
            *out
                << " i " << i 
                << " j " << j 
                << " ii " << ii 
                << " jj " << jj
                << " ixf " << ( ixf ? "Y" : "N" ) 
                << " jxf " << ( jxf ? "Y" : "N" ) 
                << std::endl 
                ; 

           if(ixf) *out << stra<double>::Desc( ixf->t, ixf->v, "(ixf.t)", "(ixf.v)" ) << std::endl ;   
           if(jxf) *out << stra<double>::Desc( jxf->t, jxf->v, "(jxf.t)", "(jxf.v)" ) << std::endl ;   
        }


        if(ixf) tp *= ixf->t ; 
        if(jxf) vp *= jxf->v ;  // // inverse-transform product in opposite order
    }
    memcpy( glm::value_ptr(t), glm::value_ptr(tp), sizeof(glm::tmat4x4<double>) );
    memcpy( glm::value_ptr(v), glm::value_ptr(vp), sizeof(glm::tmat4x4<double>) );

    if(out) *out << stra<double>::Desc( tp, vp , "tp", "vp" ) << std::endl ;
}

std::string snd::DescNodeTransformProduct(int root, glm::tmat4x4<double>& t, glm::tmat4x4<double>& v,  bool reverse) // static
{
    std::stringstream ss ; 
    ss << "snd::DescNodeTransformProduct" << std::endl ;
    NodeTransformProduct( root, t, v, reverse, &ss );     
    std::string str = ss.str(); 
    return str ; 
}


void snd::node_transform_product(glm::tmat4x4<double>& t, glm::tmat4x4<double>& v,  bool reverse, std::ostream* out ) const 
{
    NodeTransformProduct(index, t, v, reverse, out); 
}






void snd::SetLabel(int idx, const char* label_ ) // static
{
    snd* nd = Get_(idx); 
    nd->setLabel(label_); 
}




/**
snd::SetLVID : label node tree
----------------------------------

This gets invoked from the snd roots only, 
as its called for each root from U4Tree::initSolid

**/
void snd::SetLVID(int idx, int lvid)  // static
{
    snd* nd = Get_(idx); 
    nd->setLVID(lvid);   

    int chk = nd->checktree(); 
    if( chk != 0 )
    { 
        if(Level() > 0 ) std::cerr 
           << "snd::SetLVID" 
           << " idx " << idx 
           << " lvid " << lvid 
           << " checktree " << chk 
           << std::endl 
           << " POOL.desc " 
           << POOL->desc() 
           ; 
    }
    assert( chk == 0 ); 
}


/**
snd::GetLVID (GetLVIDNodes more appropos)
-------------------------------------------------

Q: Is the last snd returned always root ? 
A: As trees are created postorder and nodes get added to the POOL 
   in creation order I think that indeed the last must always be root. 

**/

void snd::GetLVID( std::vector<snd>& nds, int lvid )  // static
{ 
    POOL->getLVID(nds, lvid); 

    int num_nd = nds.size(); 
    assert( num_nd > 0 ); 
    const snd& last = nds[num_nd-1] ; 
    assert( last.is_root() ); 
}

/**
snd::GetLVRoot
---------------

Returns pointer to first snd in the (scsg)POOL with the lvid provided.

**/
const snd* snd::GetLVRoot( int lvid ) // static
{
    const snd* root = POOL->getLVRoot(lvid); 
    return root ; 
}

int snd::GetLVNumNode( int lvid ) // static
{
    const snd* root = GetLVRoot(lvid); 
    return root ? root->getLVNumNode() : -1 ; 
}
int snd::GetLVBinNode( int lvid ) // static    NumBinNode
{
    const snd* root = GetLVRoot(lvid); 
    return root ? root->getLVBinNode() : -1 ; 
}
int snd::GetLVSubNode( int lvid ) // static    NumSubNode
{
    const snd* root = GetLVRoot(lvid); 
    return root ? root->getLVSubNode() : -1 ; 
}




/**
snd::getLVNumNode
-------------------

Returns total number of nodes that can contain 
a complete binary tree + listnode constituents
serialization of this node.  

**/

int snd::getLVNumNode() const 
{
    int bn = getLVBinNode() ; 
    int sn = getLVSubNode() ; 
    return bn + sn ; 
}

/**
snd::getLVBinNode
------------------

Returns the number of nodes in a complete binary tree
of height corresponding to the max_binary_depth 
of this node. 
**/

int snd::getLVBinNode() const 
{
    int h = max_binary_depth(); 
    return st::complete_binary_tree_nodes( h );  
}

/**
snd::getLVSubNode
-------------------

Sum of children of compound nodes found beneath this node. 
HMM: this assumes compound nodes only contain leaf nodes 

**/
int snd::getLVSubNode() const 
{
    int constituents = 0 ; 
    std::vector<int> subs ; 
    typenodes_(subs, CSG_CONTIGUOUS, CSG_DISCONTIGUOUS, CSG_OVERLAP );  
    int nsub = subs.size(); 
    for(int i=0 ; i < nsub ; i++)
    {
        int idx = subs[i] ; 
        const snd* nd = Get(idx); 
        assert( nd->typecode == CSG_CONTIGUOUS || nd->typecode == CSG_DISCONTIGUOUS ); 
        constituents += nd->num_child ; 
    } 
    return constituents ; 
}



/**
snd::GetLVNodesComplete
-------------------------

As the traversal is constrained to the binary tree portion of the n-ary snd tree 
can populate a vector of *snd* pointers in complete binary tree level order indexing
with nullptr left for the zeros.  This is similar to the old NCSG::export_tree_r.

**/

void snd::GetLVNodesComplete(std::vector<const snd*>& nds, int lvid) // static 
{
    const snd* root = GetLVRoot(lvid);  // first snd in (scsg)POOL
    root->getLVNodesComplete(nds);    

    int level = Level(); 

    if(level > 0 && nds.size() > 8 )
    {
        std::cout
            << "snd::GetLVNodesComplete"
            << " lvid " << lvid
            << " level " << level
            << std::endl
            << root->rbrief()
            << std::endl
            << root->render(3)
            ;
    }
}


/**
snd::getLVNodesComplete
-------------------------

**/

void snd::getLVNodesComplete(std::vector<const snd*>& nds) const 
{
    int bn = getLVBinNode();  
    int sn = getLVSubNode();  
    int numParts = bn + sn ; 
    nds.resize(numParts); 

    assert( sn == 0 ); // CHECKING : AS IMPL LOOKS LIKE ONLY HANDLES BINARY NODES

    GetLVNodesComplete_r( nds, this, 0 ); 
}


void snd::GetLVNodesComplete_r(std::vector<const snd*>& nds, const snd* nd, int idx)  // static
{
    assert( idx < int(nds.size()) ); 
    nds[idx] = nd ; 

    if( nd->num_child > 0 && nd->is_listnode() == false ) // non-list operator node
    {
        assert( nd->num_child == 2 ) ;
        int ch = nd->first_child ;
        for(int i=0 ; i < nd->num_child ; i++)
        {
            const snd* child = snd::Get(ch) ;
            assert( child->index == ch );

            int cidx = 2*idx + 1 + i ; // 0-based complete binary tree level order indexing 

            GetLVNodesComplete_r(nds, child, cidx );

            ch = child->next_sibling ;
        }
    }
}




std::string snd::Desc(      int idx){ return POOL ? POOL->descND(idx) : "-" ; } // static
std::string snd::DescParam( int idx){ return POOL ? POOL->descPA(idx) : "-" ; } // static
std::string snd::DescXForm( int idx){ return POOL ? POOL->descXF(idx) : "-" ; } // static
std::string snd::DescAABB(  int idx){ return POOL ? POOL->descBB(idx) : "-" ; } // static


int snd::Add(const snd& nd) // static
{
    assert( POOL && "snd::Add MUST SET snd::SetPOOL to scsg instance first" ); 
    return POOL->addND(nd); 
}


bool snd::is_listnode() const 
{
    return CSG::IsList(typecode); 
}
std::string snd::tag() const
{
    return typecode < 0 ? "negative-typecode-ERR" : CSG::Tag(typecode) ; 
}




std::string snd::brief() const 
{
    char l0 = label[0] == '\0' ? '-' : label[0] ; 
    int w = 5 ; 
    std::stringstream ss ; 
    ss
       << l0  
       << " ix:" << std::setw(w) << index
       << " dp:" << std::setw(w) << depth
       << " sx:" << std::setw(w) << sibdex
       << " pt:" << std::setw(w) << parent
       << "    "
       << " nc:" << std::setw(w) << num_child 
       << " fc:" << std::setw(w) << first_child
       << " ns:" << std::setw(w) << next_sibling
       << " lv:" << std::setw(w) << lvid
       << "    "
       << " tc:" << std::setw(w) << typecode 
       << " pa:" << std::setw(w) << param 
       << " bb:" << std::setw(w) << aabb
       << " xf:" << std::setw(w) << xform
       << "    "
       << tag()
       ; 
    std::string str = ss.str(); 
    return str ; 
}

std::string snd::rbrief() const 
{
    std::stringstream ss ; 
    ss << "snd::rbrief" << std::endl ; 

    rbrief_r(ss, 0) ; 
    std::string str = ss.str(); 
    return str ; 
}


void snd::rbrief_r(std::ostream& os, int d) const 
{
    os << brief() << std::endl ; 
    int ch = first_child ; 
    while( ch > -1 )
    {
        snd& child = POOL->node[ch] ; 
        child.rbrief_r(os, d+1) ; 
        ch = child.next_sibling ;
    }
}






const char* snd::ERROR_NO_POOL_NOTES = R"(
snd::ERROR_NO_POOL_NOTES
======================================

The POOL static scsg instance is nullptr

* MUST call snd::SetPOOL to an scsg instance before using snd 

* stree instanciation will do this, so the preferred approach 
  is to instanciate stree in the main prior to using any snd methods. 

::

    #include "stree.h"
    #include "snd.hh"

    int main(int argc, char** argv)
    {
        stree st ; 
        int a = snd::Sphere(100.) ; 
        return 0 ; 
    }


)" ; 

void snd::CheckPOOL(const char* msg) // static 
{
    if(POOL) return ; 
    std::cout << "snd::CheckPOOL " << msg << " FATAL " << std::endl ; 
    std::cout <<  ERROR_NO_POOL_NOTES  ; 
    assert( POOL ); 
}


void snd::setParam( double x, double y, double z, double w, double z1, double z2 )
{
    CheckPOOL("snd::setParam") ; 
    spa o = { x, y, z, w, z1, z2 } ; 
    param = POOL->addPA(o) ; 
}
void snd::setAABB( double x0, double y0, double z0, double x1, double y1, double z1 )
{
    CheckPOOL("snd::setAABB") ; 
    sbb o = {x0, y0, z0, x1, y1, z1} ; 

    aabb = POOL->addBB(o) ; 
}

const double* snd::getParam() const 
{
    if(param == -1 ) return nullptr ; 
    assert( param > -1 ); 
    const spa& pa = POOL->param[param] ; 
    return pa.data() ; 
}
const double* snd::getAABB() const 
{
    if(aabb == -1 ) return nullptr ; 
    assert( aabb > -1 );  
    const sbb& bb = POOL->aabb[aabb] ; 
    return bb.data() ; 
}

bool snd::hasUnsetAABB() const   // nullptr or all zero
{
    const double* aabb = getAABB();  
    if(aabb == nullptr) return true ; 
    return sbb::IsZero(aabb); 
}
bool snd::hasAABB() const   // not-nullptr and not all zero 
{
    const double* aabb = getAABB();  
    return aabb != nullptr && !sbb::IsZero(aabb) ; 
}





void snd::setLabel( const char* label_ )
{
    strncpy( &label[0], label_, sizeof(label) );
}

void snd::setLVID(int lvid_)
{
    setLVID_r(lvid_, 0); 
}
void snd::setLVID_r(int lvid_, int d )
{
    lvid = lvid_ ;  
    depth = d ;     

    int ch = first_child ; 
    while( ch > -1 )
    {
        snd& child = POOL->node[ch] ; 
        child.setLVID_r(lvid, depth+1 ); 
        ch = child.next_sibling ;
    }
}

int snd::checktree() const 
{
    int chk_D = checktree_r('D', 0); 
    int chk_P = checktree_r('P', 0); 
    int chk = chk_D + chk_P ; 

    if( chk > 0 ) 
    {
        if(Level()>0) std::cerr 
            << "snd::checktree"
            << " chk_D " << chk_D
            << " chk_P " << chk_P
            << brief()
            << std::endl
            ;
    }

    return chk ; 
}
int snd::checktree_r(char code,  int d ) const 
{
    int chk = 0 ; 
    int ch = first_child ; 

    if( code == 'D' ) // check expected depth
    {
        if(d != depth) chk += 1 ; 
    }
    else if( code == 'P' ) // check for non-roots without parent set 
    {
        if( depth > 0 && parent < 0 ) chk += 1 ; 
    }


    while( ch > -1 )
    {
        snd& child = POOL->node[ch] ; 

        chk += child.checktree_r(code,  d + 1 );  

        ch = child.next_sibling ;
    }
    return chk ; 
}

 
/**
snd::Visit
-----------

Example static function that can be passed to the traverse using::

    snd::PreorderTraverse(root, std::bind( &snd::Visit, std::placeholders::_1 ) );   

Note that the fn above is a static member function.

Although it is possible to bind to a non-static member function using "this" 
object pointer as first argument that would not be useful with a recursive traversal 
as the point of the traversal is to visit multiple nodes of the tree.  Using a static 
with int argument which picks the node sidesteps this. 

**/
void snd::Visit(int idx)  // static 
{
    snd* nd = Get_(idx); 

    std::cout 
        << "snd::Visit" 
        << " idx " << std::setw(3) << idx 
        << " : " << nd->brief() 
        << std::endl 
        ; 
}


void snd::PreorderTraverse(int idx, std::function<void(int)> fn) // static 
{
    snd* nd = Get_(idx); 
    nd->preorder_traverse( fn );  
}

void snd::preorder_traverse( std::function<void(int)> fn ) 
{
    preorder_traverse_r(fn, 0); 
}
void snd::preorder_traverse_r( std::function<void(int)> fn, int d) 
{
    fn(index); 

    int ch = first_child ; 
    while( ch > -1 )
    {
        snd* child = Get_(ch) ; 
        child->preorder_traverse_r(fn,  d + 1 );  
        ch = child->next_sibling ;
    }
}


void snd::PostorderTraverse(int idx, std::function<void(int)> fn ) // static
{
    snd* nd = Get_(idx); 
    nd->postorder_traverse( fn );  
}
void snd::postorder_traverse( std::function<void(int)> fn ) 
{
    postorder_traverse_r(fn, 0); 
}
void snd::postorder_traverse_r( std::function<void(int)> fn, int d) 
{
    int ch = first_child ; 
    while( ch > -1 )
    {
        snd* child = Get_(ch) ; 
        child->postorder_traverse_r(fn,  d + 1 );  
        ch = child->next_sibling ;
    }

    fn(index); 

}








/**
snd::max_depth
---------------

Q: How to handle compound nodes CSG_CONTIGUOUS/CSG_DISCONTIGUOUS with regard to depth ?
A: Kinda depends on the purpose of the depth ? Need separate methods max_treedepth ?
A: As compound nodes should only hold leaves the default way to get 
   depth should usually be correct. But there are special cases where it 
   might not be. For example where the root node is compound the max depth
   should be 0. Suggests should stop traversal when hit compound ? 

**/

int snd::max_depth() const 
{
    return max_depth_r(0);
}
int snd::max_depth_r(int d) const   
{
    int mx = d ; 
    int ch = first_child ; 
    while( ch > -1 )
    {
        snd& child = POOL->node[ch] ; 
        mx = std::max( mx,  child.max_depth_r(d + 1) ) ; 
        ch = child.next_sibling ;
    }
    return mx ; 
}

/**
snd::max_binary_depth
-----------------------

Maximum depth of the binary compliant portion of the n-ary tree, 
ie with listnodes not recursed and where nodes have either 0 or 2 children.  
The listnodes are regarded as leaf node primitives.  

* Despite the *snd* tree being an n-ary tree (able to hold polycone and multiunion compounds)
  it must be traversed as a binary tree by regarding the compound nodes as effectively 
  leaf node "primitives" in order to generate the indices into the complete binary 
  tree serialization in level order 

* hence the recursion is halted at list nodes

**/
int snd::max_binary_depth() const 
{
    return max_binary_depth_r(0) ; 
}
int snd::max_binary_depth_r(int d) const   
{
    int mx = d ; 

    if( is_listnode() == false )
    {
        if( num_child > 0 ) assert( num_child == 2 ) ; 
        int ch = first_child ; 
        for(int i=0 ; i < num_child ; i++)  
        {
            snd& child = POOL->node[ch] ; 
            assert( child.index == ch ); 
            mx = std::max( mx,  child.max_binary_depth_r(d + 1) ) ; 
            ch = child.next_sibling ;
        }
    }
    return mx ; 
}


int snd::num_node() const 
{
    return num_node_r(0);
}
int snd::num_node_r(int d) const   
{
    int nn = 1 ;   // always at least 1 node,  HMM: no exclusion of CSG_ZERO ? 
    int ch = first_child ; 
    while( ch > -1 )
    {
        snd& child = POOL->node[ch] ; 
        nn += child.num_node_r(d + 1); 
        ch = child.next_sibling ;
    }
    return nn ; 
}


bool snd::is_root() const  // NB depth gets set by calling setLVID
{
    return depth == 0 && parent == -1 ; 
}

bool snd::is_leaf() const
{
    return num_child == 0 ;
}

bool snd::is_binary_leaf() const 
{
    return num_child == 0 || CSG::IsList(typecode ) ; 
}

bool snd::is_sibdex(int q_sibdex) const 
{
    return sibdex == q_sibdex ; 
}






/**
inorder
------------

Try to adapt X4SolidTree::inorder_r to n-ary tree 

For inorder traversal of an n-ary tree need to define how to split the children. 

binary:  i = 0, 1,     num_child = 2  split = 1 = num_child - 1 
3-ary :  i = 0, 1, 2   num_child = 3  split = 2 = num_child - 1 

:google:`inorder traveral of n-ary tree`

https://www.geeksforgeeks.org/inorder-traversal-of-an-n-ary-tree/

The inorder traversal of an N-ary tree is defined as visiting all the children
except the last then the root and finally the last child recursively. 

https://ondrej-kvasnovsky-2.gitbook.io/algorithms/data-structures/n-ary-tree

A binary tree can be traversed in preorder, inorder, postorder or level-order.
Among these traversal methods, preorder, postorder and level-order traversal
are suitable to be extended to an N-ary tree.

https://stackoverflow.com/questions/23778489/in-order-tree-traversal-for-non-binary-trees

**/


void snd::Inorder(std::vector<int>& order, int idx ) // static
{
    const snd* nd = Get(idx); 
    nd->inorder(order); 
}

void snd::inorder(std::vector<int>& order ) const 
{
    inorder_r(order, 0); 
}

void snd::inorder_r(std::vector<int>& order, int d ) const 
{
    if( num_child <= 0 )
    {
        order.push_back(index) ; 
    }
    else
    {
        int split = num_child - 1 ; 
        int ch = first_child ; 

        for(int i=0 ; i < split ; i++)  
        {
            snd& child = POOL->node[ch] ; 
            assert( child.index == ch ); 
            child.inorder_r( order, d+1 ); 
            ch = child.next_sibling ;
        }

        order.push_back(index) ; 

        for(int i=split ; i < num_child ; i++)
        {
            snd& child = POOL->node[ch] ; 
            assert( child.index == ch ); 
            child.inorder_r( order, d+1 ); 
            ch = child.next_sibling ;
        }
    }
}

/**
snd::Ancestors
---------------

Collect by following parent links then reverse 
the vector to put into root first order. 

**/

void snd::Ancestors(int idx, std::vector<int>& nodes)  // static 
{
    const snd* nd = Get(idx) ; 
    while( nd->parent > -1 ) 
    {    
        nodes.push_back(nd->parent);
        nd = Get(nd->parent) ; 
    }    
    std::reverse( nodes.begin(), nodes.end() );
}

void snd::ancestors(std::vector<int>& nodes) const
{
    Ancestors(index, nodes);  
}







void snd::leafnodes( std::vector<int>& nodes ) const
{
    leafnodes_r(nodes, 0 ); 
}

void snd::leafnodes_r( std::vector<int>& nodes, int d  ) const 
{
    if(is_leaf()) nodes.push_back(index); 

    int ch = first_child ; 
    for(int i=0 ; i < num_child ; i++)  
    {
        snd& child = POOL->node[ch] ; 
        assert( child.index == ch ); 
        child.leafnodes_r(nodes, d+1 );  
        ch = child.next_sibling ;
    }
}

int snd::Find(int idx, char l0) // static
{
    const snd* nd = Get(idx) ; 
    return nd ? nd->find(l0) : -1 ; 
} 

int snd::find(char l0) const 
{
    std::vector<int> nodes ; 
    find_r(nodes, l0, 0) ; 
    return nodes.size() == 1 ? nodes[0] : -1 ;
}

void snd::find_r(std::vector<int>& nodes, char l0, int d) const
{
    if(label[0] == l0) nodes.push_back(index); 

    int ch = first_child ; 
    for(int i=0 ; i < num_child ; i++)  
    {
        const snd* child = Get(ch) ; 
        assert( child->index == ch ); 
        child->find_r(nodes, l0, d+1 );  
        ch = child->next_sibling ;
    }
}




template<typename ... Args> 
void snd::typenodes_(std::vector<int>& nodes, Args ... tcs ) const 
{
    std::vector<OpticksCSG_t> types = {tcs ...};
    typenodes_r_(nodes, types, 0 ); 
}

// NB MUST USE SYSRAP_API TO PLANT THE SYMBOLS IN THE LIB  
template SYSRAP_API void snd::typenodes_(std::vector<int>& nodes, OpticksCSG_t ) const ; 
template SYSRAP_API void snd::typenodes_(std::vector<int>& nodes, OpticksCSG_t, OpticksCSG_t ) const ; 
template SYSRAP_API void snd::typenodes_(std::vector<int>& nodes, OpticksCSG_t, OpticksCSG_t, OpticksCSG_t ) const ; 


void snd::typenodes_r_(std::vector<int>& nodes, const std::vector<OpticksCSG_t>& types, int d) const 
{
    if(has_type(types)) nodes.push_back(index); 

    int ch = first_child ; 
    for(int i=0 ; i < num_child ; i++)  
    {
        snd& child = POOL->node[ch] ; 
        assert( child.index == ch ); 
        child.typenodes_r_(nodes, types, d+1 );  
        ch = child.next_sibling ;
    }
}

bool snd::has_type(const std::vector<OpticksCSG_t>& types) const 
{
    return std::find( types.begin(), types.end(), typecode ) != types.end() ; 
}


template<typename ... Args> 
std::string snd::DescType(Args ... tcs)  // static
{
    std::vector<OpticksCSG_t> types = {tcs ...};
    int num_tc = types.size(); 

    std::stringstream ss ; 
    for(int i=0 ; i < num_tc ; i++)
    {
        int tc = types[i]; 
        ss << CSG::Tag(tc)  ;
        if(i < num_tc - 1) ss << "," ; 
    }
    std::string str = ss.str(); 
    return str ; 
}

template SYSRAP_API std::string snd::DescType(OpticksCSG_t ); 
template SYSRAP_API std::string snd::DescType(OpticksCSG_t, OpticksCSG_t ); 
template SYSRAP_API std::string snd::DescType(OpticksCSG_t, OpticksCSG_t, OpticksCSG_t ); 




void snd::typenodes(std::vector<int>& nodes, int tc ) const 
{
    typenodes_r(nodes, tc, 0 ); 
}
void snd::typenodes_r(std::vector<int>& nodes, int tc, int d) const 
{
    if(typecode == tc ) nodes.push_back(index); 

    int ch = first_child ; 
    for(int i=0 ; i < num_child ; i++)  
    {
        snd& child = POOL->node[ch] ; 
        assert( child.index == ch ); 
        child.typenodes_r(nodes, tc, d+1 );  
        ch = child.next_sibling ;
    }
}



int snd::get_ordinal( const std::vector<int>& order ) const 
{
    int ordinal = std::distance( order.begin(), std::find(order.begin(), order.end(), index )) ; 
    return ordinal < int(order.size()) ? ordinal : -1 ; 
}


std::string snd::dump() const 
{
    std::stringstream ss ; 
    dump_r( ss, 0 ); 
    std::string str = ss.str(); 
    return str ; 
}

void snd::dump_r( std::ostream& os, int d ) const 
{
    os << "snd::dump_r"
       << " d " << d 
       << brief()
       << std::endl
       ;  

    int ch = first_child ; 
    while( ch > -1 )
    {
        snd& child = POOL->node[ch] ; 
        child.dump_r( os, d+1 ); 
        ch = child.next_sibling ;
    }
}



std::string snd::dump2() const 
{
    std::stringstream ss ; 
    dump2_r( ss, 0 ); 
    std::string str = ss.str(); 
    return str ; 
}

void snd::dump2_r( std::ostream& os, int d ) const 
{
    os << "snd::dump2_r"
       << " d " << d 
       << brief()
       << std::endl
       ;  

    int ch = first_child ; 
    for(int i=0 ; i < num_child ; i++)
    {
        snd& child = POOL->node[ch] ; 
        child.dump2_r( os, d+1 ); 
        ch = child.next_sibling ;
    }
}



std::string snd::Render(int idx, int mode)  // static
{  
    const snd* n = Get(idx); 
    return n ? n->render(mode) : "snd::Render bad idx "; 
}

std::string snd::render(int mode_) const 
{
    int width = num_node(); 
    int height = max_depth(); 
    int defmode = width > 16 ? 0 : 1 ; 
    int mode = mode_ > -1 ? mode_ : defmode ; 

    std::vector<int> order ; 
    inorder(order); 
    assert( int(order.size()) == width ); 

    scanvas canvas( width+1, height+2, 4, 2 );  
    render_r(&canvas, order, mode, 0); 

    std::stringstream ss ; 
    ss 
       << std::endl 
       << "snd::render"
       << " width " << width 
       << " height " << height  
       << " mode " << mode  
       << std::endl 
       << std::endl 
       << canvas.c 
       << std::endl
       ;

    std::string str = ss.str(); 
    return str ; 
}

void snd::render_r(scanvas* canvas, const std::vector<int>& order, int mode, int d) const 
{
    render_v(canvas, order, mode, d);   // visit in preorder 

    int ch = first_child ; 
    while( ch > -1 )
    {
        snd& child = POOL->node[ch] ; 
        child.render_r( canvas, order, mode, d+1 ); 
        ch = child.next_sibling ;
    }
}


/**
snd::render_v
-----------------
+-------+--------------------------------+
| mode  |  notes                         |
+=======+================================+
|  0    |  first character of snd label  |        
+-------+--------------------------------+
|  1    |  snd::index                    |
+-------+--------------------------------+
|  2    |  snd::typecode                 |
+-------+--------------------------------+
|  3    |  snd::typecode CSG::Tag        |
+-------+--------------------------------+

**/


void snd::render_v( scanvas* canvas, const std::vector<int>& order, int mode, int d ) const 
{
    int ordinal = get_ordinal( order ); 
    assert( ordinal > -1 ); 

    //std::cout << "snd::render_v " << brief() << " ordinal " << ordinal << std::endl ;  
 
    int ix = ordinal ; 
    int iy = d ;        // using depth instead of d would require snd::SetLVID to have been called
 
    char l0 = label[0] ;  
    if(l0 == '\0' ) l0 = 'o' ; 

    if( mode == 0 )
    {
        canvas->drawch( ix, iy, 0,0,  l0 );
    }
    else if( mode == 1 )
    {
        canvas->draw( ix, iy, 0,0,  index );
    }
    else if( mode == 2 )
    {
        canvas->draw( ix, iy, 0,0,  typecode );
    }
    else if( mode == 3 )
    {
        std::string tc = tag(); 
        canvas->draw( ix, iy, 0,0,  tc.c_str() );
    }
}









double snd::zmin() const 
{
    assert( CSG::CanZNudge(typecode) ); 
    assert( param > -1 ); 
    const spa& pa = POOL->param[param] ; 
    return pa.zmin(); 
}

double snd::zmax() const 
{
    assert( CSG::CanZNudge(typecode) ); 
    assert( param > -1 ); 
    const spa& pa = POOL->param[param] ; 
    return pa.zmax(); 
}

void snd::check_z() const 
{
    assert( CSG::CanZNudge(typecode) ); 
    assert( param > -1 ); 
    assert( aabb > -1 ); 

    const spa& pa = POOL->param[param] ; 
    const sbb& bb = POOL->aabb[aabb] ; 

    assert( pa.zmin() == bb.zmin() ); 
    assert( pa.zmax() == bb.zmax() ); 
}


/**
snd::decrease_zmin
-------------------

   bb.z1 +--------+ pa.z2 
         |        |
         |        |
         |________|
   bb.z0 +~~~~~~~~+ pa.z1

**/

void snd::decrease_zmin( double dz )
{
    check_z(); 

    spa& pa = POOL->param[param] ; 
    sbb& bb = POOL->aabb[aabb] ; 

    pa.decrease_zmin(dz); 
    bb.decrease_zmin(dz); 
}

/**
snd::increase_zmax
-------------------

::

   bb.z1 +~~~~~~~~+ pa.z2
         +--------+       
         |        |
         |        |
         |        |
   bb.z0 +--------+ pa.z1

**/

void snd::increase_zmax( double dz )
{
    check_z(); 

    spa& pa = POOL->param[param] ; 
    sbb& bb = POOL->aabb[aabb] ; 

    pa.increase_zmax(dz) ; 
    bb.increase_zmax(dz) ; 
}

/**
snd::ZDesc
-----------

   +----+
   |    |
   +----+
   |    |
   +----+
   |    |
   +----+

**/

std::string snd::ZDesc(const std::vector<int>& prims) // static
{
    std::stringstream ss ; 
    ss << "snd::ZDesc" ; 
    ss << " prims(" ;
    for(unsigned i=0 ; i < prims.size() ; i++) ss << prims[i] << " " ; 
    ss << ") " ;
    ss << std::endl ;  

    for(unsigned i=0 ; i < prims.size() ; i++)
    {
        int _a = prims[i];
        snd& a = POOL->node[_a] ; 
        ss << std::setw(3) << _a 
           << ":" 
           << " " << std::setw(10) << a.zmin() 
           << " " << std::setw(10) << a.zmax()
           << std::endl 
           ; 
    }
    std::string str = ss.str(); 
    return str ; 
}

/**
snd::ZNudgeEnds
-----------------

CAUTION: changes geometry, only appropriate 
for subtracted consituents eg inners 

**/

void snd::ZNudgeEnds(const std::vector<int>& prims) // static
{
    int level = Level(); 
    if(level > 0) std::cout 
       << std::endl
       << "snd::ZNudgeEnds "
       << std::endl
       << ZDesc(prims)
       << std::endl
       ;

    /*
    for(unsigned i=1 ; i < prims.size() ; i++)
    {
        int _a = prims[i-1]; 
        int _b = prims[i]; 

        snd& a = POOL->node[_a] ; 
        snd& b = POOL->node[_b] ; 
         
        a.check_z(); 
        b.check_z();
    }
    */
}

void snd::ZNudgeJoints(const std::vector<int>& prims) // static
{
    int level = Level(); 
    if(level > 0) std::cout 
       << std::endl
       << "snd::ZNudgeJoints "
       << std::endl
       << ZDesc(prims)
       << std::endl
       ;
}




std::string snd::desc() const 
{
    std::stringstream ss ; 
    ss 
       << "[snd::desc" << std::endl
       << brief() << std::endl  
       << DescParam(param) << std::endl  
       << DescAABB(aabb) << std::endl 
       << DescXForm(xform) << std::endl 
       ; 

    const snd& nd = *this ; 
    int ch = nd.first_child ; 
    int count = 0 ; 

    while( ch > -1 )
    {
        ss << Desc(ch) << std::endl ; 
        const snd& child = POOL->node[ch] ; 

        bool consistent_parent_index = child.parent == nd.index ; 

        if(!consistent_parent_index) ss 
            << "snd::desc "
            << " FAIL consistent_parent_index "
            << " ch " << ch 
            << " count " << count 
            << " child.parent " << child.parent
            << " nd.index " << nd.index
            << " nd.lvid "  << nd.lvid
            << " child.index " << child.index
            << " child.lvid "  << child.lvid
            << std::endl 
            ;

        //assert(consistent_parent_index);  
        count += 1 ;         
        ch = child.next_sibling ;
    }

    bool expect_child = count == nd.num_child ; 

    if(!expect_child) 
    {
        ss << std::endl << " FAIL count " << count << " num_child " << num_child << std::endl; 
    }
    assert(expect_child); 
    ss << "]snd::desc" << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}

std::ostream& operator<<(std::ostream& os, const snd& v)  
{
    os << v.desc() ;  
    return os; 
}







snd snd::Init(int tc)  // static
{
    snd nd = {} ;
    nd.init(); 
    nd.typecode = tc ; 
    nd.num_child = 0 ; 
    return nd ; 
}


/**
snd::init
-----------

**/

void snd::init()
{
    index = -1 ; 
    depth = -1 ; 
    sibdex = -1 ; 
    parent = -1 ; 

    num_child = -1 ; 
    first_child = -1 ; 
    next_sibling = -1 ; 
    lvid = -1 ;

    typecode = -1  ; 
    param = -1 ; 
    aabb = -1 ; 
    xform = -1 ; 
}







/**
snd::Boolean
--------------

NB forming a boolean sets up the 
sibling and parent node linkages 

HMM: for positivization need to first 
create the full flexible sn tree 
and only mint 



**/

int snd::Boolean( int op, int l, int r ) // static 
{
    assert( l > -1 && r > -1 );

    snd nd = Init( op );   
    assert( nd.xform == -1 );

    nd.num_child = 2 ; 
    nd.first_child = l ;

    snd* ln = Get_(l) ; 
    snd* rn = Get_(r) ; 

    ln->next_sibling = r ; 
    ln->sibdex = 0 ; 

    rn->next_sibling = -1 ; 
    rn->sibdex = 1 ; 

    int idx_0 = POOL->node.size() ;  

    ln->parent = idx_0 ; 
    rn->parent = idx_0 ; 

    int idx = Add(nd) ; 

    assert( idx_0 == idx ); 

    /*
    ln->parent = idx ;  // <-- INSIDIOUS BUG : DONT USE PTRS/REFS AFTER snd::Add 
    rn->parent = idx ;  // <-- INSIDIOUS BUG : DONT USE PTRS/REFS AFTER snd::Add 
    
    NB : IT WOULD BE AN INSIDIOUS BUG TO USE *ln/rn* POINTERS/REFERENCES
    HERE AS REALLOC WILL SOMETIMES HAPPEN WHEN DO snd::Add WHICH 
    WOULD INVALIDATE THE POINTERS/REFERENCES OBTAINED PRIOR TO snd::Add
    
    THE BUG MANIFESTS AS PARENT FIELDS NOT BEING SET AS THE ln/rn WOULD 
    NO LONGER BE POINTING INTO THE NODE VECTOR DUE TO THE REALLOCATION.
    */

    return idx ; 
}

int snd::Compound(int type, const std::vector<int>& prims )
{
    assert( type == CSG_CONTIGUOUS || type == CSG_DISCONTIGUOUS ); 

    int num_prim = prims.size(); 
    assert( num_prim > 0 ); 

    snd nd = Init( type ); 
    nd.num_child = num_prim ; 
    nd.first_child = prims[0] ;
    int idx = Add(nd) ; 

    for(int i=0 ; i < num_prim ; i++)
    {
        int i_sib = prims[i]; 
        int p_sib = i > 0 ? prims[i-1] : -1 ; 

        snd& i_child = POOL->node[i_sib] ; 
        i_child.sibdex = i ; 
        i_child.parent = idx ; 
        i_child.next_sibling = -1 ; 

        // other than for last i = num_prim-1 
        // the next_sibling gets reset by prior "reach back" below  

        if(i > 0)
        {
            assert( p_sib > -1 ); 
            snd& p_child = POOL->node[p_sib] ; 
            p_child.next_sibling = i_sib ; 
        }
    }
    return idx ; 
}


int snd::UnionTree(const std::vector<int>& prims )
{
    int idx = sndtree::CommonTree( prims, CSG_UNION ); 
    return idx ; 
}

int snd::Contiguous( const std::vector<int>& prims )
{
    int idx = snd::Compound( CSG_CONTIGUOUS, prims ); 
    return idx ; 
}


/**
snd::Collection
-----------------

Used for example from U4Polycone::init 

+-------------+-------------------+-------------------+
|  VERSION    |  Impl             |  Notes            |
+=============+===================+===================+ 
|     0       |  snd::UnionTree   | backward looking  | 
+-------------+-------------------+-------------------+
|     1       |  snd::Contiguous  | forward looking   |   
+-------------+-------------------+-------------------+

**/

int snd::Collection( const std::vector<int>& prims ) // static
{ 
    int idx = -1 ; 
    switch(VERSION)
    {   
        case 0: idx = UnionTree(prims)  ; break ; 
        case 1: idx = Contiguous(prims) ; break ;
    }   
    return idx ; 
}


int snd::Cylinder(double radius, double z1, double z2) // static
{
    assert( z2 > z1 );  
    snd nd = Init(CSG_CYLINDER); 
    nd.setParam( 0.f, 0.f, 0.f, radius, z1, z2)  ;   
    nd.setAABB( -radius, -radius, z1, +radius, +radius, z2 );   
    return Add(nd) ; 
}

int snd::Cone(double r1, double z1, double r2, double z2)  // static
{   
    assert( z2 > z1 );
    double rmax = fmax(r1, r2) ; 
    snd nd = Init(CSG_CONE) ;
    nd.setParam( r1, z1, r2, z2, 0., 0. ) ;
    nd.setAABB( -rmax, -rmax, z1, rmax, rmax, z2 );
    return Add(nd) ;
}

int snd::Sphere(double radius)  // static
{
    assert( radius > zero ); 
    snd nd = Init(CSG_SPHERE) ; 
    nd.setParam( zero, zero, zero, radius, zero, zero );  
    nd.setAABB(  -radius, -radius, -radius,  radius, radius, radius  );  
    return Add(nd) ;
}

int snd::ZSphere(double radius, double z1, double z2)  // static
{
    assert( radius > zero ); 
    assert( z2 > z1 );  
    snd nd = Init(CSG_ZSPHERE) ; 
    nd.setParam( zero, zero, zero, radius, z1, z2 );  
    nd.setAABB(  -radius, -radius, z1,  radius, radius, z2  );  
    return Add(nd) ;
}

int snd::Box3(double fullside)  // static 
{
    return Box3(fullside, fullside, fullside); 
}
int snd::Box3(double fx, double fy, double fz )  // static 
{
    assert( fx > 0. );  
    assert( fy > 0. );  
    assert( fz > 0. );  

    snd nd = Init(CSG_BOX3) ; 
    nd.setParam( fx, fy, fz, 0.f, 0.f, 0.f );  
    nd.setAABB( -fx*0.5 , -fy*0.5, -fz*0.5, fx*0.5 , fy*0.5, fz*0.5 );   
    return Add(nd) ; 
}

int snd::Zero(double  x,  double y,  double z,  double w,  double z1, double z2) // static 
{
    snd nd = Init(CSG_ZERO); 
    nd.setParam( x, y, z, w, z1, z2 );  
    return Add(nd) ; 
}

int snd::Zero() // static
{
    snd nd = Init(CSG_ZERO); 
    return Add(nd) ; 
}


