#include <cassert>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>

#include "G4SolidStore.hh"
#include "G4UnionSolid.hh"
#include "G4IntersectionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4Ellipsoid.hh"
#include "G4Tubs.hh"
#include "G4Polycone.hh"
#include "G4Torus.hh"
#include "G4Box.hh"
#include "G4Orb.hh"
#include "G4MultiUnion.hh"

#include "G4RotationMatrix.hh"

#include "SCanvas.hh"
#include "X4SolidTree.hh"


const bool X4SolidTree::verbose = getenv("X4SolidTree_verbose") != nullptr ; 

G4VSolid* X4SolidTree::ApplyZCutTree( const G4VSolid* original, double zcut ) // static
{
    if(verbose)
    std::cout << "[ X4SolidTree::ApplyZCutTree zcut " << zcut << " original.GetName " << original->GetName() << std::endl ; 

    X4SolidTree* zs = new X4SolidTree(original); 
    zs->apply_cut( zcut );  
    //zs->dump("X4SolidTree::ApplyZCutTree"); 

    if(verbose)
    std::cout << "] X4SolidTree::ApplyZCutTree  zs.root.GetName " << zs->root->GetName()  << std::endl ; 
    return zs->root ; 
}

void X4SolidTree::Draw(const G4VSolid* original, const char* msg ) // static
{
    if(original == nullptr )
    {
        std::cout << "ERROR X4SolidTree::Draw got nullptr original : msg " << msg << std::endl ; 
        return ; 
    }    

    X4SolidTree* zs = new X4SolidTree(original); 
    zs->draw(msg);
    zs->dumpNames(msg); 
    zs->zdump(msg); 
}

X4SolidTree::X4SolidTree(const G4VSolid* original_ ) 
    :
    original(original_),
    root(DeepClone(original_)),
    edited(false),
    parent_map(new std::map<const G4VSolid*, const G4VSolid*>),

    in_map(    new std::map<const G4VSolid*, int>),
    rin_map(   new std::map<const G4VSolid*, int>),
    pre_map(   new std::map<const G4VSolid*, int>),
    rpre_map(  new std::map<const G4VSolid*, int>),
    post_map(  new std::map<const G4VSolid*, int>),
    rpost_map( new std::map<const G4VSolid*, int>),

    zcls_map(  new std::map<const G4VSolid*, int>),
    depth_map( new std::map<const G4VSolid*, int>),
    mkr_map(   new std::map<const G4VSolid*, char>),

    width(0),
    height(0), 
    extra_width(1),    // +1 for annotations to the right
    extra_height(1+1), // +1 as height zero tree is still one node, +1 for annotation  
    canvas(  new SCanvas(width+extra_width, height+extra_height, 8, 5) ),
    names( new std::vector<std::string>),
    nameprefix(nullptr),
    inorder( new std::vector<const G4VSolid*> ),
    rinorder( new std::vector<const G4VSolid*> ),
    preorder( new std::vector<const G4VSolid*> ),
    rpreorder( new std::vector<const G4VSolid*> ),
    postorder( new std::vector<const G4VSolid*> ),
    rpostorder( new std::vector<const G4VSolid*> ),
    crux( new std::vector<G4VSolid*> )
{
    init(); 
}





void X4SolidTree::init()
{
    if(verbose) std::cout << "X4SolidTree::init" << std::endl ; 
    instrumentTree(); 
}

void X4SolidTree::dump(const char* msg) const 
{
    dumpTree(msg); 
    dumpUp(msg); 
}

void X4SolidTree::instrumentTree()
{
    if(verbose) std::cout << "X4SolidTree::instrumentTree parent " << std::endl ; 
    parent_map->clear(); 
    (*parent_map)[root] = nullptr ;  // root has no parent by definition
    parent_r( root, 0 ); 

    if(verbose) std::cout << "X4SolidTree::instrumentTree depth " << std::endl ; 
    depth_map->clear(); 
    depth_r(root, 0 ); 

    if(verbose) std::cout << "X4SolidTree::instrumentTree inorder " << std::endl ; 
    inorder->clear();
    inorder_r(root, 0 ); 

    if(verbose) std::cout << "X4SolidTree::instrumentTree rinorder " << std::endl ; 
    rinorder->clear();
    rinorder_r(root, 0 ); 

    if(verbose) std::cout << "X4SolidTree::instrumentTree preorder " << std::endl ; 
    preorder->clear();
    preorder_r(root, 0 ); 

    if(verbose) std::cout << "X4SolidTree::instrumentTree rpreorder " << std::endl ; 
    rpreorder->clear();
    rpreorder_r(root, 0 ); 

    if(verbose) std::cout << "X4SolidTree::instrumentTree postorder " << std::endl ; 
    postorder->clear(); 
    postorder_r(root, 0 ); 

    if(verbose) std::cout << "X4SolidTree::instrumentTree rpostorder " << std::endl ; 
    rpostorder->clear(); 
    rpostorder_r(root, 0 ); 

    if(verbose) std::cout << "X4SolidTree::instrumentTree names" << std::endl ; 
    names->clear(); 
    collectNames_inorder_r( root, 0 ); 

    if(verbose)
    {
        std::cout << "X4SolidTree::instrumentTree names.size " << names->size() << std::endl ; 
        for(unsigned i=0 ; i < names->size() ; i++) std::cout << "[" << (*names)[i] << "]" << std::endl ;  
    }
    if(verbose) std::cout << "X4SolidTree::instrumentTree nameprefix" << std::endl ; 
    nameprefix = CommonPrefix(names); 
    if(verbose) std::cout << "X4SolidTree::instrumentTree nameprefix [" << nameprefix << "]" << std::endl ; 

    if(verbose) std::cout << "X4SolidTree::instrumentTree [ original_num_node " << std::endl ; 
    int original_num_node = NumNode_r(original, 0); 
    if(verbose) std::cout << "X4SolidTree::instrumentTree ] original_num_node : " << original_num_node  << std::endl ; 

    if(verbose) std::cout << "X4SolidTree::instrumentTree [ root_num_node " << std::endl ; 
    int root_num_node = NumNode_r(root, 0); 
    if(verbose) std::cout << "X4SolidTree::instrumentTree ] root_num_node : " << root_num_node << std::endl ; 

    int depth_size = depth_map->size(); 
    int inorder_size = inorder->size(); 
    int rinorder_size = rinorder->size(); 
    int preorder_size = preorder->size(); 
    int rpreorder_size = rpreorder->size(); 
    int postorder_size = postorder->size(); 
    int rpostorder_size = rpostorder->size(); 

    if(verbose) std::cout 
        << "X4SolidTree::instrumentTree"
        << " depth_size " << depth_size
        << " inorder_size " << inorder_size
        << " rinorder_size " << rinorder_size
        << " preorder_size " << preorder_size
        << " rpreorder_size " << rpreorder_size
        << " postorder_size " << postorder_size
        << " rpostorder_size " << rpostorder_size
    //    << " original_num_node " << original_num_node
        << " root_num_node " << root_num_node
        << std::endl 
        ;

    assert( depth_size == root_num_node );   
    assert( inorder_size == root_num_node );   
    assert( rinorder_size == root_num_node );   
    assert( preorder_size == root_num_node );   
    assert( rpreorder_size == root_num_node );   
    assert( postorder_size == root_num_node );   
    assert( rpostorder_size == root_num_node );   

    if( edited == false )
    {
        assert( original_num_node == root_num_node ); 
    }

    width = root_num_node ; 
    height = maxdepth() ; 

    if(verbose) printf("X4SolidTree::instrumentTree width %d height %d \n", width, height );     
    canvas->resize( width+extra_width, height+extra_height );    
}


const char* X4SolidTree::CommonPrefix(const std::vector<std::string>* a) // static
{
    std::vector<std::string> aa(*a); 

    if(verbose)
    {
        std::cout << "X4SolidTree::CommonPrefix aa.size (before sort) " << aa.size() << std::endl ;  
        for(unsigned i=0 ; i < aa.size() ; i++) std::cout << "[" << aa[i] << "]" << std::endl ;  
    }

    std::sort( aa.begin(), aa.end() );  

    if(verbose)
    {
        std::cout << "X4SolidTree::CommonPrefix aa.size (after sort) " << aa.size() << std::endl ;  
        for(unsigned i=0 ; i < aa.size() ; i++) std::cout << "[" << aa[i] << "]" << std::endl ;  
    }

    const std::string& s1 = aa[0] ; 
    const std::string& s2 = aa[aa.size()-1] ; 

    std::string prefix(s1) ; 
    for(unsigned i=0 ; i < s1.size() ; i++) 
    {
        if( s1[i] != s2[i] ) 
        {
            prefix = s1.substr(0,i) ; 
            break ; 
        }
    }
    const char* prefix_ = strdup(prefix.c_str()) ; 

    if(verbose)
    {
        std::cout << "X4SolidTree::CommonPrefix s1 " << s1 << std::endl ;  
        std::cout << "X4SolidTree::CommonPrefix s2 " << s2 << std::endl ;  
        std::cout << "X4SolidTree::CommonPrefix prefix " << prefix << std::endl ;  
        std::cout << "X4SolidTree::CommonPrefix prefix_ " << prefix_ << std::endl ;  
    }

    return prefix_  ; 
} 

/**
X4SolidTree::parent_r
-----------------------

Note that the parent_map uses the raw constituent G4DisplacedSolid 
rather than the moved G4VSolid that it points to in order to have 
treewise access to the transform up the lineage. 

**/
void X4SolidTree::parent_r( const G4VSolid* node_, int depth)
{
    if( node_ == nullptr ) return ; 
    const G4VSolid* node = Moved(node_ ); 

    if(Boolean(node))
    {
        const G4VSolid* l = Left(node) ; 
        const G4VSolid* r = Right(node) ; 

        parent_r( l, depth+1 );  
        parent_r( r, depth+1 );  

        // postorder visit 
        (*parent_map)[l] = node_ ; 
        (*parent_map)[r] = node_ ; 
    }
}

void X4SolidTree::depth_r(const G4VSolid* node_, int depth)
{
    if( node_ == nullptr ) return ; 
    const G4VSolid* node = Moved(node_ ); 

    depth_r(Left(node), depth+1) ; 
    depth_r(Right(node), depth+1) ; 

    (*depth_map)[node_] = depth ;   
}

void X4SolidTree::inorder_r(const G4VSolid* node_, int depth)
{
    if( node_ == nullptr ) return ; 
    const G4VSolid* node = Moved(node_ ); 

    inorder_r(Left(node), depth+1) ; 

    (*in_map)[node_] = inorder->size();   
    inorder->push_back(node_) ; 

    inorder_r(Right(node), depth+1) ; 
} 

void X4SolidTree::rinorder_r(const G4VSolid* node_, int depth)
{
    if( node_ == nullptr ) return ; 
    const G4VSolid* node = Moved(node_ ); 

    rinorder_r(Right(node), depth+1) ; 

    (*rin_map)[node_] = rinorder->size();   
    rinorder->push_back(node_) ; 

    rinorder_r(Left(node), depth+1) ; 
} 

void X4SolidTree::preorder_r(const G4VSolid* node_, int depth)
{
    if( node_ == nullptr ) return ; 
    const G4VSolid* node = Moved(node_ ); 

    (*pre_map)[node_] = preorder->size();   
    preorder->push_back(node_) ; 

    preorder_r(Left(node), depth+1) ; 
    preorder_r(Right(node), depth+1) ; 
} 

void X4SolidTree::rpreorder_r(const G4VSolid* node_, int depth)
{
    if( node_ == nullptr ) return ; 
    const G4VSolid* node = Moved(node_ ); 

    (*rpre_map)[node_] = rpreorder->size();   
    rpreorder->push_back(node_) ; 

    rpreorder_r(Right(node), depth+1) ; 
    rpreorder_r(Left(node), depth+1) ; 
} 

void X4SolidTree::postorder_r(const G4VSolid* node_, int depth)
{
    if( node_ == nullptr ) return ; 
    const G4VSolid* node = Moved(node_ ); 

    postorder_r(Left(node), depth+1) ; 
    postorder_r(Right(node), depth+1) ; 

    (*post_map)[node_] = postorder->size();   
    postorder->push_back(node_) ; 
}

void X4SolidTree::rpostorder_r(const G4VSolid* node_, int depth)
{
    if( node_ == nullptr ) return ; 
    const G4VSolid* node = Moved(node_ ); 

    rpostorder_r(Right(node), depth+1) ; 
    rpostorder_r(Left(node), depth+1) ; 

    (*rpost_map)[node_] = rpostorder->size();   
    rpostorder->push_back(node_) ; 
}

const G4VSolid* X4SolidTree::parent(  const G4VSolid* node ) const { return parent_map->count(node) == 1 ? (*parent_map)[node] : nullptr ; }
G4VSolid*       X4SolidTree::parent_( const G4VSolid* node ) const { const G4VSolid* p = parent(node); return const_cast<G4VSolid*>(p) ; }

int X4SolidTree::depth( const G4VSolid* node_) const { return (*depth_map)[node_] ; }
int X4SolidTree::in(    const G4VSolid* node_) const { return (*in_map)[node_] ; }
int X4SolidTree::rin(   const G4VSolid* node_) const { return (*rin_map)[node_] ; }
int X4SolidTree::pre(   const G4VSolid* node_) const { return (*pre_map)[node_] ; }
int X4SolidTree::rpre(  const G4VSolid* node_) const { return (*rpre_map)[node_] ; }
int X4SolidTree::post(  const G4VSolid* node_) const { return (*post_map)[node_] ; }
int X4SolidTree::rpost( const G4VSolid* node_) const { return (*rpost_map)[node_] ; }

/**
X4SolidTree::index
---------------

Returns the index of a node within various traversal orders, 
obtained by lookups on the maps collected by instrumentTree.


IN inorder
   left-to-right index (aka side) 

RIN reverse inorder
   right-to-left index 

PRE preorder
   with left unbalanced tree this does a anti-clockwise rotation starting from root
   [by observation is opposite to RPOST]

RPRE reverse preorder
   with left unbalanced does curios zig-zag that looks to be exact opposite of 
   the familiar postorder traversal, kinda undo of the postorder.
   [by observation is opposite to POST]

   **COULD BE USEFUL FOR TREE EDITING FROM RIGHT** 

POST postorder
   old familar traversal starting from bottom left 
   [by observation is opposite to RPRE]

RPOST reverse postorder
   with left unbalanced does a clockwise cycle starting
   from rightmost visiting all prim before getting 
   to any operators
   [by observation is opposite to PRE]

**/

int X4SolidTree::index( const G4VSolid* n, int mode ) const  
{
    int idx = -1 ; 
    switch(mode)
    {
        case IN:    idx = in(n)    ; break ; 
        case RIN:   idx = rin(n)   ; break ; 
        case PRE:   idx = pre(n)   ; break ; 
        case RPRE:  idx = rpre(n)  ; break ; 
        case POST:  idx = post(n)  ; break ; 
        case RPOST: idx = rpost(n) ; break ;  
    }
    return idx ; 
}

const char* X4SolidTree::IN_ = "IN" ; 
const char* X4SolidTree::RIN_ = "RIN" ; 
const char* X4SolidTree::PRE_ = "PRE" ; 
const char* X4SolidTree::RPRE_ = "RPRE" ; 
const char* X4SolidTree::POST_ = "POST" ; 
const char* X4SolidTree::RPOST_ = "RPOST" ; 

const char* X4SolidTree::OrderName(int mode) // static
{
    const char* s = nullptr ; 
    switch(mode)
    {
        case IN:    s = IN_    ; break ; 
        case RIN:   s = RIN_   ; break ; 
        case PRE:   s = PRE_   ; break ; 
        case RPRE:  s = RPRE_  ; break ; 
        case POST:  s = POST_  ; break ; 
        case RPOST: s = RPOST_ ; break ;  
    }
    return s ; 
}

/**
X4SolidTree::zcls
--------------

Returns zcls value associated with a node, when *move* is true any 
G4DisplacedSolid *node_* are dereferenced to get to the moved solid *node* 
within.

Am veering towards using move:false as standard because can 
always get the moved solid from the G4DisplacedSolid but not vice versa.
Note that it is a technicality : either could work its just a case of
which is most convenient. 

**/

int X4SolidTree::zcls( const G4VSolid* node_ ) const 
{ 
    return zcls_map->count(node_) == 1 ? (*zcls_map)[node_] : UNDEFINED ; 
}

void X4SolidTree::set_zcls( const G4VSolid* node_, int zc )
{
    (*zcls_map)[node_] = zc  ;
} 

char X4SolidTree::mkr( const G4VSolid* node_) const 
{ 
    return mkr_map->count(node_) == 1 ? (*mkr_map)[node_] : ' ' ; 
}

void X4SolidTree::set_mkr( const G4VSolid* node_, char mk )
{
    (*mkr_map)[node_] = mk  ;
} 


bool X4SolidTree::is_include( const G4VSolid* node_) const 
{
    return zcls(node_) == INCLUDE ; 
}
bool X4SolidTree::is_exclude( const G4VSolid* node_) const 
{
    return zcls(node_) == EXCLUDE ; 
}

bool X4SolidTree::is_exclude_include( const G4VSolid* node_) const 
{
    if(!Boolean(node_)) return false ; 
    const G4VSolid* left_  = Left(node_); 
    const G4VSolid* right_ = Right(node_); 
    return is_exclude(left_) &&  is_include(right_) ; 
}

bool X4SolidTree::is_include_exclude( const G4VSolid* node_) const 
{
    if(!Boolean(node_)) return false ; 
    const G4VSolid* left_  = Left(node_); 
    const G4VSolid* right_ = Right(node_); 
    return is_include(left_) &&  is_exclude(right_) ; 
}

bool X4SolidTree::is_crux( const G4VSolid* node_ ) const 
{
    if(!Boolean(node_)) return false ; 
    const G4VSolid* left_  = Left(node_); 
    const G4VSolid* right_ = Right(node_); 
    bool exclude_include = zcls(left_) == EXCLUDE &&  zcls(right_) == INCLUDE ;
    bool include_exclude = zcls(left_) == INCLUDE &&  zcls(right_) == EXCLUDE ; 
    return exclude_include ^ include_exclude ;   // XOR one or other, not both 
}

int X4SolidTree::num_prim_r(const G4VSolid* node_) const
{   
    if(!node_) return 0 ; 
    const G4VSolid* n = Moved(node_); 
    const G4VSolid* l = Left(n); 
    const G4VSolid* r = Right(n); 
    return ( l && r ) ? num_prim_r(l) + num_prim_r(r) : 1 ;
}

int X4SolidTree::num_prim() const 
{
    return num_prim_r(root); 
}

int X4SolidTree::num_node() const
{
    return NumNode_r(root, 0) ; 
}

/**
X4SolidTree::NumNode_r
------------------

In order to visit all nodes it is essential to deref potentially 
G4DisplacedSolid with Moved otherwise may for example miss displaced booleans. 
Noticed this with the JUNO PMT with tubs subtract torus neck. 

**/

int X4SolidTree::NumNode_r(const G4VSolid* node_, int depth) // static 
{
    int num = node_ ? 1 : 0 ;
    if( node_ )
    {
        const G4VSolid* node = Moved(node_ ); 
        const G4VSolid* left = Left(node); 
        const G4VSolid* right = Right(node); 

        if( left && right )
        {   
            num += NumNode_r( left,  depth+1 );  
            num += NumNode_r( right, depth+1 );  
        }   

        if(verbose) std::cout 
            << "X4SolidTree::NumNode_r " 
            << " depth " << std::setw(3) << depth
            << " type " << std::setw(20) << EntityTypeName(node)
            << " name " << std::setw(20) << node->GetName()
            << " left " << std::setw(20) << ( left ? left->GetName() : "-" ) 
            << " right " << std::setw(20) << ( right ? right->GetName() : "-" )
            << " num " << std::setw(3) << num
            << std::endl 
            ;       
    }
    return num ; 
} 

int X4SolidTree::num_node_select(int qcls) const
{
    return num_node_select_r(root, qcls) ; 
}
int X4SolidTree::num_node_select_r(const G4VSolid* node_, int qcls) const 
{
    int zcl = zcls(node_) ;
    const G4VSolid* node = Moved(node_ ); 
 

    int num = ( node_ && zcl == qcls ) ? 1 : 0 ;
    if( node )
    {
        if(verbose) std::cout 
            << "X4SolidTree::num_node_select_r"
            << " zcl " << zcl
            << " zcn " << ClassifyMaskName(zcl)
            << " Desc " << Desc(node)
            << std::endl
            ;  

        const G4VSolid* l = Left(node); 
        const G4VSolid* r = Right(node); 
        if( l && r )
        {   
            num += num_node_select_r( l, qcls );  
            num += num_node_select_r( r, qcls );  
        }   
    }
    return num ; 
} 

const char* X4SolidTree::desc() const 
{
    int num_node_     = num_node(); 
    int num_prim_     = num_prim(); 

    int num_undefined = num_node_select(UNDEFINED);  
    int num_exclude   = num_node_select(EXCLUDE);  
    int num_include   = num_node_select(INCLUDE);  
    int num_mixed     = num_node_select(MIXED);  

    std::stringstream ss ; 
    ss 
       << " NODE:" << num_node_
       << " PRIM:" << num_prim_
       << " UNDEFINED:" << num_undefined
       << " EXCLUDE:" << num_exclude
       << " INCLUDE:" << num_include
       << " MIXED:" << num_mixed
       ; 
     
    std::string s = ss.str(); 
    return strdup(s.c_str());
}

void X4SolidTree::prune(bool act, int pass)
{
    int num_include = num_node_select(INCLUDE) ;

    if(verbose)
    printf("X4SolidTree::prune act:%d pass:%d num_include %d \n", act, pass, num_include);

    if( num_include == 0)
    {
        if(verbose)
        printf("X4SolidTree::prune act:%d pass:%d find zero remaining nodes : num_include %d, will set root to nullptr \n", act, pass, num_include);

        if(act)
        {
            if(verbose)
            printf("X4SolidTree:::prune act:%d pass:%d setting root to nullptr \n", act, pass);

            root = nullptr ;
            edited = true ; 
        }
    }

    if(crux->size() == 0 ) return ;
    bool expect= crux->size() == 1 ; 
    if(!expect) exit(EXIT_FAILURE); 
    assert(expect) ;       // more than one crux node not expected

    G4VSolid* x = (*crux)[0] ;
    prune_crux(x, act, pass);
}

/**
X4SolidTree::prune_crux
-------------------

**/

void X4SolidTree::prune_crux(G4VSolid* x, bool act, int pass)
{
    assert( x );
    bool ie = is_include_exclude(x) ;  // include left, exclude right
    bool ei = is_exclude_include(x) ;  // exclude left, include right 
    bool expect_0 =  ie ^ ei ;     // XOR definition of crux node 
    if(!expect_0) exit(EXIT_FAILURE) ; 

    G4VSolid* survivor = ie ? Left_(x) : Right_(x) ;
    bool expect_1 = survivor != nullptr ; 
    if(!expect_1) exit(EXIT_FAILURE); 
    assert( survivor );

    set_mkr(survivor, 'S') ;

    G4VSolid* p = parent_(x); 

    if(!is_include(p)) // hmm problems with this will not be apparent with the maximally unbalanced trees 
    {
        if(verbose) 
        printf("X4SolidTree::prune_crux act:%d pass:%d parent disqualified as not INCLUDE : handle as root of the new tree\n", act, pass) ; 
        p = nullptr ;    // as *p* is just local variable are changing it even when act=false 
    }

    if( p != nullptr )   // non-root prune
    {
        G4VSolid* p_left = Left_(p); 
        G4VSolid* p_right = Right_(p); 

        bool x_is_p_left  = x == p_left ; 
        bool x_is_p_right = x == p_right ; 

        set_mkr(p, 'P') ;

        if(x_is_p_left)
        {
            if(act)
            {
                if(verbose)
                printf("X4SolidTree:::prune_crux act:%d pass:%d SetLeft changing p.left to survivor \n", act, pass);
                SetLeft(p, survivor) ; 
                edited = true ; 
            }
        }
        else if(x_is_p_right)
        {
            if(act)
            {
                if(verbose)
                printf("X4SolidTree:prune_crux act:%d pass:%d SetRight changing p.right to survivor  \n", act, pass);
                SetRight(p, survivor) ; 
                edited = true ; 
            }
        }
    }
    else           // root prune
    {
        if( act )
        {

            G4String root_name = root->GetName(); 
            G4String survivor_name = survivor->GetName() ; 
            if(verbose)
            printf("X4SolidTree::prune_crux act:%d pass:%d changing root %s to survivor %s \n", act, pass, root_name.c_str(), survivor_name.c_str() );
            root = survivor ;
            edited = true ; 
        }
    }

    if(act)
    {
        instrumentTree();
    }
}

void X4SolidTree::draw(const char* msg, int pass) 
{
    canvas->clear();

    //int mode = RPRE ; 
    int mode = IN ; 
    draw_r(root, mode);

    canvas->draw(   -1, -1, 0,0,  "zdelta" ); 
    canvas->draw(   -1, -1, 0,2,  "az1" ); 
    canvas->draw(   -1, -1, 0,3,  "az0" ); 

    std::cout 
        << msg 
        << " [" << std::setw(2) << pass << "]" 
        << " nameprefix " << ( nameprefix ? nameprefix : "-" ) 
        << " " << desc() 
        << " Order:" << OrderName(mode) 
        << std::endl
        ;
 
    for(unsigned i=0 ; i < names->size() ; i++ ) 
    {
        const std::string& name = (*names)[i] ; 
        std::string nam = nameprefix ? name.substr(strlen(nameprefix)) : "" ; 
        std::string snam = nam.substr(0,6) ;  
        if(false) std::cout 
            << std::setw(3) << i 
            << " : " 
            << std::setw(20) << name 
            << " : " 
            << std::setw(20) << nam 
            << " : [" 
            << std::setw(6) << snam << "]" 
            << std::endl
            ;
        canvas->draw(  i,  -1, 0,4, snam.c_str() ); 
    } 
    canvas->print(); 
}

void X4SolidTree::dumpNames(const char* msg) const 
{
    std::cout << msg << std::endl ; 
    for(unsigned i=0 ; i < names->size() ; i++ ) 
    {
        const std::string& name = (*names)[i] ; 
        std::string nam = nameprefix ? name.substr(strlen(nameprefix)) : "" ; 
        std::string snam = nam.substr(0,6) ;  
        std::cout 
            << std::setw(3) << i 
            << " : " 
            << std::setw(35) << name 
            << " : " 
            << std::setw(35) << nam 
            << " : [" 
            << std::setw(10) << snam << "]" 
            << std::endl
            ;
    } 
}

void X4SolidTree::zdump(const char* msg) const 
{
    std::cout << msg << std::endl ; 
    int mode = IN ; 
    zdump_r(root, mode); 
}

void X4SolidTree::zdump_r( const G4VSolid* node_, int mode ) const 
{
    if( node_ == nullptr ) return ;

    const G4VSolid* node = Moved(node_); 
    zdump_r( Left(node),  mode  );
    zdump_r( Right(node), mode );

    int ix = in(node_) ;            // inorder index, aka "side", increasing from left to right 
    int iy = depth(node_) ;         // increasing downwards
    int idx = index(node_, mode);  // index for presentation 

    const char* tag = EntityTag(node, true) ; 
    int zcl = zcls(node_);                 
    const char* zcn = ClassifyMaskName(zcl) ; 
    G4String name = node->GetName(); 


    bool can_z = X4SolidTree::CanZ(node) ;

    if(can_z)
    {
        double zdelta = getZ(node_) ;  
        double z0, z1 ; 
        ZRange(z0, z1, node);  

        double az0 = z0 + zdelta ;  
        double az1 = z1 + zdelta ;  

        std::cout 
            << " ix " << std::setw(2) << ix
            << " iy " << std::setw(2) << iy
            << " idx " << std::setw(2) << idx
            << " tag " << std::setw(10) << tag 
            << " zcn " << std::setw(10) << zcn 
            << " zdelta " << std::setw(10) << std::fixed << std::setprecision(3) << zdelta 
            << " az0 " << std::setw(10) << std::fixed << std::setprecision(3) << az0
            << " az1 " << std::setw(10) << std::fixed << std::setprecision(3) << az1
            << " name " << name
            << std::endl 
            ; 
    }
}



/**
X4SolidTree::draw_r
----------------

Recursively paints nodes of the tree onto the canvas
using the *mode* traversal order to label the nodes

**/

void X4SolidTree::draw_r( const G4VSolid* node_, int mode )
{
    if( node_ == nullptr ) return ;

    const G4VSolid* node = Moved(node_); 
    draw_r( Left(node),  mode );
    draw_r( Right(node), mode );

    int ix = in(node_) ;            // inorder index, aka "side", increasing from left to right 
    int iy = depth(node_) ;         // increasing downwards
    int idx = index(node_, mode);  // index for presentation 

    const char* tag = EntityTag(node, true) ; 
    int zcl = zcls(node_);                 
    const char* zcn = ClassifyMaskName(zcl) ; 
    char mk = mkr(node); 

    canvas->draw(   ix, iy, 0,0,  tag); 
    canvas->draw(   ix, iy, 0,1,  zcn); 
    canvas->draw(   ix, iy, 0,2,  idx); 
    canvas->drawch( ix, iy, 0,3,  mk ); 

    bool can_y = X4SolidTree::CanY(node) ;
    bool can_z = X4SolidTree::CanZ(node) ;
    if(verbose) std::cout << "X4SolidTree::draw_r can_z " << can_z << std::endl ;  

    if(can_y)
    {
        double y_delta = getY(node_) ;  
        double y0, y1 ; 
        YRange(y0, y1, node);  

        const char* fmt = "%7.2f" ;  
        canvas->drawf(   ix, -1, 0,0,  y_delta   , fmt); 
        canvas->drawf(   ix, -1, 0,2,  y1+y_delta, fmt  ); 
        canvas->drawf(   ix, -1, 0,3,  y0+y_delta, fmt ); 
    }
    else if(can_z)
    {
        double z_delta = getZ(node_) ;  
        double z0, z1 ; 
        ZRange(z0, z1, node);  

        // below is coercing double into int val 
        canvas->draw(   ix, -1, 0,0,  z_delta ); 
        canvas->draw(   ix, -1, 0,2,  z1+z_delta ); 
        canvas->draw(   ix, -1, 0,3,  z0+z_delta ); 
    }







}


/**
X4SolidTree::EntityTypeName
-----------------------

Unexpectedly G4 returns EntityType by value rather than by reference
so have to strdup to avoid corruption when the G4String goes out of scope. 

**/

const char* X4SolidTree::EntityTypeName(const G4VSolid* solid)   // static
{
    G4GeometryType type = solid->GetEntityType();  // G4GeometryType typedef for G4String
    return strdup(type.c_str()); 
}

const char* X4SolidTree::EntityTag(const G4VSolid* node_, bool move)   // static
{
    const G4VSolid*  node = move ? Moved(nullptr, nullptr, node_ ) : node_ ; 
    return EntityTag_(node); 
}

const char* X4SolidTree::G4Ellipsoid_          = "Ell" ; 
const char* X4SolidTree::G4Tubs_               = "Tub" ;
const char* X4SolidTree::G4Polycone_           = "Pol" ;
const char* X4SolidTree::G4Torus_              = "Tor" ;
const char* X4SolidTree::G4Box_                = "Box" ;
const char* X4SolidTree::G4Orb_                = "Orb" ;
const char* X4SolidTree::G4MultiUnion_         = "MUN" ;

const char* X4SolidTree::G4UnionSolid_         = "Uni" ;
const char* X4SolidTree::G4SubtractionSolid_   = "Sub" ;
const char* X4SolidTree::G4IntersectionSolid_  = "Int" ;
const char* X4SolidTree::G4DisplacedSolid_     = "Dis" ;


const char* X4SolidTree::DirtyEntityTag_( const G4VSolid* node )
{
    G4GeometryType type = node->GetEntityType();  // G4GeometryType typedef for G4String
    char* tag = strdup(type.c_str() + 2);  // +2 skip "G4"
    assert( strlen(tag) > 3 ); 
    tag[3] = '\0' ;
    return tag ;  
}

const char* X4SolidTree::EntityTag_( const G4VSolid* solid )
{
    int etype = EntityType(solid); 
    const char* s = nullptr ;
    switch(etype)
    {
       case _G4Ellipsoid:         s = G4Ellipsoid_         ; break ; 
       case _G4Tubs:              s = G4Tubs_              ; break ; 
       case _G4Polycone:          s = G4Polycone_          ; break ; 
       case _G4Torus:             s = G4Torus_             ; break ; 
       case _G4Box:               s = G4Box_               ; break ; 
       case _G4Orb:               s = G4Orb_               ; break ; 
       case _G4MultiUnion:        s = G4MultiUnion_        ; break ; 
 
       case _G4UnionSolid:        s = G4UnionSolid_        ; break ; 
       case _G4SubtractionSolid:  s = G4SubtractionSolid_  ; break ; 
       case _G4IntersectionSolid: s = G4IntersectionSolid_ ; break ; 
       case _G4DisplacedSolid:    s = G4DisplacedSolid_    ; break ; 
    }
    return s ; 
}
 
 


int X4SolidTree::EntityType(const G4VSolid* solid)   // static 
{
    G4GeometryType etype = solid->GetEntityType();  // G4GeometryType typedef for G4String
    const char* name = etype.c_str(); 

    int type = _G4Other ; 
    if( strcmp(name, "G4Ellipsoid") == 0 )         type = _G4Ellipsoid ; 
    if( strcmp(name, "G4Tubs") == 0 )              type = _G4Tubs ; 
    if( strcmp(name, "G4Polycone") == 0 )          type = _G4Polycone ; 
    if( strcmp(name, "G4Torus") == 0 )             type = _G4Torus ; 
    if( strcmp(name, "G4Box") == 0 )               type = _G4Box ; 
    if( strcmp(name, "G4Orb") == 0 )               type = _G4Orb ; 
    if( strcmp(name, "G4MultiUnion") == 0 )        type = _G4MultiUnion ; 

    if( strcmp(name, "G4UnionSolid") == 0 )        type = _G4UnionSolid ; 
    if( strcmp(name, "G4SubtractionSolid") == 0 )  type = _G4SubtractionSolid ; 
    if( strcmp(name, "G4IntersectionSolid") == 0 ) type = _G4IntersectionSolid ; 
    if( strcmp(name, "G4DisplacedSolid") == 0 )    type = _G4DisplacedSolid ; 
    return type ; 
}

std::string X4SolidTree::Desc(const G4VSolid* solid) // static
{
    std::stringstream ss ; 

    ss << EntityTypeName(solid)
       << " name " << solid->GetName()
       << " bool " << Boolean(solid)
       << " disp " << Displaced(solid)
       ; 

    std::string s = ss.str(); 
    return s ; 
}

bool X4SolidTree::Boolean(const G4VSolid* solid) // static
{
    return dynamic_cast<const G4BooleanSolid*>(solid) != nullptr ; 
}
bool X4SolidTree::Displaced(const G4VSolid* solid) // static
{
    return dynamic_cast<const G4DisplacedSolid*>(solid) != nullptr ; 
}
const G4VSolid* X4SolidTree::Left(const G4VSolid* solid ) // static
{
    return Boolean(solid) ? solid->GetConstituentSolid(0) : nullptr ; 
}
const G4VSolid* X4SolidTree::Right(const G4VSolid* solid ) // static
{
    return Boolean(solid) ? solid->GetConstituentSolid(1) : nullptr ; 
}
G4VSolid* X4SolidTree::Left_(G4VSolid* solid ) // static
{
    return Boolean(solid) ? solid->GetConstituentSolid(0) : nullptr ; 
}
G4VSolid* X4SolidTree::Right_(G4VSolid* solid ) // static
{
    return Boolean(solid) ? solid->GetConstituentSolid(1) : nullptr ; 
}

/**
X4SolidTree::Moved
---------------

When node isa G4DisplacedSolid sets the rotation and translation and returns the constituentMovedSolid
otherwise returns the input node.

**/
const G4VSolid* X4SolidTree::Moved( G4RotationMatrix* rot, G4ThreeVector* tla, const G4VSolid* node )  // static
{
    const G4DisplacedSolid* disp = dynamic_cast<const G4DisplacedSolid*>(node) ; 
    if(disp)
    {
        if(rot) *rot = disp->GetFrameRotation();
        if(tla) *tla = disp->GetObjectTranslation();  
    }
    return disp ? disp->GetConstituentMovedSolid() : node  ;
}

const G4VSolid* X4SolidTree::Moved( const G4VSolid* node )  // static
{
    const G4DisplacedSolid* disp = dynamic_cast<const G4DisplacedSolid*>(node) ; 
    return disp ? disp->GetConstituentMovedSolid() : node  ;
}

G4VSolid* X4SolidTree::Moved_( G4RotationMatrix* rot, G4ThreeVector* tla, G4VSolid* node )  // static
{
    G4DisplacedSolid* disp = dynamic_cast<G4DisplacedSolid*>(node) ; 
    if(disp)
    {
        if(rot) *rot = disp->GetFrameRotation();
        if(tla) *tla = disp->GetObjectTranslation();  
    }
    return disp ? disp->GetConstituentMovedSolid() : node  ;
}

G4VSolid* X4SolidTree::Moved_( G4VSolid* node )  // static
{
    G4DisplacedSolid* disp = dynamic_cast<G4DisplacedSolid*>(node) ; 
    return disp ? disp->GetConstituentMovedSolid() : node  ;
}




int X4SolidTree::maxdepth() const  
{
    return Maxdepth_r( root, 0 );  
}

int X4SolidTree::Maxdepth_r( const G4VSolid* node_, int depth)  // static 
{
    return Boolean(node_) ? std::max( Maxdepth_r(X4SolidTree::Left(node_), depth+1), Maxdepth_r(X4SolidTree::Right(node_), depth+1)) : depth ; 
}

/**
X4SolidTree::dumpTree
------------------
 
Postorder traversal of CSG tree
**/
void X4SolidTree::dumpTree(const char* msg ) const 
{
    std::cout << msg << " maxdepth(aka height) " << height << std::endl ; 
    dumpTree_r(root, 0 ); 
}

void X4SolidTree::dumpTree_r( const G4VSolid* node_, int depth  ) const
{
    if(Boolean(node_))
    {
        dumpTree_r(X4SolidTree::Left(node_) , depth+1) ; 
        dumpTree_r(X4SolidTree::Right(node_), depth+1) ; 
    }

    // postorder visit 

    G4RotationMatrix node_rot ;   // node (not tree) transforms
    G4ThreeVector    node_tla(0., 0., 0. ); 
    const G4VSolid*  node = Moved(&node_rot, &node_tla, node_ ); 
    assert( node ); 

    double zdelta_always_zero = getZ(node); 

    bool expect = zdelta_always_zero == 0. ; 
    if(!expect) exit(EXIT_FAILURE); 

    assert(expect ); 
    // Hmm thats tricky, using node arg always gives zero.
    // Must use node_ which is the one that might be G4DisplacedSolid. 

    double zdelta = getZ(node_) ;  

    std::cout 
        << " type " << std::setw(20) << EntityTypeName(node) 
        << " name " << std::setw(20) << node->GetName() 
        << " depth " << depth 
        << " zdelta "
        << std::fixed << std::setw(7) << std::setprecision(2) << zdelta
        << " node_tla (" 
        << std::fixed << std::setw(7) << std::setprecision(2) << node_tla.x() << " " 
        << std::fixed << std::setw(7) << std::setprecision(2) << node_tla.y() << " "
        << std::fixed << std::setw(7) << std::setprecision(2) << node_tla.z() << ")"
        ;

    if(X4SolidTree::CanZ(node))
    {
        double z0, z1 ; 
        ZRange(z0, z1, node);  
        std::cout 
            << " z1 " << std::fixed << std::setw(7) << std::setprecision(2) << z1
            << " z0 " << std::fixed << std::setw(7) << std::setprecision(2) << z0 
            << " az1 " << std::fixed << std::setw(7) << std::setprecision(2) << ( z1 + zdelta )
            << " az0 " << std::fixed << std::setw(7) << std::setprecision(2) << ( z0 + zdelta )
            ;
    }
    std::cout 
        << std::endl
        ; 
}

/**
X4SolidTree::classifyTree
----------------------

postorder traveral classifies every node doing bitwise-OR
combination of the child classifications.

**/

int X4SolidTree::classifyTree(double zcut)  
{
    crux->clear(); 
    if(verbose) std::cout << "X4SolidTree::classifyTree against zcut " << zcut  << std::endl ; 
    int zc = classifyTree_r(root, 0, zcut); 
    return zc ; 
}


int X4SolidTree::classifyTree_r(G4VSolid* node_, int depth, double zcut )
{
    int zcl = 0 ; 
    int sid = in(node_);    // inorder 
    int pos = post(node_); 

    if(Boolean(node_))
    {
        int left_zcl = classifyTree_r(Left_(node_) , depth+1, zcut) ; 
        int right_zcl = classifyTree_r(Right_(node_), depth+1, zcut) ; 

        zcl |= left_zcl ; 
        zcl |= right_zcl ; 


        if(left_zcl == INCLUDE && right_zcl == EXCLUDE )
        {
            crux->push_back(node_); 
            set_mkr( node_, 'X' ); 
        }
        else if(left_zcl == EXCLUDE && right_zcl == INCLUDE )
        {
            crux->push_back(node_); 
            set_mkr( node_, 'Y' ); 
        }

        if(false) std::cout 
            << "X4SolidTree::classifyTree_r" 
            << " sid " << std::setw(2) << sid
            << " pos " << std::setw(2) << pos
            << " left_zcl " << ClassifyMaskName(left_zcl)
            << " right_zcl " << ClassifyMaskName(right_zcl)
            << " zcl " << ClassifyMaskName(zcl)
            << std::endl
            ;

        set_zcls( node_, zcl ); 
    }
    else
    {
        // node_ is the raw one which may be G4DisplacedSolid
        double zd = getZ(node_) ;  
        G4VSolid* node = Moved_(node_); 
        bool can_z = CanZ(node) ; 
        assert(can_z); 
        if(can_z)
        {
            double z0, z1 ; 
            ZRange(z0, z1, node);  

            double az0 =  z0 + zd ; 
            double az1 =  z1 + zd ; 

            zcl = ClassifyZCut( az0, az1, zcut ); 

            if(verbose) std::cout 
                << "X4SolidTree::classifyTree_r"
                << " sid " << std::setw(2) << sid
                << " pos " << std::setw(2) << pos
                << " zd " << std::fixed << std::setw(7) << std::setprecision(2) << zd
                << " z1 " << std::fixed << std::setw(7) << std::setprecision(2) << z1
                << " z0 " << std::fixed << std::setw(7) << std::setprecision(2) << z0 
                << " az1 " << std::fixed << std::setw(7) << std::setprecision(2) << az1
                << " az0 " << std::fixed << std::setw(7) << std::setprecision(2) << az0 
                << " zcl " << ClassifyName(zcl)
                << std::endl
                ;

            set_zcls( node_, zcl );   // HMM node_ or node ?
        }
    }
    return zcl ; 
}

int X4SolidTree::classifyMask(const G4VSolid* top) const   // NOT USED ?
{
    return classifyMask_r(top, 0); 
}
int X4SolidTree::classifyMask_r( const G4VSolid* node_, int depth ) const 
{
    int mask = 0 ; 
    if(Boolean(node_))
    {
        mask |= classifyMask_r( Left(node_) , depth+1 ) ; 
        mask |= classifyMask_r( Right(node_), depth+1 ) ; 
    }
    else
    {
        mask |= zcls(node_) ; 
    }
    return mask ; 
}

/**
X4SolidTree::ClassifyZCut
------------------------

Inclusion status of solid with regard to a particular zcut::

                       --- 
                        .
                        .   EXCLUDE  : zcut entirely above the solid
                        .
                        .
      +---zd+z1----+   --- 
      |            |    .   
      | . zd . . . |    .   STRADDLE : zcut within z range of solid
      |            |    .
      +---zd+z0 ---+   ---
                        .
                        .   INCLUDE  : zcut fully below the solid 
                        .
                        .
                       ---  

**/

int X4SolidTree::ClassifyZCut( double az0, double az1, double zcut ) // static
{
    assert( az1 > az0 ); 
    int cls = UNDEFINED ; 
    if(       zcut <= az0 )              cls = INCLUDE ; 
    else if ( zcut < az1 && zcut > az0 ) cls = STRADDLE ; 
    else if ( zcut >= az1              ) cls = EXCLUDE ; 
    return cls ; 
}

const char* X4SolidTree::UNDEFINED_ = "UNDEFINED" ; 
const char* X4SolidTree::INCLUDE_   = "INCLUDE" ; 
const char* X4SolidTree::STRADDLE_  = "STRADDLE" ; 
const char* X4SolidTree::EXCLUDE_   = "EXCLUDE" ; 

const char* X4SolidTree::ClassifyName( int zcl ) // static 
{
    const char* s = nullptr ; 
    switch( zcl )
    {
        case UNDEFINED: s = UNDEFINED_ ; break ; 
        case INCLUDE  : s = INCLUDE_ ; break ; 
        case STRADDLE : s = STRADDLE_ ; break ; 
        case EXCLUDE  : s = EXCLUDE_ ; break ; 
    }
    return s ; 
}

const char* X4SolidTree::ClassifyMaskName( int zcl ) // static
{
    std::stringstream ss ; 

    if( zcl == UNDEFINED ) ss << UNDEFINED_[0] ; 
    if( zcl & INCLUDE )  ss << INCLUDE_[0] ; 
    if( zcl & STRADDLE ) ss << STRADDLE_[0] ; 
    if( zcl & EXCLUDE )  ss << EXCLUDE_[0] ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}

/**
X4SolidTree::apply_cut
------------------

See opticks/sysrap/tests/TreePruneTest.cc and opticks/sysrap/tests/tree.sh
for development of tree cutting and pruning.

Steps:

1. classify the nodes of the tree against the zcut 
2. change STRADDLE nodes params and transforms according to the zcut
3. reclassify against the zcut, so STRADDLE nodes should become INCLUDE nodes
4. edit the tree to remove the EXCLUDE nodes

**/

void X4SolidTree::apply_cut(double zcut)
{
    if(verbose)
    printf("X4SolidTree::apply_cut %7.2f \n", zcut );

    if(verbose)
    Draw(root, "X4SolidTree::apply_cut root before cut"); 

    unsigned pass = 0 ;
    unsigned maxpass = 10 ;

    while( root != nullptr && zcls(root) != INCLUDE && pass < maxpass )
    {
        classifyTree(zcut);   // set n.cls n.mkr

        cutTree_r(root, 0, zcut); 

        classifyTree(zcut);   // set n.cls n.mkr

        prune(false, pass);

        if(verbose)
        draw("X4SolidTree::apply_cut before prune", pass );

        prune(true, pass);

        classifyTree(zcut);
        instrumentTree();

        if(verbose)
        draw("tree::apply_cut after prune and re-classify", pass );

        pass++ ;
    }
}

void X4SolidTree::collectNames_inorder_r( const G4VSolid* n_, int depth )
{
    if(n_ == nullptr) return ; 
    const G4VSolid* n = Moved(nullptr, nullptr, n_) ;  
    collectNames_inorder_r(Left(n),  depth+1); 

    G4String name = n->GetName(); 
    names->push_back(name);  

    collectNames_inorder_r(Right(n), depth+1); 
}

void X4SolidTree::cutTree_r( const G4VSolid* node_, int depth, double zcut )
{
    if(Boolean(node_))
    {
        cutTree_r( X4SolidTree::Left(node_) , depth+1, zcut ) ; 
        cutTree_r( X4SolidTree::Right(node_), depth+1, zcut ) ; 
    }
    else
    {
        // get tree frame zcut into local frame of the node 
        double zdelta = getZ(node_) ; 
        double local_zcut = zcut - zdelta ; 

        int zcl = zcls(node_);   
        if( zcl == STRADDLE )
        {
            if(verbose) std::cout 
                << "X4SolidTree::cutTree_r"
                << " depth " << std::setw(2) << depth
                << " zdelta " << std::fixed << std::setw(10) << std::setprecision(4) << zdelta 
                << " zcut " << std::fixed << std::setw(10) << std::setprecision(4) << zcut
                << " local_zcut " << std::fixed << std::setw(10) << std::setprecision(4) << local_zcut
                << std::endl
                ;

            ApplyZCut( const_cast<G4VSolid*>(node_), local_zcut ); 
        } 
    }
}

void X4SolidTree::collectNodes( std::vector<const G4VSolid*>& nodes, const G4VSolid* top, int query_zcls  )
{
    collectNodes_r(nodes, top, 0, query_zcls);  
}

void X4SolidTree::collectNodes_r( std::vector<const G4VSolid*>& nodes, const G4VSolid* node_, int query_zcls, int depth  )
{
    if(Boolean(node_))
    {
        collectNodes_r( nodes, Left(node_) , query_zcls, depth+1 ) ; 
        collectNodes_r( nodes, Right(node_), query_zcls, depth+1 ) ; 
    }
    else
    {
        int zcl = zcls(node_) ; 
        if( zcl == query_zcls )
        {
            nodes.push_back(node_) ; // node_ or node ?
        } 
    }
} 

void X4SolidTree::ApplyZCut( G4VSolid* node_, double local_zcut ) // static
{
    G4VSolid* node = Moved_(node_ ); 
    if(verbose) std::cout << "X4SolidTree::ApplyZCut " << EntityTypeName(node) << std::endl ; 
    switch(EntityType(node))
    {
        case _G4Ellipsoid: ApplyZCut_G4Ellipsoid( node  , local_zcut);  break ; 
        case _G4Tubs:      ApplyZCut_G4Tubs(      node_ , local_zcut);  break ; // cutting tubs requires changing transform, hence node_
        case _G4Polycone:  ApplyZCut_G4Polycone(  node  , local_zcut);  break ; 
        default: 
        { 
            std::cout 
                << "X4SolidTree::ApplyZCut FATAL : not implemented for entityType " 
                << EntityTypeName(node) 
                << std::endl 
                ; 
            assert(0) ; 
        } ;
    }
}

/**
X4SolidTree::ApplyZCut_G4Ellipsoid
--------------------------------
     
::

    local                                             absolute 
    frame                                             frame    

    z1  +-----------------------------------------+    zd   
         \                                       /
           
            .                              .
    _________________________________________________ zcut 
                .                     .
                                 
    z0                 .      .   
                         
                                     
**/


void X4SolidTree::ApplyZCut_G4Ellipsoid( G4VSolid* node, double local_zcut)
{  
    G4Ellipsoid* ellipsoid =  dynamic_cast<G4Ellipsoid*>(node) ;  
    assert(ellipsoid); 

    double z0 = ellipsoid->GetZBottomCut() ; 
    double z1 = ellipsoid->GetZTopCut() ;
    
    double new_z0 = local_zcut ; 

    bool expect = new_z0 >= z0 && new_z0 < z1 && z1 > z0 ; 
    if(!expect) exit(EXIT_FAILURE); 
    assert( expect ); 

    double new_z1 = z1 ; 

    ellipsoid->SetZCuts( new_z0, new_z1 ); 
}

/**
X4SolidTree::ApplyZCut_G4Polycone_NotWorking
-----------------------------------------

Currently limited to only 2 Z-planes, 
to support more that 2 would need to delve 
into the r-z details which should be straightforward, 
just it is not yet implemented. 

SMOKING GUN : CHANGING OriginalParameters looks like
its working when just look at Opticks results but the 
change does not get thru to Geant4 


**/

void X4SolidTree::ApplyZCut_G4Polycone_NotWorking( G4VSolid* node, double local_zcut)
{  
    G4Polycone* polycone = dynamic_cast<G4Polycone*>(node) ;  
    assert(polycone); 
    G4PolyconeHistorical* pars = polycone->GetOriginalParameters(); 

    unsigned num_z = pars->Num_z_planes ; 
    for(unsigned i=1 ; i < num_z ; i++)
    {
        double z0 = pars->Z_values[i-1] ; 
        double z1 = pars->Z_values[i] ; 
        bool expect = z1 > z0 ; 
        assert(expect);   
        if(!expect) exit(EXIT_FAILURE); 
    }

    assert( num_z == 2 );    // simplifying assumption 
    pars->Z_values[0] = local_zcut ;   // get into polycone local frame

    polycone->SetOriginalParameters(pars); // OOPS : SEEMS Geant4 IGNORES
}


/**
X4SolidTree::ApplyZCut_G4Polycone
------------------------------

Use placement new to replace the polycone using the ctor again 
with different params so Geant4 cannot ignore 

Note that the radii is modified according to the cut, in 
order to cut without changing shape other than the cut.::

                                     
                                      .  (z1,r1)
                               .
                         .   
                   .
             .      (zc,rc)
       +    
     (z0,r0)

**/

void X4SolidTree::ApplyZCut_G4Polycone( G4VSolid* node, double local_zcut)
{  
    G4Polycone* polycone = dynamic_cast<G4Polycone*>(node) ;  
    assert(polycone); 

    G4PolyconeHistorical* pars = polycone->GetOriginalParameters(); 

    bool expect ; 
  
    unsigned num_z = pars->Num_z_planes ; 
    G4double* zp = new G4double[num_z] ; 
    G4double* ri = new G4double[num_z] ; 
    G4double* ro = new G4double[num_z] ; 

    for(unsigned i=0 ; i < num_z ; i++)
    {
        zp[i] = pars->Z_values[i] ; 
        ri[i] = pars->Rmin[i]; 
        ro[i] = pars->Rmax[i]; 
    }

    for(unsigned i=1 ; i < num_z ; i++)
    {
        double z0 = zp[i-1] ; 
        double z1 = zp[i] ; 
        expect = z1 > z0 ; 
        assert(expect);   
        if(!expect) exit(EXIT_FAILURE); 
    }

    assert( num_z == 2 );  // simplifying assumption 

    double zfrac = (local_zcut - zp[0])/(zp[1] - zp[0]) ;
 
    double ri_zcut = ri[0] + zfrac*( ri[1] - ri[0] ) ;   
    double ro_zcut = ro[0] + zfrac*( ro[1] - ro[0] ) ; 

    zp[0] = local_zcut ;   
    ri[0] = ri_zcut ; 
    ro[0] = ro_zcut ; 
   

    G4String name = polycone->GetName() ; 
    G4double startPhi = polycone->GetStartPhi() ;  
    G4double endPhi = polycone->GetEndPhi() ; 

    G4SolidStore::GetInstance()->DeRegister(polycone);  // avoids G4SolidStore segv at cleanup

    // placement new 
    G4Polycone* polycone_replacement = new (polycone) G4Polycone( name, startPhi, endPhi, num_z, zp, ri, ro ); 
    expect = polycone_replacement == polycone ;  
    assert( expect ); 
    if(!expect) exit(EXIT_FAILURE); 

}




/**
X4SolidTree::ApplyZCut_G4Tubs
--------------------------

* SEE THE TAIL OF THIS FILE FOR A DERIVATION OF new_hz AND zoffset 

G4Tubs is more difficult to cut than G4Polycone because it is 
symmetrically defined, so cutting requires shifting too. 
Replacing all G4Tubs with G4Polycone at the initial clone stage
avoids using this method at all. 

BUT if you prefer not to promote all G4Tubs to G4Polycone 
it is also possible to cut G4Tubs by changing the 
G4DisplacedSolid transforms appropriately. 

Initially tried using G4DisplacedSolid::SetObjectTranslation
but looking at the implementation that only changes one
of the two transforms held by the object. Hence used 
placement new yet again to rerun the ctor, after deregistering 
the disp solid from G4SolidStore.

NOTE THAT THE G4Tubs MUST HAVE AN ASSOCIATED TRANSFORM (EVEN IDENTITY MATRIX)
FOR THIS TO WORK : SO ALWAYS USE THE G4BooleanSolid ctor with rot and tla
  
**/

void X4SolidTree::ApplyZCut_G4Tubs( G4VSolid* node_ , double local_zcut )
{ 
    if( PROMOTE_TUBS_TO_POLYCONE )
        assert(0 && "All G4Tubs should have been promoted to G4Polycone at clone stage, how did you get here ?") ; 

    bool expect ; 
    G4RotationMatrix node_rot ; 
    G4ThreeVector    node_tla(0., 0., 0. ); 
    G4VSolid*  node = Moved_(&node_rot, &node_tla, node_ ); 

    G4Tubs* tubs = dynamic_cast<G4Tubs*>(node) ;  
    expect = tubs != nullptr ; 
    assert(expect); 
    if(!expect) exit(EXIT_FAILURE); 

    G4DisplacedSolid* disp = dynamic_cast<G4DisplacedSolid*>(node_) ; 
    expect = disp != nullptr ; // transform must be associated as must change offset to cut G4Tubs
    assert( expect ); 
    if(!expect) exit(EXIT_FAILURE); 

    double hz = tubs->GetZHalfLength() ; 
    double new_hz  = (hz - local_zcut)/2. ;  
    double zoffset = (hz + local_zcut)/2. ; 

    tubs->SetZHalfLength(new_hz);  
    node_tla.setZ( node_tla.z() + zoffset ); 

    std::cout 
        << "X4SolidTree::ApplyZCut_G4Tubs"
        << " hz " << hz
        << " local_zcut " << local_zcut
        << " new_hz " << new_hz
        << " zoffset " << zoffset 
        << std::endl 
        ;

    G4String disp_name = disp->GetName() ; 
    //disp->CleanTransformations() ; // needed ? perhaps avoids small leak 
    G4SolidStore::GetInstance()->DeRegister(disp);  // avoids G4SolidStore segv at cleanup

    G4DisplacedSolid* disp2 = new (disp) G4DisplacedSolid(disp_name, node, &node_rot, node_tla );  

    expect =  disp2 == disp ; 
    assert(expect); 
    if(!expect) exit(EXIT_FAILURE); 
}




/**
X4SolidTree::dumpUp
----------------

Ordinary postorder recursive traverse in order to get to all nodes. 
This approach should allow to obtain combination transforms in
complex trees. 

**/

void X4SolidTree::dumpUp(const char* msg) const 
{
    assert( parent_map ); 
    std::cout << msg << std::endl ; 
    dumpUp_r(root, 0); 
}

void X4SolidTree::dumpUp_r(const G4VSolid* node, int depth) const  
{
    if(Boolean(node))
    {
        dumpUp_r(X4SolidTree::Left(  node ), depth+1 ); 
        dumpUp_r(X4SolidTree::Right( node ), depth+1 ); 
    }
    else
    {
        G4RotationMatrix* tree_rot = nullptr ; 
        G4ThreeVector    tree_tla(0., 0., 0. ); 
        getTreeTransform(tree_rot, &tree_tla, node ); 

        const G4VSolid* nd = Moved(nullptr, nullptr, node); 
        std::cout 
            << "X4SolidTree::dumpUp_r" 
            << " depth " << depth 
            << " type " << std::setw(20) << EntityTypeName(nd) 
            << " name " << std::setw(20) << nd->GetName() 
            << " tree_tla (" 
            << std::fixed << std::setw(7) << std::setprecision(2) << tree_tla.x() << " " 
            << std::fixed << std::setw(7) << std::setprecision(2) << tree_tla.y() << " "
            << std::fixed << std::setw(7) << std::setprecision(2) << tree_tla.z() 
            << ")"
            << std::endl
            ; 
   }
}

/**
X4SolidTree::getTreeTransform
-------------------------------

Would normally use parent links to determine all transforms relevant to a node, 
but Geant4 boolean trees do not have parent links. 
Hence use an external parent_map to provide uplinks enabling iteration 
up the tree from any node up to the root. 

**/

void X4SolidTree::getTreeTransform( G4RotationMatrix* rot, G4ThreeVector* tla, const G4VSolid* node ) const 
{
    bool expect = rot == nullptr ; 
    assert(expect && "non null rotation not implemented"); 
    if(!expect) exit(EXIT_FAILURE); 

    const G4VSolid* nd = node ; 

    unsigned count = 0 ; 
    while(nd)
    {
        G4RotationMatrix r ; 
        G4ThreeVector    t(0., 0., 0. ); 
        const G4VSolid* dn = Moved( &r, &t, nd ); 
        assert( r.isIdentity() );  // simplifying assumption 

        *tla += t ;    // add up the translations 

        if(false) std::cout
            << "X4SolidTree::getTreeTransform" 
            << " count " << std::setw(2) << count 
            << " dn.name " << std::setw(20) << dn->GetName()
            << " dn.type " << std::setw(20) << dn->GetEntityType()
            << " dn.t (" 
            << std::fixed << std::setw(7) << std::setprecision(2) << t.x() << " " 
            << std::fixed << std::setw(7) << std::setprecision(2) << t.y() << " "
            << std::fixed << std::setw(7) << std::setprecision(2) << t.z() 
            << ")"
            << " tla (" 
            << std::fixed << std::setw(7) << std::setprecision(2) << tla->x() << " " 
            << std::fixed << std::setw(7) << std::setprecision(2) << tla->y() << " "
            << std::fixed << std::setw(7) << std::setprecision(2) << tla->z() 
            << ")"
            << std::endl 
            ; 

        nd = (*parent_map)[nd] ; // parentmap lineage uses G4DisplacedSolid so not using *dn* here
        count += 1 ; 
    }     
}

/**
X4SolidTree::DeepClone
-------------------

Clones a CSG tree of solids, assuming that the tree is
composed only of the limited set of primitives that are supported. 

G4BooleanSolid copy ctor just steals constituent pointers so 
it does not make an independent copy.  
Unlike the primitive copy ctors (at least those looked at: G4Polycone, G4Tubs) 
which appear to make properly independent copies 

**/

G4VSolid* X4SolidTree::DeepClone( const  G4VSolid* solid )  // static 
{
    G4RotationMatrix* rot = nullptr ; 
    G4ThreeVector* tla = nullptr ; 
    int depth = 0 ; 
    return DeepClone_r(solid, depth, rot, tla );  
}

/**
X4SolidTree::DeepClone_r
--------------------

G4DisplacedSolid is a wrapper for the right hand side boolean constituent 
which serves the purpose of holding the transform. The G4DisplacedSolid 
is automatically created by the G4BooleanSolid ctor when there is an associated transform.  

The below *rot* and *tla* look at first glance like they are not used. 
But look more closely, the recursive DeepClone_r calls within BooleanClone are using them 
across the generations. This structure is necessary for BooleanClone because the 
transform from the child is needed when cloning the parent.

**/

G4VSolid* X4SolidTree::DeepClone_r( const G4VSolid* node_, int depth, G4RotationMatrix* rot, G4ThreeVector* tla )  // static 
{
    const G4VSolid* node = Moved( rot, tla, node_ ); // if node_ isa G4DisplacedSolid node will not be and rot/tla will be set 

    if(verbose) 
    {
        std::cout 
            << "X4SolidTree::DeepClone_r(preorder visit)"
            << " type " << std::setw(20) << EntityTypeName(node)
            << " name " << std::setw(20) << node->GetName()
            << " depth " << std::setw(2) << depth
            ; 
        if(tla) std::cout 
            << " tla (" 
            << tla->x() 
            << " " 
            << tla->y() 
            << " " 
            << tla->z() 
            << ")"  
            ; 
        std::cout 
            << std::endl 
            ; 
    }

    G4VSolid* clone = Boolean(node) ? BooleanClone(node, depth, rot, tla ) : PrimitiveClone(node) ; 

    bool expect = clone != nullptr ; 
    if(!expect) std::cout << "X4SolidTree::DeepClone_r GOT null clone " << std::endl ; 
    assert(expect);
    if(!expect) exit(EXIT_FAILURE); 
    return clone ; 
}    

/**
X4SolidTree::BooleanClone
-----------------------

The left and right G4VSolid outputs from DeepClone_r will not be G4DisplacedSolid because
those get "dereferenced" by Moved and the rot/tla set.  This means that the information 
from the G4DisplacedSolid is available. This approach is necessary as the G4DisplacedSolid
is an "internal" object that the G4BooleanSolid ctor creates from the rot and tla ctor arguments. 

**/

G4VSolid* X4SolidTree::BooleanClone( const  G4VSolid* solid, int depth, G4RotationMatrix* rot, G4ThreeVector* tla ) // static
{
    if(verbose) std::cout << "X4SolidTree::BooleanClone" << std::endl ; 

    // HMM : rot and tla arguments were not used and they are not always null ... 
    // that suggests there is a lack of generality here for booleans of booleans etc... 
    // with translations that apply on top of translations. 
    // However for JUNO PMTs are not expecting any rotations
    // and are not expecting more than one level of translation. 

    G4ThreeVector zero(0., 0., 0.); 
    double epsilon = 1e-6 ; 

    bool expect_rot = rot == nullptr || rot->isIdentity() ;   
    if(!expect_rot) std::cout << "X4SolidTree::BooleanClone expect_rot ERROR " << std::endl ; 
    assert( expect_rot ); 
    if(!expect_rot) exit(EXIT_FAILURE); 

    bool expect_tla = tla == nullptr || tla->isNear(zero, epsilon) ; 
    if(!expect_tla) 
    {
        std::cout << "X4SolidTree::BooleanClone expect_tla ERROR (not expecting more than one level of translation) " << std::endl ; 
        if(tla) std::cout 
            << "X4SolidTree::BooleanClone" 
            << " tla( " 
            << tla->x() 
            << " " 
            << tla->y() 
            << " " 
            << tla->z() 
            << ") " 
            << std::endl
            ; 
    }
    assert( expect_tla ); 
    if(!expect_tla) exit(EXIT_FAILURE); 

    G4String name = solid->GetName() ; 
    G4RotationMatrix lrot, rrot ;  
    G4ThreeVector    ltra, rtra ; 

    const G4BooleanSolid* src_boolean = dynamic_cast<const G4BooleanSolid*>(solid) ; 
    G4VSolid* left  = DeepClone_r( src_boolean->GetConstituentSolid(0), depth+1, &lrot, &ltra ) ; 
    G4VSolid* right = DeepClone_r( src_boolean->GetConstituentSolid(1), depth+1, &rrot, &rtra ) ; 

    // not expecting left or right to be displaced   

    bool expect_left = dynamic_cast<const G4DisplacedSolid*>(left) == nullptr ;  
    assert( expect_left );
    if(!expect_left) std::cout << "X4SolidTree::BooleanClone expect_left ERROR " << std::endl ;
    if(!expect_left) exit(EXIT_FAILURE); 

    bool expect_right = dynamic_cast<const G4DisplacedSolid*>(right) == nullptr ; 
    assert( expect_right );
    if(!expect_right) std::cout << "X4SolidTree::BooleanClone expect_right ERROR " << std::endl ;
    if(!expect_right) exit(EXIT_FAILURE); 

    bool expect_lrot = lrot.isIdentity() ;  // lrot is expected to always be identity, as G4 never has left transforms
    assert( expect_lrot );
    if(!expect_lrot) std::cout << "X4SolidTree::BooleanClone expect_lrot ERROR " << std::endl ;
    if(!expect_lrot) exit(EXIT_FAILURE); 

    bool expect_ltra = ltra.x() == 0. && ltra.y() == 0. && ltra.z() == 0. ; // not expecting translations on the left
    assert( expect_ltra );
    if(!expect_ltra) std::cout << "X4SolidTree::BooleanClone expect_ltra ERROR " << std::endl ;
    if(!expect_ltra) exit(EXIT_FAILURE); 

    bool expect_rrot = rrot.isIdentity() ; // rrot identity is a simplifying assumption
    assert( expect_rrot );
    if(!expect_rrot) std::cout << "X4SolidTree::BooleanClone expect_rrot ERROR " << std::endl ;
    if(!expect_rrot) exit(EXIT_FAILURE); 


    G4VSolid* clone = nullptr ; 
    switch(EntityType(solid))
    {
        case _G4UnionSolid        : clone = new G4UnionSolid(       name, left, right, &rrot, rtra ) ; break ; 
        case _G4SubtractionSolid  : clone = new G4SubtractionSolid( name, left, right, &rrot, rtra ) ; break ;
        case _G4IntersectionSolid : clone = new G4IntersectionSolid(name, left, right, &rrot, rtra ) ; break ; 
    } 
    CheckBooleanClone( clone, left, right ); 
    return clone ; 
}

void X4SolidTree::CheckBooleanClone( const G4VSolid* clone, const G4VSolid* left, const G4VSolid* right ) // static
{
    bool expect ; 

    if(!clone) std::cout << "X4SolidTree::CheckBooleanClone FATAL " << std::endl ; 

    expect = clone != nullptr ; 
    assert(expect);
    if(!expect) exit(EXIT_FAILURE); 
 
    const G4BooleanSolid* boolean = dynamic_cast<const G4BooleanSolid*>(clone) ; 

    // lhs is never wrapped in G4DisplacedSolid 
    const G4VSolid* lhs = boolean->GetConstituentSolid(0) ; 
    const G4DisplacedSolid* lhs_disp = dynamic_cast<const G4DisplacedSolid*>(lhs) ; 
    expect = lhs_disp == nullptr && lhs == left ; 
    assert(expect) ;      
    if(!expect) exit(EXIT_FAILURE); 

    // rhs will be wrapped in G4DisplacedSolid as above G4BooleanSolid ctor has transform rrot/rtla
    const G4VSolid* rhs = boolean->GetConstituentSolid(1) ; 
    const G4DisplacedSolid* rhs_disp = dynamic_cast<const G4DisplacedSolid*>(rhs) ; 

    expect = rhs_disp != nullptr && rhs != right ;  
    assert(expect);    
    if(!expect) exit(EXIT_FAILURE); 

    const G4VSolid* right_check = rhs_disp->GetConstituentMovedSolid() ;
    expect = right_check == right ; 

    assert(expect);    
    if(!expect) exit(EXIT_FAILURE); 
}

void X4SolidTree::GetBooleanBytes(char** bytes, int& num_bytes, const G4VSolid* solid ) // static
{
    int type = EntityType(solid); 
    switch(type)
    {
        case _G4UnionSolid        : num_bytes = sizeof(G4UnionSolid)        ; break ; 
        case _G4SubtractionSolid  : num_bytes = sizeof(G4SubtractionSolid)  ; break ; 
        case _G4IntersectionSolid : num_bytes = sizeof(G4IntersectionSolid) ; break ; 
    } 

    *bytes = new char[num_bytes]; 
    memcpy( *bytes, (char*) solid, num_bytes ); 
}

int X4SolidTree::CompareBytes( char* bytes0, char* bytes1, int num_bytes ) // static
{
    int mismatch = 0 ; 
    for(int i=0 ; i < num_bytes ; i++ ) 
    {   
        if( bytes0[i] != bytes1[i] ) 
        {   
            printf("mismatch %5d : %3d : %3d  \n", i, int(bytes0[i]), int(bytes1[i]) ); 
            mismatch++ ;   
        }   
    }   
    printf("mismatch %d\n", mismatch); 
    return mismatch ; 
}

/**
X4SolidTree::PlacementNewDupe
--------------------------

Technical test that should do nothing and in the process
demonstrate that can use placement new to replace a boolean solid 
object with a bit perfect duplicate in the same memory address. 

1. access and hold innards of the solid in local scope
2. de-registers the solid from G4SolidStore
3. placement new instantiate replacement object at the same memory address 
   putting the pieces back together again 

The motivation for this is that tree pruning needs to be able 
to SetRight/SetLeft but G4BooleanSolid has no such methods. 
So in order to workaround this can use placement new to construct 
a duplicate object or one with some inputs changed.

G4VSolid registers/deregisters with G4SolidStore in ctor/dtor.
This means that using placement new to replace a solid causes memory 
corruption with::

    malloc: *** error for object 0x7f888c5084b0: pointer being freed was not allocated

Avoided this problem by manually deregistering from G4SolidStore
Although G4VSolid dtor deregisters it would be wrong to delete 
as do not want to deallocate the memory, want to reuse it.

When using the no transform ctor the dupe is a bit perfect match. 
With the transform ctor see ~2 bytes different from fPtrSolidB.
Revealed using dirty offsetof(G4UnionSolid, member) checks using:: 

   #define private   public
   #define protected public

That is not a problem, just a little leak. 
The reason is that the transform ctor allocates, so will normally get a different fPtrSolidB::

     77 G4BooleanSolid::G4BooleanSolid( const G4String& pName,
     78                                       G4VSolid* pSolidA ,
     79                                       G4VSolid* pSolidB ,
     80                                       G4RotationMatrix* rotMatrix,
     81                                 const G4ThreeVector& transVector    ) :
     82   G4VSolid(pName), fStatistics(1000000), fCubVolEpsilon(0.001),
     83   fAreaAccuracy(-1.), fCubicVolume(-1.), fSurfaceArea(-1.),
     84   fRebuildPolyhedron(false), fpPolyhedron(0), fPrimitivesSurfaceArea(0.),
     85   createdDisplacedSolid(true)
     86 {
     87   fPtrSolidA = pSolidA ;
     88   fPtrSolidB = new G4DisplacedSolid("placedB",pSolidB,rotMatrix,transVector) ;
     89 }

**/

void X4SolidTree::PlacementNewDupe( G4VSolid* solid) // static
{
    G4BooleanSolid* src = dynamic_cast<G4BooleanSolid*>(solid) ; 
    assert( src ); 

    G4String name = src->GetName() ; 
    G4VSolid* left = src->GetConstituentSolid(0) ; 
    G4VSolid* disp_right = src->GetConstituentSolid(1) ;  // may be G4DisplacedSolid

    G4RotationMatrix rrot ; 
    G4ThreeVector    rtla ; 
    G4VSolid* right = Moved_( &rrot, &rtla, disp_right ); 
    int type = EntityType(solid); 

    G4SolidStore::GetInstance()->DeRegister(solid);

    G4VSolid* dupe = nullptr ; 
    switch(type)
    {
        case _G4UnionSolid        : dupe = new (solid) G4UnionSolid(        name, left, right, &rrot, rtla ) ; break ; 
        case _G4SubtractionSolid  : dupe = new (solid) G4SubtractionSolid(  name, left, right, &rrot, rtla ) ; break ;
        case _G4IntersectionSolid : dupe = new (solid) G4IntersectionSolid( name, left, right, &rrot, rtla ) ; break ; 
    } 
    bool expect = dupe == solid ; 
    assert(expect); 
    if(!expect) exit(EXIT_FAILURE); 

} 

/**
X4SolidTree::SetRight
-----------------

The right hand side CSG constituent of *node* is changed to the provided *right* G4VSolid, 
with the transform rotation/translation applied to the *right* solid.
Use nullptr for rrot or rtla when no rotation or translation is required.

As there is no G4BooleanSolid::SetConstituentSolid method this sneakily replaces *node* with 
another at the same memory addess (using placement new) with the *right* changed by re-construction.

Note that it is not appropriate to use a G4DisplacedSolid argument for either
*solid* or *right* as G4DisplacedSolid objects are internal implementation details of 
G4BooleanSolid that automatically gets instanciated by the G4BooleanSolid ctors 
with transform arguments. So that means that G4DisplacedSolid are not suitable inputs
to G4BooleanSolid ctors. 

**/

void X4SolidTree::SetRight(  G4VSolid* node, G4VSolid* right, G4RotationMatrix* rrot, G4ThreeVector* rtla )
{
    assert( dynamic_cast<G4DisplacedSolid*>(node) == nullptr ) ; 
    assert( dynamic_cast<G4DisplacedSolid*>(right) == nullptr ) ; 

    G4BooleanSolid* src = dynamic_cast<G4BooleanSolid*>(node) ; 
    assert( src ); 

    int type = EntityType(src); 
    G4String name = src->GetName() ; 
    G4VSolid* left = src->GetConstituentSolid(0) ; 

    G4SolidStore::GetInstance()->DeRegister(src);

    G4VSolid* replacement = nullptr ; 

    G4ThreeVector tlate(0.,0.,0.); 
    if( rtla == nullptr ) rtla = &tlate ; 
    switch(type)
    {
        case _G4UnionSolid        : replacement = new (src) G4UnionSolid(        name, left, right, rrot, *rtla ) ; break ; 
        case _G4SubtractionSolid  : replacement = new (src) G4SubtractionSolid(  name, left, right, rrot, *rtla ) ; break ;
        case _G4IntersectionSolid : replacement = new (src) G4IntersectionSolid( name, left, right, rrot, *rtla ) ; break ; 
    }

    bool expect = replacement == src ;
    assert(expect); 
    if(!expect) exit(EXIT_FAILURE); 
}

/**
X4SolidTree::SetLeft
-------------------

The lefthand side constituent of *node* is changed to *left* 
This is implemented using placement new trickery to replace the 
*node* with another at the same memory address with changed *left*.

**/

void X4SolidTree::SetLeft(  G4VSolid* node, G4VSolid* left)  // static 
{
    assert( dynamic_cast<G4DisplacedSolid*>(node) == nullptr ) ; 
    assert( dynamic_cast<G4DisplacedSolid*>(left) == nullptr ) ; 

    G4BooleanSolid* src = dynamic_cast<G4BooleanSolid*>(node) ; 
    assert( src ); 

    int type = EntityType(src); 
    G4String name = src->GetName() ; 
    G4VSolid* disp_right = src->GetConstituentSolid(1) ;  // may be G4DisplacedSolid

    G4RotationMatrix rrot ; 
    G4ThreeVector    rtla(0.,0.,0.) ; 
    G4VSolid* right = Moved_( &rrot, &rtla, disp_right );  // if *right* isa G4DisplacedSolid get the moved solid inside and transforms
 
    G4SolidStore::GetInstance()->DeRegister(src);

    G4VSolid* replacement = nullptr ; 
    switch(type)
    {
        case _G4UnionSolid        : replacement = new (src) G4UnionSolid(        name, left, right, &rrot, rtla ) ; break ; 
        case _G4SubtractionSolid  : replacement = new (src) G4SubtractionSolid(  name, left, right, &rrot, rtla ) ; break ;
        case _G4IntersectionSolid : replacement = new (src) G4IntersectionSolid( name, left, right, &rrot, rtla ) ; break ; 
    } 
    bool expect = replacement == src ; 
    assert(expect); 
    if(!expect) exit(EXIT_FAILURE); 
}


G4VSolid* X4SolidTree::PrimitiveClone( const  G4VSolid* solid )  // static 
{
    G4VSolid* clone = nullptr ; 
    int type = EntityType(solid); 
    if( type == _G4Ellipsoid )
    {
        const G4Ellipsoid* ellipsoid = dynamic_cast<const G4Ellipsoid*>(solid) ; 
        clone = new G4Ellipsoid(*ellipsoid) ;
    }
    else if( type == _G4Tubs && PROMOTE_TUBS_TO_POLYCONE == false)
    {
        const G4Tubs* tubs = dynamic_cast<const G4Tubs*>(solid) ; 
        clone = new G4Tubs(*tubs) ;  
    }
    else if( type == _G4Tubs && PROMOTE_TUBS_TO_POLYCONE == true)
    {
        clone = PromoteTubsToPolycone( solid );  
    }
    else if( type == _G4Polycone )
    {
        const G4Polycone* polycone = dynamic_cast<const G4Polycone*>(solid) ; 
        clone = new G4Polycone(*polycone) ;  
    }
    else if( type == _G4Torus )
    {
        const G4Torus* torus = dynamic_cast<const G4Torus*>(solid) ; 
        clone = new G4Torus(*torus) ;  
    }
    else if( type == _G4Box )
    {
        const G4Box* box = dynamic_cast<const G4Box*>(solid) ; 
        clone = new G4Box(*box) ;  
    }
    else if( type == _G4Orb )
    {
        const G4Orb* orb = dynamic_cast<const G4Orb*>(solid) ; 
        clone = new G4Orb(*orb) ;  
    }
    else if( type == _G4MultiUnion )
    {
        const G4MultiUnion* mun = dynamic_cast<const G4MultiUnion*>(solid) ; 
        clone = new G4MultiUnion(*mun) ;  
    }
    else
    {
        std::cout 
            << "X4SolidTree::PrimitiveClone FATAL unimplemented prim type " << EntityTypeName(solid) 
            << std::endl 
            ;
        assert(0); 
    } 
    return clone ; 
}


G4VSolid* X4SolidTree::PrimitiveClone( const  G4VSolid* solid, const char* name )  // static 
{
    G4VSolid* clone = PrimitiveClone(solid); 
    clone->SetName(name); 
    return clone ; 
}


//const bool X4SolidTree::PROMOTE_TUBS_TO_POLYCONE = true ; 
const bool X4SolidTree::PROMOTE_TUBS_TO_POLYCONE = false ; 

G4VSolid* X4SolidTree::PromoteTubsToPolycone( const G4VSolid* solid ) // static
{
    const G4Tubs* tubs = dynamic_cast<const G4Tubs*>(solid) ; 
    assert(tubs); 

    G4String name = tubs->GetName(); 
    double dz = tubs->GetDz(); 
    double rmin = tubs->GetRMin(); 
    double rmax = tubs->GetRMax(); 
    double sphi = tubs->GetSPhi(); 
    double dphi = tubs->GetDPhi(); 
 
    G4int numZPlanes = 2 ; 
    G4double zPlane[] = { -dz    , dz } ;   
    G4double rInner[] = {  rmin  , rmin   } ;   
    G4double rOuter[] = {  rmax  , rmax } ;    

    G4Polycone* polycone = new G4Polycone(
                               name,
                               sphi,
                               dphi,
                               numZPlanes,
                               zPlane,
                               rInner,
                               rOuter
                               );  

    G4VSolid* clone = polycone ; 
    return clone ; 
}


double X4SolidTree::getX( const G4VSolid* node ) const
{
    G4RotationMatrix* tree_rot = nullptr ; 
    G4ThreeVector    tree_tla(0., 0., 0. ); 
    getTreeTransform(tree_rot, &tree_tla, node ); 

    double x_delta = tree_tla.x() ; 
    return x_delta ; 
}

double X4SolidTree::getY( const G4VSolid* node ) const
{
    G4RotationMatrix* tree_rot = nullptr ; 
    G4ThreeVector    tree_tla(0., 0., 0. ); 
    getTreeTransform(tree_rot, &tree_tla, node ); 

    double y_delta = tree_tla.y() ; 
    return y_delta ; 
}

double X4SolidTree::getZ( const G4VSolid* node ) const
{
    G4RotationMatrix* tree_rot = nullptr ; 
    G4ThreeVector    tree_tla(0., 0., 0. ); 
    getTreeTransform(tree_rot, &tree_tla, node ); 

    double z_delta = tree_tla.z() ; 
    return z_delta ; 
}






bool X4SolidTree::CanX(  const G4VSolid* solid ) 
{
    int type = EntityType(solid) ; 
    return type == _G4Box ; 
}
bool X4SolidTree::CanY(  const G4VSolid* solid ) 
{
    int type = EntityType(solid) ; 
    return type == _G4Box ; 
}
bool X4SolidTree::CanZ( const G4VSolid* solid ) // static
{
    int type = EntityType(solid) ; 
    bool can = type == _G4Ellipsoid || type == _G4Tubs || type == _G4Polycone || type == _G4Torus || type == _G4Box ; 
    G4String name = solid->GetName(); 

    if( can == false && verbose )
    {
        std::cout 
            << "X4SolidTree::CanZ ERROR "
            << " false for solid.EntityTypeName " << EntityTypeName(solid)
            << " solid.name " << name.c_str() 
            << std::endl ; 
            ;
    }

    return can ; 
}


void X4SolidTree::XRange( double& x0, double& x1, const G4VSolid* solid) // static  
{
    switch(EntityType(solid))
    {
        case _G4Box:       GetXRange( dynamic_cast<const G4Box*>(solid),       x0, x1 );  break ; 
        case _G4Other:    { std::cout << "X4SolidTree::GetX FATAL : not implemented for entityType " << EntityTypeName(solid) << std::endl ; assert(0) ; } ; break ;  
    }
}
void X4SolidTree::YRange( double& y0, double& y1, const G4VSolid* solid) // static  
{
    switch(EntityType(solid))
    {
        case _G4Box:       GetYRange( dynamic_cast<const G4Box*>(solid),       y0, y1 );  break ; 
        case _G4Other:    { std::cout << "X4SolidTree::GetY FATAL : not implemented for entityType " << EntityTypeName(solid) << std::endl ; assert(0) ; } ; break ;  
    }
}
void X4SolidTree::ZRange( double& z0, double& z1, const G4VSolid* solid) // static  
{
    switch(EntityType(solid))
    {
        case _G4Ellipsoid: GetZRange( dynamic_cast<const G4Ellipsoid*>(solid), z0, z1 );  break ; 
        case _G4Tubs:      GetZRange( dynamic_cast<const G4Tubs*>(solid)    ,  z0, z1 );  break ; 
        case _G4Polycone:  GetZRange( dynamic_cast<const G4Polycone*>(solid),  z0, z1 );  break ; 
        case _G4Torus:     GetZRange( dynamic_cast<const G4Torus*>(solid),     z0, z1 );  break ; 
        case _G4Box:       GetZRange( dynamic_cast<const G4Box*>(solid),       z0, z1 );  break ; 
        case _G4Other:    { std::cout << "X4SolidTree::GetZ FATAL : not implemented for entityType " << EntityTypeName(solid) << std::endl ; assert(0) ; } ; break ;  
    }
}



void X4SolidTree::GetXRange( const G4Box* const box, double& _x0, double& _x1 )  // static 
{
    _x1 = box->GetXHalfLength() ; 
    _x0 = -_x1 ; 
    assert( _x1 > 0. ); 
}
void X4SolidTree::GetYRange( const G4Box* const box, double& _y0, double& _y1 )  // static 
{
    _y1 = box->GetYHalfLength() ; 
    _y0 = -_y1 ; 
    assert( _y1 > 0. ); 
}
void X4SolidTree::GetZRange( const G4Box* const box, double& _z0, double& _z1 )  // static 
{
    _z1 = box->GetZHalfLength() ; 
    _z0 = -_z1 ; 
    assert( _z1 > 0. ); 
}




void X4SolidTree::GetZRange( const G4Ellipsoid* const ellipsoid, double& _z0, double& _z1 )  // static 
{
    _z1 = ellipsoid->GetZTopCut() ; 
    _z0 = ellipsoid->GetZBottomCut() ;  
}
void X4SolidTree::GetZRange( const G4Tubs* const tubs, double& _z0, double& _z1 )  // static 
{
    _z1 = tubs->GetZHalfLength() ;  
    _z0 = -_z1 ;  
    assert( _z1 > 0. ); 
}
void X4SolidTree::GetZRange( const G4Polycone* const polycone, double& _z0, double& _z1 )  // static 
{
    G4PolyconeHistorical* pars = polycone->GetOriginalParameters(); 
    unsigned num_z = pars->Num_z_planes ; 
    for(unsigned i=1 ; i < num_z ; i++)
    {
        double z0 = pars->Z_values[i-1] ; 
        double z1 = pars->Z_values[i] ; 
        bool expect = z1 > z0 ; 
        assert(expect);   
        if(!expect) exit(EXIT_FAILURE); 
    }
    _z1 = pars->Z_values[num_z-1] ; 
    _z0 = pars->Z_values[0] ;  
}
void X4SolidTree::GetZRange( const G4Torus* const torus, double& _z0, double& _z1 )  // static 
{
    G4double rmax = torus->GetRmax() ; 
    _z1 = rmax ; 
    _z0 = -rmax ;  
}



/**
X4SolidTree::ApplyZCut_G4Tubs
----------------------------

An alternative to using this is to promote all G4Tubs to G4Polycone at the clone stage,  
avoiding the need to change boolean displacements as G4Polycone is not symmetrically defined. 

* INITIALLY HAD PROBLEM WITH THE IMPLEMENTATION : CHANGING OFFSETS DID NOT WORK
* SOLVED WITH PLACEMENT NEW APPLIED TO THE G4DisplacedSolid 

Cutting G4Tubs::


     zd+hz  +---------+               +---------+     new_zd + new_hz
            |         |               |         |  
            |         |               |         |
            |         |               |         |
            |         |             __|_________|__   new_zd
            |         |               |         |
     zd   --|---------|--             |         |
            |         |               |         |
            |         |               |         |
         .  | . . . . | . .zcut . . . +---------+ . . new_zd - new_hz  . . . . . .
            |         | 
            |         |
    zd-hz   +---------+ 


     cut position

          zcut = new_zd - new_hz 
          new_zd = zcut + new_hz  


     original height:  2*hz                         
      
     cut height :     

          loc_zcut = zcut - zd 

          2*new_hz = 2*hz - (zcut-(zd-hz)) 

                   = 2*hz - ( zcut - zd + hz )

                   = 2*hz -  zcut + zd - hz 

                   = hz + zd - zcut 

                   = hz - (zcut - zd)             

                   = hz - loc_zcut 


                                    hz + zd - zcut
                        new_hz =  -----------------
                                         2

                                    hz - loc_zcut 
                        new_hz =  --------------------      new_hz( loc_zcut:-hz ) = hz     unchanged
                                          2                 new_hz( loc_zcut:0   ) = hz/2   halved
                                                            new_hz( loc_zcut:+hz ) =  0     made to disappear 



Simpler way to derive the same thing, is to use the initial local frame::


       +hz  +---------+               +---------+     zoff + new_hz
            |         |               |         |  
            |         |               |         |
            |         |               |         |
            |         |             __|_________|__   zoff 
            |         |               |         |
        0 --|---------|- . . . . . . . . . . . . . . . 0 
            |         |               |         |
            |         |               |         |
   loc_zcut | . . . . | . . . . .  .  +---------+ . . zoff - new_hz  . . . . . .
            |         | 
            |         |
      -hz   +---------+. . . . . . . . . . . . 



            loc_zcut = zoff - new_hz

                zoff = loc_zcut + new_hz 

                     = loc_zcut +  (hz - loc_zcut)/2

                     =  2*loc_zcut + (hz - loc_zcut)
                        ------------------------------
                                   2

               zoff  =  loc_zcut + hz                   zoff( loc_zcut:-hz ) = 0      unchanged
                        ---------------                 zoff( loc_zcut:0   ) = hz/2   
                              2                         zoff( loc_zcut:+hz ) = hz     makes sense, 
                                                                                      think about just before it disappears


Simpler way,  notice the top and bottom line equations, add or subtract and rearrange::

            hz       = zoff + new_hz

            loc_zcut = zoff - new_hz


      Add them:

            hz + loc_zcut = 2*zoff 

                   zoff =  hz + loc_zcut
                           --------------
                                 2 
      Subtract them:

               hz - loc_zcut = 2*new_hz

                   new_hz = hz - loc_zcut 
                            ---------------
                                 2

**/

