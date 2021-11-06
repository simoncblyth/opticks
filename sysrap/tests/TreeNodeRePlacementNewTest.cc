// name=TreeNodeRePlacementNewTest ; gcc $name.cc -I. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <cassert>
#include <cstring>
#include <cstdio>
#include <vector>
#include <map>

#include "SCanvas.hh"


enum { 
  UNDEFINED = 0, 
  INCLUDE   = 1<<0 , 
  EXCLUDE   = 1<<1 ,
  MIXED     = INCLUDE | EXCLUDE    
};  

char pcls( int cls ) // static
{
    char pcl = '_' ; 
    switch( cls )
    {
        case UNDEFINED: pcl = ' ' ; break ; 
        case INCLUDE:   pcl = ' ' ; break ; 
        case EXCLUDE:   pcl = 'E' ; break ; 
        case MIXED:     pcl = 'X' ; break ; 
        default:        pcl = '?' ; break ; 
    } 
    return pcl ; 
}

struct nd 
{
    int value ; 
    nd* left ; 
    nd* right ;    

    nd( int value, nd* left, nd* right ); 

    const char* desc() const ; 
    bool is_prim() const ; 
    bool is_boolean() const ; 
    bool is_bileaf() const ; 
    bool is_mixed() const ; 
    bool is_cut_right() const ; 
    bool is_cut_left() const ; 
    bool is_crux() const ; 

    int cls ; 
    int depth ; 
    int in ; 
    int pre ; 
    char mkr ; 

}; 

nd::nd(int value_, nd* left_, nd* right_)
    :
    value(value_), 
    left(left_),
    right(right_),
    cls(UNDEFINED),
    depth(UNDEFINED),
    in(UNDEFINED),
    pre(UNDEFINED),
    mkr(' ')
{
}

const char* nd::desc() const 
{
    char tmp[30] ; 
    snprintf(tmp, 30, "%d:%c:%c", value, pcls(cls), mkr );  
    return strdup(tmp); 
}

bool nd::is_prim()   const {  return left == nullptr && right == nullptr ; }
bool nd::is_boolean() const { return left != nullptr && right != nullptr ; }
bool nd::is_bileaf() const {  return is_boolean() && left->is_prim() && right->is_prim() ; }
bool nd::is_mixed() const {   return cls == MIXED ; }
bool nd::is_cut_right() const {   return is_boolean() && left->cls == INCLUDE && right->cls == EXCLUDE ; }
bool nd::is_cut_left()  const {   return is_boolean() && left->cls == EXCLUDE && right->cls == INCLUDE ; }
bool nd::is_crux() const {        return is_cut_right() ^ is_cut_left() ; }  // XOR : is_cut_right OR is_cut_left BUT not both 

struct tree
{
    static int NumNode(int height); 

    bool verbose ; 
    int count ; 
    int height ; 
    nd* root ; 

    std::vector<nd*> inorder ; 
    std::vector<nd*> preorder ; 
    std::vector<nd*> crux ; 
    std::map<nd*, nd*> parentmap ; 

    int width ; 
    SCanvas* canvas ; 

    tree( int height_ ); 

    nd* build_r(int h); 
    void initvalue_r( nd* n ); 
     
    void instrument();
    void inorder_r( nd* n ); 
    void preorder_r( nd* n ); 
    void parent_r( nd* n, int depth ); 
    void depth_r( nd* n, int depth ); 
    void clear_mkr(); 
    void clear_mkr_r( nd* n ); 

    int num_prim() const ; 
    int num_prim_r( nd* n ) const ; 

    int num_node(int cls) const ; 
    int num_node_r( nd* n, int cls) const ; 
    const char* desc() const ; 

    nd* parent( nd* n ) const ;

    void dump(const char* msg) const ; 
    void dump_r( nd* n ) const ; 

    int maxdepth_r(nd* n, int depth) const ;
    int maxdepth() const  ;

    void draw(const char* msg=nullptr, int meta=-1); 
    void draw_r( nd* n ); 

    void apply_cut(int cut);

    void classify( int cut );
    int classify_r( nd* n, int cut );

    void prune( bool act ); 
    void prune( nd* n, bool act ); 


};


int tree::NumNode(int height){ return (1 << (height+1)) - 1  ; } // static

tree::tree(int height_)
    :
    verbose(false),
    count(0),
    height(height_),
    root(build_r(height)),
    width(count),
    canvas(new SCanvas(width,height,5,3))
{
    instrument();
    initvalue_r(root);   // must be after instrument, as uses n.pre
}
nd* tree::build_r(int h)
{
    nd* l = h > 0 ? build_r( h-1) : nullptr ;  
    nd* r = h > 0 ? build_r( h-1) : nullptr ;  
    return new nd( count++, l, r) ; 
}
void tree::initvalue_r( nd* n )
{
    if( n == nullptr ) return ; 
    initvalue_r( n->left ); 
    initvalue_r( n->right ); 
    n->value = n->pre ; 
}


int tree::maxdepth_r(nd* n, int depth) const 
{
    return n->left && n->right ? std::max( maxdepth_r( n->left, depth+1), maxdepth_r(n->right, depth+1)) : depth ; 
}
int tree::maxdepth() const  
{
    return maxdepth_r( root, 0 ); 
}

int tree::num_prim_r(nd* n) const 
{
    if(!n) return 0 ; 
    return ( n->left && n->right ) ? num_prim_r( n->left) + num_prim_r(n->right) : 1 ; 
}
int tree::num_prim() const 
{
    return num_prim_r(root); 
}

int tree::num_node_r(nd* n, int qcls) const 
{
    int num = ( n && n->cls == qcls ) ? 1 : 0 ; 
    if( n && n->left && n->right )
    { 
        num += num_node_r( n->left,  qcls ); 
        num += num_node_r( n->right, qcls ); 
    }
    return num ; 
}
int tree::num_node(int cls) const 
{
    return num_node_r(root, cls); 
}

const char* tree::desc() const 
{
    char tmp[100]; 
    int num_undefined = num_node(UNDEFINED);  
    int num_exclude   = num_node(EXCLUDE);  
    int num_include   = num_node(INCLUDE);  
    int num_mixed     = num_node(MIXED);  
    int num_prim_     = num_prim(); 

    snprintf(tmp, 100, "UN:%d EX:%d IN:%d MX:%d prim:%d",num_undefined, num_exclude, num_include, num_mixed, num_prim_ ); 
    return strdup(tmp); 
}


void tree::instrument()
{
    if(!root) return ; 
    clear_mkr();   

    inorder.clear();
    inorder_r(root); 

    preorder.clear();
    preorder_r(root); 

    parentmap.clear();
    parentmap[root] = nullptr ; 
    parent_r(root, 0 ); 

    depth_r(root, 0); 
}

nd* tree::parent( nd* n ) const 
{
    return parentmap.count(n) == 1 ? parentmap.at(n) : nullptr ;
}

void tree::depth_r( nd* n, int d )
{
    if( n == nullptr ) return ; 
    depth_r( n->left , d+1 ); 
    depth_r( n->right, d+1 ); 
    n->depth = d ; 
}

void tree::clear_mkr()
{
    clear_mkr_r(root); 
}

void tree::clear_mkr_r( nd* n )
{
    if( n == nullptr ) return ; 
    clear_mkr_r( n->left  ); 
    clear_mkr_r( n->right ); 
    n->mkr = ' ' ; 
}

void tree::parent_r( nd* n, int depth )
{
    if( n->is_boolean() )
    {
        parent_r( n->left, depth+1 );  
        parent_r( n->right, depth+1 );  

        parentmap[n->left] = n ; 
        parentmap[n->right] = n ; 
    }
} 

void tree::apply_cut(int cut)
{
    if(verbose) 
    printf("tree::apply_cut %d \n", cut ); 

    unsigned cycle = 0 ; 
    while( root != nullptr && root->cls != INCLUDE && cycle < 5 )
    {
        classify(cut);   // set n.cls n.mkr
        prune(false); 

        if(verbose)
        draw("tree::apply_cut before prune", count ); 

        prune(true); 
        classify(cut); 
        instrument();

        if(verbose) 
        draw("tree::apply_cut after prune and re-classify", count ); 

        cycle++ ; 
    }
}


/**
tree::classify
---------------

1. for all tree nodes sets n.cls according to the cut 
2. for crux nodes sets n.mkr and collects nodes into crux vector
3. invokes prune(false) that sets prune n.mkr without proceeding with the prune

**/

void tree::classify( int cut )
{
    crux.clear(); 
    classify_r( root, cut ); 
    //printf("crux %lu \n", crux.size() ) ; 
}

/**
tree::classify_r
-------------------

NB classification is based on the node value set by initvalue_r 
it is not using other node props like pre/in etc.. 
that get changed by instrument as the tree changes. 

bitwise-OR up the tree is not so convenient for 
identification of the crucial MIXED nodes where
**/

int tree::classify_r( nd* n, int cut )
{
    if(!n) return UNDEFINED ; 
    n->cls = UNDEFINED ; 
    if( n->left && n->right )
    {
        n->cls |= classify_r( n->left , cut ) ;   // dont do on one line, as then order is unclear
        n->cls |= classify_r( n->right, cut ) ;  

        if( n->is_crux() ) 
        {
            n->mkr = '*' ; 
            crux.push_back(n) ;

            if(verbose) 
            printf("tree::classify_r set crux %s \n", n->desc() ); 
        }
    }
    else
    {
        n->cls = n->value >= cut ? EXCLUDE : INCLUDE ; 
    }
    return n->cls ; 
}


/**
tree::prune
------------

Crux nodes by definition have mixed INCLUDE/EXCLUDE child status.
Tree pruning is done in the context of a crux node, eg 3X below
in a tree of 4 prim::


                   1
                   I    

         2                   3
         I                   X    
                
    4         5         6          7
    I         I         I          E


Gets pruned to a tree of 3 prim::

                   1
                   I    

         2                   6
         I                   I    
                
    4         5                   
    I         I                   

Mechanics of non-root prune:

* find parent of crux node (if no parent it is root so see below)  
* find left/right child status of crux node
* change the child slot occupied by the crux node in its parent 
  with the surviving child node or subtree

Mechanics of root prune:

* find surviving side of the crux and promote it to root

Following tree changes need to update tree instrumentation.

**/

void tree::prune(bool act)
{
    int num_include = num_node(INCLUDE) ; 

    if(verbose)
    printf("tree::prune num_include %d \n", num_include);  

    if( num_include == 0)
    {
        if(verbose)
        printf("tree::prune find zero remaining nodes : num_include %d, will set root to nullptr \n", num_include);  

        if(act) 
        {
            if(verbose)
            printf("tree::prune setting root to nullptr \n");  

            root = nullptr ; 
        }
    }

    if(crux.size() == 0 ) return ; 
    assert(crux.size() == 1) ;  // more than one crux node not expected

    nd* x = crux[0] ; 
    prune(x, act); 
}

void tree::prune( nd* x, bool act )
{
    assert( x && x->is_crux() ); 
    bool cut_left = x->is_cut_left() ; 
    bool cut_right = x->is_cut_right() ; 
    assert( cut_left ^ cut_right );  // XOR definition of crux node : both cannot be cut 

    nd* survivor = cut_right ? x->left : x->right ; 
    assert( survivor ); 
    survivor->mkr = 'S' ; 
    nd* p = parent(x); 

    if( p != nullptr )
    {
        p->mkr = 'P' ; 

        bool x_left_child = x == p->left ; 
        bool x_right_child = x == p->right ; 

        //printf(" left_child %d right_child %d \n", left_child, right_child ); 

        if( x_left_child )
        {
            if(act) 
            {
                if(verbose)
                printf("tree:prune setting p->left %s to survivor %s \n", p->left->desc(), survivor->desc() ); 
                p->left = survivor ; 
            }
        }
        else if( x_right_child )
        {
            if(act) 
            {
                if(verbose)
                printf("tree:prune setting p->right %s to survivor %s \n", p->right->desc(), survivor->desc() ); 
                p->right = survivor ; 
            }
        }
    }
    else
    {
        if( act )
        { 
            if(verbose)
            printf("tree::prune changing root to survivor\n"); 
            root = survivor ;  
        }
    }

    if(act)
    {
        instrument(); 
    }
}

void tree::inorder_r( nd* n )
{
    if( n == nullptr ) return ; 

    inorder_r(n->left) ; 

    n->in = inorder.size() ; inorder.push_back(n);  

    inorder_r(n->right) ; 
}

void tree::preorder_r( nd* n )
{
    if( n == nullptr ) return ; 
    n->pre = preorder.size() ; preorder.push_back(n);  

    preorder_r(n->left) ; 
    preorder_r(n->right) ; 
}

void tree::dump( const char* msg ) const 
{
    printf("%s\n",msg); 
    dump_r(root); 
}

void tree::dump_r( nd* n ) const
{
    if( n == nullptr ) return ; 
    dump_r( n->left ); 
    dump_r( n->right ); 
    printf(" value %2d depth %2d mkr %c cls %c\n", n->value, n->depth, n->mkr, pcls(n->cls) ) ; 
}

void tree::draw(const char* msg, int meta)
{
    if(!root) return ; 
    if(msg) printf("%s [%d] \n", msg, meta ); 
    canvas->clear(); 
    draw_r(root); 
    canvas->print(); 
}

void tree::draw_r( nd* n )
{
    if( n == nullptr ) return ; 
    draw_r( n->left ); 
    draw_r( n->right ); 
 
    int x = n->in ; 
    int y = n->depth  ; 

    canvas->draw( x, y, 0,   n->value ); 
    //canvas->draw( x, y, 1,   n->cls   ); 
    canvas->drawch( x, y, 1, pcls(n->cls) ); 
    canvas->drawch( x, y, 2, n->mkr ); 
}

void test_flip( nd* n )
{
    n->value = -n->value ;  // easy way to flip the nodes value 
}

void test_placement_new( nd* n )
{
    nd* p = new (n) nd( -n->value, n->left, n->right) ; 
    assert( p == n ); 
    // placement new creation of new object to replace the nd at same location 
}



void test_cuts(int height0)
{
    int count0 = tree::NumNode(height0) ; 

    for( int cut=count0 ; cut > 0 ; cut-- )
    {
        tree* t = new tree(height0) ;
        int count0 = t->count ; 
        t->apply_cut(cut); 
        printf("count0 %d cut %d t.desc %s\n", count0, cut, t->desc() ); 
        t->draw(); 
    }
}

void test_no_remaining_nodes()
{
    int height0 = 3 ; 
    tree* t = new tree(height0) ;
    t->draw(); 
    t->verbose = true ; 

    printf("t.desc %s \n", t->desc() );  

    int count0 = t->count ; 
    int cut = 3 ; 
    t->apply_cut(cut); 
    printf("count0 %d cut %d t.desc %s\n", count0, cut, t->desc() ); 
    t->draw(); 
}

int main(int argc, char**argv )
{
    //test_cuts(4); 
    test_cuts(3); 
    //test_no_remaining_nodes();

    return 0 ; 
}
