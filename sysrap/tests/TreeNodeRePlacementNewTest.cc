// name=TreeNodeRePlacementNewTest ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <cassert>
#include <cstring>
#include <cstdio>
#include <vector>

struct nd 
{
    int label ; 
    nd* left ; 
    nd* right ;    

    int depth ; 
    int side ; 
}; 


struct tree
{
    int count ; 
    int scale ; 
    int height ; 
    nd* root ; 
    std::vector<nd*> inorder ; 
    int width ; 
    char* canvas ; 
    int ny ; 
    int nx ; 

    tree( int height_ ); 

    nd* build_r(int d, int h); 
    int init_inorder( nd* n ); 
    void inorder_r( nd* n ); 
    void dump_r( nd* n ); 
    char* init_canvas();

    int maxdepth_r(nd* n, int depth) const ;
    int maxdepth() const  ;

    void draw(); 
    void draw_r( nd* n ); 
    void draw(int x, int y, int val);
    void draw(int x, int y, char* txt);

};


tree::tree(int height_)
    :
    count(0),
    scale(3),
    height(height_),
    root(build_r(0, height)),
    width(init_inorder(root)),
    canvas(init_canvas())
{
}

nd* tree::build_r(int d, int h)
{
    nd* l = h > 0 ? build_r( d+1, h-1) : nullptr ;  
    nd* r = h > 0 ? build_r( d+1, h-1) : nullptr ;  

    count++ ;          // postorder node creation count 
    int label = count ;   
    int side = -1 ;   // this gets filled in by init_inorder

    nd* n = new nd { label, l, r, d, side } ;   
 
    return n ; 
}

int tree::init_inorder(nd* root)
{
    inorder.clear();
    inorder_r(root); 
    return inorder.size(); 
}

void tree::inorder_r( nd* n )
{
    if( n == nullptr ) return ; 
    inorder_r(n->left) ; 
    n->side = inorder.size() ; 
    inorder.push_back(n);  
    inorder_r(n->right) ; 
}

char* tree::init_canvas()
{
    ny = height*(scale+1) ; 
    nx = width*(scale+1) + 1 ;   // +1 for newline
    int num = nx*ny ;  

    char* c = new char[num+1] ;   // +1 for string termination
    for(int y=0 ; y < ny ; y++) for(int x=0 ; x < nx ; x++)  c[y*nx+x] = ' ' ;  
    for(int y=0 ; y < ny ; y++) c[y*nx+nx-1] = '\n' ;  
    c[num] = '\0' ;  // string terminate 

    return c ; 
}

void tree::dump_r( nd* n )
{
    if( n == nullptr ) return ; 
    dump_r( n->left ); 
    dump_r( n->right ); 
    printf(" label %d depth %d \n", n->label, n->depth) ; 
}

void tree::draw(int x, int y, int val)
{
    char tmp[10] ; 
    int rc = sprintf(tmp, "%d", val );  
    draw(x, y, tmp);  
}

void tree::draw(int x, int y, char* txt)   // 0,0 is at top left 
{
    memcpy( canvas + y*nx + x , txt, strlen(txt) ); 
    // memcpy to avoid string termination 
    // hmm: drawing near the righthand side may stomp on newlines
    // hmm: drawing near bottom right risks writing off the end of the canvas
} 

void tree::draw()
{
    printf("tree::draw maxdepth %d \n", maxdepth()) ; 
    draw_r(root); 
    printf("\n%s",canvas) ; 
}

void tree::draw_r( nd* n )
{
    if( n == nullptr ) return ; 
    draw_r( n->left ); 
    draw_r( n->right ); 
 
    int x = n->side ; 
    int y = n->depth  ; 

    draw( x*scale, y*scale, n->label ); 
}




int tree::maxdepth_r(nd* n, int depth) const 
{
    return n->left && n->right ? std::max( maxdepth_r( n->left, depth+1), maxdepth_r(n->right, depth+1)) : depth ; 
}
int tree::maxdepth() const  
{
    return maxdepth_r( root, 0 ); 
}

int main(int argc, char**argv )
{
    tree* t = new tree(4) ; 
    //t->draw();    

    nd* n = t->inorder[10] ; 
    //n->label = -n->label ;  // easy way to flip the nodes label 

    nd* p = new (n) nd { -n->label, n->left, n->right, n->depth, n->side } ;  
    assert( p == n ); 
    // placement new creation of new object to replace the nd at same location 

    t->draw();    
 
    return 0 ; 
}
