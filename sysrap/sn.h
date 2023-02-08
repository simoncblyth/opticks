#pragma once
/**
sn.h : minimal pointer based transient binary tree node
========================================================

**/

#include <vector>
#include <sstream>
#include "scanvas.h"

struct sn
{
    int t ; 
    sn* l ; 
    sn* r ;     


    int num_node() const ; 
    int num_node_r(int d) const ; 

    int max_depth() const ; 
    int max_depth_r(int d) const ; 

    void inorder(std::vector<const sn*>& order ) const ; 
    void inorder_r(std::vector<const sn*>& order, int d ) const ; 

    std::string render() const ; 
    void render_r(scanvas* canvas, std::vector<const sn*>& order, int d) const ; 

    static sn* Zero() ; 
    static sn* Build_r(int elevation, int op); 

    static int BinaryTreeHeight(int num_leaves); 
    static int BinaryTreeHeight_1(int num_leaves); 

    static sn* CommonTree(int num_leaves, int op ); 
};


inline int sn::num_node() const
{
    return num_node_r(0);
}
inline int sn::num_node_r(int d) const
{
    int nn = 1 ;   // always at least 1 node,  no exclusion of CSG_ZERO
    nn += l ? l->num_node_r(d+1) : 0 ; 
    nn += r ? r->num_node_r(d+1) : 0 ; 
    return nn ;
}

inline int sn::max_depth() const
{
    return max_depth_r(0);
}
inline int sn::max_depth_r(int d) const
{
    return l && r ? std::max( l->max_depth_r(d+1), r->max_depth_r(d+1)) : d ; 
}

inline void sn::inorder(std::vector<const sn*>& order ) const
{
    inorder_r(order, 0);
}
inline void sn::inorder_r(std::vector<const sn*>& order, int d ) const
{
    if(l) l->inorder_r(order, d+1) ; 
    order.push_back(this); 
    if(r) r->inorder_r(order, d+1) ; 
}


inline std::string sn::render() const
{
    int width = num_node();
    int height = max_depth();

    std::vector<const sn*> order ;
    inorder(order);
    assert( int(order.size()) == width );

    scanvas canvas( width+1, height+2, 4, 2 );
    render_r(&canvas, order,  0);

    std::stringstream ss ;
    ss << std::endl ;
    ss << "sn::render width " << width << " height " << height  << std::endl ;
    ss << canvas.c << std::endl ;
    std::string str = ss.str();
    return str ;
}

void sn::render_r(scanvas* canvas, std::vector<const sn*>& order, int d) const
{
    int ordinal = std::distance( order.begin(), std::find(order.begin(), order.end(), this )) ;
    assert( ordinal < int(order.size()) );

    int ix = ordinal ;
    int iy = d ;
    char l0 = '.' ;

    canvas->drawch( ix, iy, 0,0,  l0 );

    if(l) l->render_r( canvas, order, d+1 );
    if(r) r->render_r( canvas, order, d+1 );
}


inline sn* sn::Zero()   // static
{
    return new sn {0, nullptr, nullptr} ; 
}
inline sn* sn::Build_r( int elevation, int op )  // static
{
    sn* l = elevation > 1 ? Build_r( elevation - 1 , op ) : sn::Zero() ; 
    sn* r = elevation > 1 ? Build_r( elevation - 1 , op ) : sn::Zero() ; 
    return new sn { op, l, r } ;  
}



inline int sn::BinaryTreeHeight(int q_leaves )
{
    int h = 0 ; 
    while( (1 << h) < q_leaves )  h += 1 ; 
    return h ; 
}

inline int sn::BinaryTreeHeight_1(int q_leaves )
{
    int  height = -1 ;
    for(int h=0 ; h < 10 ; h++ )
    {
        int tprim = 1 << h ;
        if( tprim >= q_leaves )
        {
           height = h ;
           break ;
        }
    }
    return height ; 
}

inline sn* sn::CommonTree( int num_leaves, int op ) // static
{   
    int height = BinaryTreeHeight(num_leaves) ;
    return Build_r( height, op );
}          

