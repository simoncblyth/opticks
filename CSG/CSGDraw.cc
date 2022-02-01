#include "PLOG.hh"
#include "SCanvas.hh"

#include "scuda.h"
#include "squad.h"

#include "CSGNode.h"
#include "CSGQuery.h"
#include "CSGDraw.h"

CSGDraw::CSGDraw(const CSGQuery* q_)
    :
    q(q_),
    width(q->select_numNode),
    height(q->getSelectedTreeHeight()),
    canvas(new SCanvas(width+1, height+2, 10, 5)),
    dump(false)
{
}

void CSGDraw::draw(const char* msg)
{
    int nodeIdxRel_root = 1 ;
    int inorder = 0 ; 
    char axis = 'Y' ; 

    LOG(info) << msg << " axis " << axis ; 

    draw_r( nodeIdxRel_root,  0, inorder, axis ); 

    canvas->print();
} 

/**
CSGDraw::dump_r
------------------

nodeIdxRel
   1-based tree index, root=1 

**/

void CSGDraw::draw_r(int nodeIdxRel, int depth, int& inorder, char axis ) 
{
    const CSGNode* nd = q->getSelectedNode( nodeIdxRel ); 
    if( nd == nullptr ) return ; 
    if( nd->is_zero() ) return ; 


    const float* aabb = nd->AABB();  
    float a0, a1 ; 
    switch(axis)
    {
       case 'X': a0 = aabb[0] ; a1 = aabb[0+3] ; break ; 
       case 'Y': a0 = aabb[1] ; a1 = aabb[1+3] ; break ; 
       case 'Z': a0 = aabb[2] ; a1 = aabb[2+3] ; break ; 
    }

    std::string brief = nd->brief(); 
    const char* label = brief.c_str(); 

    int left = nodeIdxRel << 1 ; 
    int right = left + 1 ; 

    draw_r(left,  depth+1, inorder, axis); 

    // inorder visit 
    {
        if(dump) std::cout 
             << " nodeIdxRel " << std::setw(5) << nodeIdxRel
             << " depth " << std::setw(5) << depth
             << " inorder " << std::setw(5) << inorder
             << " brief " << brief
             << " : " << nd->desc() 
             << std::endl 
             ;

        int ix = inorder ;  
        int iy = depth ; 

        canvas->draw(   ix, iy, 0,0,  label ); 
        canvas->draw(   ix, iy, 0,1,  nodeIdxRel ); 

        const char* fmt = "%7.2f" ; 
        canvas->drawf(  ix, iy, 0,2,  a1 , fmt); 
        canvas->drawf(  ix, iy, 0,3,  a0 , fmt); 
     
        inorder += 1 ; 
    }
    draw_r(right, depth+1, inorder, axis ); 
}


