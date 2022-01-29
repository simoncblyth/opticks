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
    canvas(new SCanvas(width+1, height+2, 8, 5)) 
{
}

void CSGDraw::draw(const char* msg)
{
    LOG(info) << msg ; 

    int nodeIdxRel_root = 1 ;
    int inorder = 0 ; 
    draw_r( nodeIdxRel_root,  0, inorder ); 

    canvas->print();
} 

/**
CSGDraw::dump_r
------------------

nodeIdxRel
   1-based tree index, root=1 

**/

void CSGDraw::draw_r(int nodeIdxRel, int depth, int& inorder ) 
{
    const CSGNode* nd = q->getSelectedNode( nodeIdxRel ); 
    if( nd == nullptr ) return ; 

    std::string brief = nd->brief(); 

    int left = nodeIdxRel << 1 ; 
    int right = left + 1 ; 

    draw_r(left,  depth+1, inorder); 

    // inorder visit 
    {
        std::cout 
             << " nodeIdxRel " << std::setw(5) << nodeIdxRel
             << " depth " << std::setw(5) << depth
             << " inorder " << std::setw(5) << inorder
             << " brief " << brief
             << " : " << nd->desc() 
           
             << std::endl 
             ;

        int ix = inorder ;  
        int iy = depth ; 

        canvas->draw(   ix, iy, 0,0,  brief.c_str() ); 
     
        inorder += 1 ; 
    }
    draw_r(right, depth+1, inorder); 
}


