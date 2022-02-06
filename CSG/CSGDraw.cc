#include "PLOG.hh"
#include "OpticksCSG.h"
#include "SCanvas.hh"

#include "scuda.h"
#include "squad.h"

#include "CSGNode.h"
#include "CSGQuery.h"
#include "CSGDraw.h"

CSGDraw::CSGDraw(const CSGQuery* q_)
    :
    q(q_),
    type(q->getSelectedType()), 
    width(q->select_numNode),
    height( CSG::IsTree((OpticksCSG_t)type) ? q->getSelectedTreeHeight() : 1),
    canvas(new SCanvas(width+1, height+2, 10, 5)),
    dump(false)
{
}

void CSGDraw::draw(const char* msg)
{
    char axis = 'Y' ; 

    LOG(info) << msg << " axis " << axis ; 

    if( CSG::IsTree((OpticksCSG_t)type) )
    {
        int nodeIdxRel_root = 1 ;
        int inorder = 0 ; 
        draw_tree_r( nodeIdxRel_root,  0, inorder, axis ); 
    }
    else if( CSG::IsList((OpticksCSG_t)type) )
    {
        draw_list(); 
    }
    else if( CSG::IsLeaf((OpticksCSG_t)type) )
    {
        draw_leaf(); 
    }
    else
    {
        assert(0) ; // unexpected type 
    }


    canvas->print();
} 

/**
CSGDraw::dump_tree_r
-----------------------

nodeIdxRel
   1-based tree index, root=1 

**/

void CSGDraw::draw_tree_r(int nodeIdxRel, int depth, int& inorder, char axis ) 
{
    const CSGNode* nd = q->getSelectedNode( nodeIdxRel - 1  );  // convert 1-based index to 0-based
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

    draw_tree_r(left,  depth+1, inorder, axis); 

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
    draw_tree_r(right, depth+1, inorder, axis ); 
}

void CSGDraw::draw_list()
{
    assert( CSG::IsList((OpticksCSG_t)type) ); 

    unsigned idx = 0 ; 
    const CSGNode* head = q->getSelectedNode(idx);
    unsigned sub_num = head->subNum() ; 

    LOG(info)
        << " sub_num " << sub_num 
        ; 

    draw_list_item( head, idx ); 

    for(unsigned isub=0 ; isub < sub_num ; isub++)
    {
        idx = 1+isub ;   // 0-based node idx
        const CSGNode* sub = q->getSelectedNode(idx); 

        draw_list_item( sub, idx ); 
    }
}

void CSGDraw::draw_leaf()
{
    assert( CSG::IsLeaf((OpticksCSG_t)type) ); 

    unsigned idx = 0 ; 
    const CSGNode* leaf = q->getSelectedNode(idx);
 
    draw_list_item( leaf, idx ); 
} 

void CSGDraw::draw_list_item( const CSGNode* nd, unsigned idx )
{
    int ix = idx == 0 ? 0 : idx+1  ; 
    int iy = idx == 0 ? 0 : 1      ;

    std::string brief = nd->brief(); 
    const char* label = brief.c_str(); 

    canvas->draw(   ix, iy, 0,0,  label ); 
    canvas->draw(   ix, iy, 0,1,  idx ); 
}


