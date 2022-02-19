#include <limits>
#include <csignal>

#include "PLOG.hh"
#include "OpticksCSGMask.h"

#include "NBBox.hpp"
#include "nmat4triple.hpp"
#include "NMultiUnion.hpp"

const plog::Severity nmultiunion::LEVEL = PLOG::EnvLevel("nmultiunion", "DEBUG"); 



void nmultiunion::pdump(const char* msg) const 
{
    std::cout << msg << " subs.size " << subs.size() << std::endl ; 
}


nbbox nmultiunion::bbox() const 
{
    LOG(LEVEL)  << "nmultiunion::bbox subs.size " << subs.size()  ; 

    nbbox bb = make_bbox() ; 

    for(unsigned isub=0 ; isub < subs.size() ; isub++)
    {
        const nnode* sub = subs[isub] ; 

        LOG(LEVEL)
            << " isub " << std::setw(5) << isub 
            << " sub->gtransform " << std::setw(10) << sub->gtransform
            << " sub->transform " << std::setw(10) << sub->transform
            ;

        nbbox sub_bb = sub->bbox();  
        //sub_bb.dump(); 

        bb.include(sub_bb); 
    }

    // gtransform is the composite one
    return gtransform ? bb.make_transformed(gtransform->t) : bb ; 
}



float nmultiunion::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform

    float sd = std::numeric_limits<float>::max() ;  

    for(unsigned isub=0 ; isub < subs.size() ; isub++)
    {
        const nnode* sub = subs[isub] ; 
        float sd_sub = (*sub)( p.x, p.y, p.z );  
        sd = std::min( sd, sd_sub );   
    }

    return complement ? -sd : sd ;
} 

/**
nmultiunion::CreateFromTree
-----------------------------

1. check all operators are CSG_UNION

hmm need to flatten transforms and bake into node local transfrom
in order to be able to chop the tree into a pile of leaves without changing geometry 

* first implement subtree clone and then add option to flatten transforms 

**/

nmultiunion* nmultiunion::CreateFromTree( OpticksCSG_t type, const nnode* src )  // static 
{
    LOG(LEVEL) << "[" ; 
    nnode* subtree = src->deepclone(); 
    subtree->prepareTree() ;  // sets parent links and gtransforms by multiplying the transforms 

    unsigned mask = subtree->get_oper_mask(); 
    OpticksCSG_t subtree_type = CSG_MonoOperator(mask); 
 
    if(subtree_type != CSG_UNION)
    {
         LOG(fatal) << "Can only create nmultiunion from a subtree that is purely composed of CSG_UNION operator nodes" ;  
         std::raise(SIGINT);  
    }

    std::vector<nnode*> prim ; 
    subtree->collect_prim_for_edit(prim); 

    unsigned num_prim = prim.size(); 
    for(unsigned i=0 ; i < num_prim ; i++) 
    {
        nnode* p = prim[i]; 
        if( p->gtransform )
        {
            p->transform = p->gtransform ; 
        }
    }

    nmultiunion* n = CreateFromList(type, prim) ; 
    
    LOG(LEVEL) << "]" ; 
    return n ; 
}

/**
nmultiunion::CreateFromList
-----------------------------

1. check all prim are not complemented
2. check all prim have overlap with at least one other : thats a bit difficult 

**/
nmultiunion* nmultiunion::CreateFromList( OpticksCSG_t type, std::vector<nnode*>& prim  )  // static 
{
    unsigned sub_num = prim.size(); 
    assert( sub_num > 0 ); 
    nmultiunion* comp = Create(type, sub_num) ;
    for(unsigned i=0 ; i < sub_num ; i++)
    {
        nnode* sub = prim[i] ; 
        assert( sub->complement == false );  
        LOG(info) << " sub.type " << sub->type ; 
        comp->subs.push_back(sub); 
    }

    nbbox bb = comp->bbox(); 
    LOG(info) << " bb " << bb.desc() ; 
    comp->set_bbox(bb);  

    return comp ; 
}

nmultiunion* nmultiunion::Create(OpticksCSG_t type )  // static 
{
    nmultiunion* n = new nmultiunion ; 
    assert( type == CSG_CONTIGUOUS  ); 
    nnode::Init(n,type) ; 

    return n ; 
}
nmultiunion* nmultiunion::Create(OpticksCSG_t type, const nquad& param  ) // static
{
    nmultiunion* n = Create(type) ; 
    n->param = param ;    
    return n ; 
}

/**
nmultiunion::Create
---------------------

Cannot set the real subOffset at this juncture, need to wait until 
are serializing all nodes of the prim in order to calculate the subOffsets 
of all the list-nodes within the tree. 

**/

nmultiunion* nmultiunion::Create(OpticksCSG_t type, unsigned sub_num  )
{
    nmultiunion* n = Create(type) ; 
    n->setSubNum(sub_num); 

    unsigned sub_offset = 0 ;  // placeholder, gets set by  NCSG::export_tree_list_ 
    n->setSubOffset(sub_offset); 
    return n ; 
}




int nmultiunion::par_euler() const { return 0 ;   }
unsigned nmultiunion::par_nsurf() const  { return 0 ;   }
unsigned nmultiunion::par_nvertices(unsigned , unsigned ) const { return 0 ; }

