#include <limits>
#include <csignal>

#include "PLOG.hh"
#include "OpticksCSGMask.h"

#include "NBBox.hpp"
#include "nmat4triple.hpp"
#include "NMultiUnion.hpp"

const plog::Severity nmultiunion::LEVEL = PLOG::EnvLevel("nmultiunion", "DEBUG"); 

nbbox nmultiunion::bbox() const 
{
    std::cout << "nmultiunion::bbox subs.size " << subs.size() << std::endl ; 

    nbbox bb = make_bbox() ; 

    for(unsigned isub=0 ; isub < subs.size() ; isub++)
    {
        const nnode* sub = subs[isub] ; 

        std::cout 
            << " isub " << std::setw(5) << isub 
            << " sub->gtransform " << std::setw(10) << sub->gtransform
            << " sub->transform " << std::setw(10) << sub->transform
            << std::endl 
            ;

        nbbox sub_bb = sub->bbox();  
        sub_bb.dump(); 

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

    return CreateFromList(type, prim) ; 
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
        comp->subs.push_back(sub); 
    }
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

nmultiunion* nmultiunion::Create(OpticksCSG_t type, unsigned sub_num  )
{
    nmultiunion* n = Create(type) ; 
    n->setSubNum(sub_num); 
    return n ; 
}




int nmultiunion::par_euler() const { return 0 ;   }
unsigned nmultiunion::par_nsurf() const  { return 0 ;   }
unsigned nmultiunion::par_nvertices(unsigned , unsigned ) const { return 0 ; }

