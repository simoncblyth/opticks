#pragma once

#include <string>

#include "NNodeEnum.hpp"
#include "NPY_API_EXPORT.hh"
struct nnode ; 


struct NPY_API NNodeCoincidence 
{
    NNodeCoincidence(nnode* i_, nnode* j_, NNodePairType p_) 
         :
         i(i_),
         j(j_),
         p(p_),
         n(NUDGE_NONE),
         fixed(false)
    {} ;

    nnode* i ; 
    nnode* j ;
    NNodePairType p ; 
    NNodeNudgeType n ; 

    bool   fixed ; 

    std::string desc() const ; 

    bool is_union_siblings() const;
    bool is_union_parents() const;
    bool is_same_union() const;
    bool is_siblings() const;
};


