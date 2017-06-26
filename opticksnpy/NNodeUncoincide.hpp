#pragma once

#include <vector>

#include <glm/fwd.hpp>
#include "NNodeEnum.hpp"
#include "NPY_API_EXPORT.hh"

struct nuv ; 
struct nnode ; 

class NPY_API NNodeUncoincide
{
    public:
        NNodeUncoincide(nnode* node);
    public:
        unsigned uncoincide();

    private:
        bool     is_uncoincidable_subtraction(nnode*& a, nnode*& b) ;
        bool     is_uncoincidable_union(nnode*& a, nnode*& b) ;
        unsigned uncoincide_subtraction(nnode* a, nnode* b) ;
        unsigned uncoincide_union(nnode* a, nnode* b) ;

    private:
        nnode* m_node ; 
        int    m_verbosity ; 

};


