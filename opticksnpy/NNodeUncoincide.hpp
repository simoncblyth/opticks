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
        bool can_uncoincide(const nnode* a, const nnode* b) const ;

    private:
        nnode* m_node ; 
        int    m_verbosity ; 

};


