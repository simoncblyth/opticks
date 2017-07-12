#pragma once

#include <vector>

#include <glm/fwd.hpp>
#include "NNodeEnum.hpp"
#include "NPY_API_EXPORT.hh"

struct nuv ; 
struct nnode ; 

/*
NNodeUncoincide
=================

See issues/NScanTest_csg_zero_crossings.rst 

* pairwise incoincidence is not helping much, 
  need to act at tree level 


Canonically invoked after CSG import 
via NCSG::postimport_uncoincide/nnode::uncoincide

*/

class NPY_API NNodeUncoincide
{
    public:
        NNodeUncoincide(nnode* node, unsigned verbosity);
    public:
        unsigned uncoincide();

    private:
        // treewise approach 
        unsigned uncoincide_treewise();
        unsigned uncoincide_treewise_fiddle();
        unsigned uncoincide_uncyco(nnode* node);
    private:
        // pairwise approach 
        bool     is_uncoincidable_subtraction(nnode*& a, nnode*& b) ;
        bool     is_uncoincidable_union(nnode*& a, nnode*& b) ;
        unsigned uncoincide_subtraction(nnode* a, nnode* b) ;
        unsigned uncoincide_union(nnode* a, nnode* b) ;

    private:
        nnode* m_node ; 
        int    m_verbosity ; 

};


