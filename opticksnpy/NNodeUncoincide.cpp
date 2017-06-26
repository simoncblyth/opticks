#include "PLOG.hh"

#include "OpticksCSG.h"

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "NNode.hpp"
#include "Nuv.hpp"
#include "NBBox.hpp"
#include "NNodeUncoincide.hpp"

#include "NPrimitives.hpp"

NNodeUncoincide::NNodeUncoincide(nnode* node)
   :
   m_node(node),
   m_verbosity(node->verbosity)
{
}



bool NNodeUncoincide::can_uncoincide(const nnode* a, const nnode* b) const 
{
    return ( a && b && a->type == CSG_BOX3 && b->type == CSG_BOX3 ) ;
}

unsigned NNodeUncoincide::uncoincide()
{
    // canonically invoked for bileaf from NCSG::import_r

    assert( m_node->is_bileaf() && "NNodeUncoincide::uncoincide expects left and right to be primitives" );
  
    float epsilon = 1e-5f ; 
    unsigned level = 1 ; 
    int margin = 1 ; 

    std::vector<nuv> coincident ;
    
    nnode* a = NULL ; 
    nnode* b = NULL ; 

    // hmm theres an implicit assumption here that all complements have 
    // already fed down to the leaves
    assert(!m_node->complement); 


    OpticksCSG_t type = m_node->type ; 
    nnode* left = m_node->left ; 
    nnode* right = m_node->right ; 

    if( type == CSG_DIFFERENCE )  // left - right 
    {
        a = left ; 
        b = right ; 
    }
    else if( type == CSG_INTERSECTION && !left->complement &&  right->complement)  // left * !right  ->   left - right
    { 
        a = left ; 
        b = right ; 
    }
    else if( type == CSG_INTERSECTION &&  left->complement && !right->complement)  // !left * right  ->  right - left 
    {
        a = right ; 
        b = left ; 
    }

    if( a && b )
    {
        if(!can_uncoincide(a, b))
        {
            LOG(debug) << "NNodeUncoincide::uncoincide detects bileaf A-B subtraction, but must skip as not implemented for: "
                         << " A " << a->csgname()
                         << " B " << b->csgname()
                          ;
            return 0 ; 
        }


        a->getCoincident( coincident, b, epsilon, level, margin, FRAME_LOCAL );

        unsigned ncoin = coincident.size() ;
        if(ncoin > 0)
        {
            LOG(info) << "NNodeUncoincide::uncoincide   START " ; 

            a->verbosity = 4 ; 
            b->verbosity = 4 ; 

            a->pdump("A");
            b->pdump("B");

            assert( ncoin == 1);
            nuv uv = coincident[0] ; 

            //float delta = 5*epsilon ; 
            float delta = 1 ;  // crazy large delta, so can see it  

            std::cout << "NNodeUncoincide::uncoincide" 
                      << " ncoin " << ncoin 
                      << " uv " << uv.desc()
                      << " epsilon " << epsilon
                      << " delta " << delta
                      << std::endl 
                      ;

            b->nudge( uv.s(),  delta ); 

            b->pdump("B(nudged)");


            LOG(info) << "NNodeUncoincide::uncoincide   DONE " ; 
        } 
    }
    return coincident.size() ;
}






