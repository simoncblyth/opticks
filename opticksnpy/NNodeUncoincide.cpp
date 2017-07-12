#include "PLOG.hh"

#include "OpticksCSG.h"
#include "OpticksCSGMask.h"

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "NNode.hpp"
#include "Nuv.hpp"
#include "NBBox.hpp"
#include "NNodeUncoincide.hpp"
#include "NNodeNudger.hpp"

#include "NPrimitives.hpp"

NNodeUncoincide::NNodeUncoincide(nnode* node, unsigned verbosity)
   :
   m_node(node),
   m_verbosity(verbosity)
{
}

unsigned NNodeUncoincide::uncoincide()
{
    // canonically invoked via nnode::uncoincide from NCSG::import_r

    nnode* a = NULL ; 
    nnode* b = NULL ; 

    unsigned rc = 0 ; 

    if(m_node->is_root())
    {
        rc = uncoincide_treewise();
    }

    // NB BELOW PAIRWISE APPROACH CURRENTLY NOT USED
    else if(is_uncoincidable_subtraction(a,b))
    {
        rc = uncoincide_subtraction(a,b);
    } 
    else if(is_uncoincidable_union(a,b))
    {
        rc = uncoincide_union(a,b);
    }

    return rc ; 
}


bool NNodeUncoincide::is_uncoincidable_subtraction(nnode*& a, nnode*& b)
{
    assert(!m_node->complement); 
    // hmm theres an implicit assumption here that all complements have 
    // already fed down to the leaves

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

    bool is_uncoincidable  =  a && b && a->type == CSG_BOX3 && b->type == CSG_BOX3 ;
    return is_uncoincidable ;
}

unsigned NNodeUncoincide::uncoincide_subtraction(nnode* a, nnode* b)
{
    float epsilon = 1e-5f ; 
    unsigned level = 1 ; 
    int margin = 1 ; 
    

    std::vector<nuv> coincident ;
    a->getCoincident( coincident, b, epsilon, level, margin, FRAME_LOCAL );

    unsigned ncoin = coincident.size() ;
    if(ncoin > 0)
    {
        LOG(info) << "NNodeUncoincide::uncoincide_subtraction   START " 
                  << " A " << a->tag()
                  << " B " << b->tag()
                  ;

        a->verbosity = 4 ; 
        b->verbosity = 4 ; 

        a->pdump("A");
        b->pdump("B");

        assert( ncoin == 1);
        nuv uv = coincident[0] ; 

        //float delta = 5*epsilon ; 
        float delta = 1 ;  // crazy large delta, so can see it  

        std::cout << "NNodeUncoincide::uncoincide_subtraction" 
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
    return ncoin  ;
}



/*

Hmm nothing fancy needed to see the coincidence,
equality of a.bbox.min.z and b.bbox.max.z or vv.
    
Shapes with separate +z and -z parameters are
easy to nudge in +z, -z direction.  

* CSG_CYLINDER
* CSG_CONE
* CSG_ZSPHERE

Shapes with symmetric parameters like box3 are a pain, as 
to grow in eg +z direction need to grow in both +z and -z
first and then transform to keep the other side at same place.

Hmm to avoid this perhaps make a CSG_ZBOX primitive ? 

*/


bool NNodeUncoincide::is_uncoincidable_union(nnode*& a, nnode*& b)
{
    OpticksCSG_t type = m_node->type ; 
    if( type != CSG_UNION ) return false ; 

    nnode* left = m_node->left ; 
    nnode* right = m_node->right ; 

    nbbox l = left->bbox();
    nbbox r = right->bbox();

    // order a,b primitives in increasing z
    //
    // The advantage of using bbox is that 
    // can check for bbox coincidence with all node shapes, 
    // not just the z-nudgeable which have z1() z2() methods.
    //

    // hmm these values sometimes have  transforms applied, 
    // so should use epsilon 

    float epsilon = 1e-5 ; 

    if( fabsf(l.max.z - r.min.z) < epsilon )  
    {
        LOG(info) << "   |----L----||--- R-------|  -> Z "    ;  
        a = left ; 
        b = right ;  
    }
    else if( fabsf(r.max.z - l.min.z )  < epsilon )    
    {
        LOG(info) << "   |----R----||---L-------|  -> Z "    ;  
        a = right ; 
        b = left ;  
    }

    bool is_uncoincidable = false ; 
    if( a && b )
    {
        bool can_fix = a->is_znudge_capable() && b->is_znudge_capable() ; 
        if(!can_fix) 
        {
            LOG(warning) << "bbox.z coincidence seen, but cannot fix as one/both primitives are not znudge-able " ;  
        } 
        else
        {
            LOG(warning) << "bbox.z coincidence seen, proceed to fix as both primitives are znudge-able " ;  
            is_uncoincidable = true ;  
        }
    }
    return is_uncoincidable ;
}


 
unsigned NNodeUncoincide::uncoincide_union(nnode* a, nnode* b)
{
    // opticks-tbool 143
    assert( a->is_znudge_capable() );
    assert( b->is_znudge_capable() );

    std::cout << std::endl << std::endl ; 
    LOG(info) << "NNodeUncoincide::uncoincide_union"
               << " A " << a->tag()
               << " B " << b->tag()
               ;

    a->dump("uncoincide_union A");
    b->dump("uncoincide_union B");

    /*

        +--------------+
        |              |
        |  A          ++-------------+
        |             ||             |
        |             ||          B  |
        |             ||             | 
        |             ||             |    
        |             ||             |
        |             ||             |
        |             ++-------------+
        |              |
        |              |
        +--------------+

       a.z1           a.z2
                      b.z1          b.z2      


       (schematic, actually there are usually transforms applied that
        prevent a.z2 == b.z1  ) 

        ------> Z

        Hmm in-principal the transforms could also change the radii 
        ordering but thats unlikely, as usually just translations.

    */


    float a_r2 = a->r2() ;
    float b_r1 = b->r1() ;

    float dz = 0.5 ;  // need some checks regarding size of the objs

    if( a_r2 > b_r1 )
    {
        b->decrease_z1( dz ); 
    }
    else
    {
        a->increase_z2( dz ); 
    }


    std::cout 
               << " A.z1 " << std::fixed << std::setw(10) << std::setprecision(4) << a->z1()
               << " A.z2 " << std::fixed << std::setw(10) << std::setprecision(4) << a->z2()
               << " B.z1 " << std::fixed << std::setw(10) << std::setprecision(4) << b->z1()
               << " B.z2 " << std::fixed << std::setw(10) << std::setprecision(4) << b->z2()
               << std::endl ; 

    std::cout 
               << " A.r1 " << std::fixed << std::setw(10) << std::setprecision(4) << a->r1()
               << " A.r2 " << std::fixed << std::setw(10) << std::setprecision(4) << a->r2()
               << " B.r1 " << std::fixed << std::setw(10) << std::setprecision(4) << b->r1()
               << " B.r2 " << std::fixed << std::setw(10) << std::setprecision(4) << b->r2()
               << std::endl ; 


    nbbox a_ = a->bbox();
    nbbox b_ = b->bbox();
    std::cout 
               << " a.bbox " << a_.desc() << std::endl 
               << " b.bbox " << b_.desc() << std::endl 
               ; 




    // to decide which one to nudge need to know the radius at the union interfaces
    // need to nudge the one with the smaller radius


    std::cout << std::endl << std::endl ; 
    //assert(0 && "hari-kari") ; 

    return 0 ; 
}








unsigned NNodeUncoincide::uncoincide_treewise()
{
    assert( m_node->is_root() );

    nnode* root = m_node ; 
    nnode* left = root->left ; 
    nnode* right = root->right ; 

    unsigned typmsk = root->get_type_mask();

    unsigned uncy     = CSGMASK_UNION | CSGMASK_CYLINDER ;
    unsigned uncyco   = CSGMASK_UNION | CSGMASK_CYLINDER | CSGMASK_CONE ;
    unsigned uncycodi = CSGMASK_UNION | CSGMASK_DIFFERENCE | CSGMASK_CYLINDER | CSGMASK_CONE ;

    bool root_di = root->type == CSG_DIFFERENCE ; 
    bool root_uncy   = typmsk == uncy ;
    bool root_uncyco = typmsk == uncyco ;
    bool root_uncycodi = typmsk == uncycodi  ;

    if(root_uncy || root_uncyco)
    {
         uncoincide_uncyco(m_node);
    }
    else if( root_uncycodi )
    {
        unsigned left_typmsk = left->get_type_mask();

        bool left_uncy   =  left_typmsk == uncy ;
        bool left_uncyco =  left_typmsk == uncyco ;

        if( root_di  && ( left_uncy || left_uncyco ))
        {
            LOG(info) << "NNodeUncoincide::uncoincide_tree"
                      << " TRYING root.left UNCOINCIDE_UNCYCO " 
                      << " root " << root->get_type_mask_string()
                      << " left " << left->get_type_mask_string()
                      << " right " << right->get_type_mask_string()
                      ;

            uncoincide_uncyco( left );
        }
    }
    return 0 ; 
}




unsigned NNodeUncoincide::uncoincide_uncyco(nnode* node)
{
    float epsilon = 1e-5f ; 
    NNodeNudger zn(node, epsilon, m_verbosity) ; 

    if(m_verbosity > 2 )
    zn.dump("NNodeUncoincide::uncoincide_uncyco before znudge");

    zn.znudge();

    if(m_verbosity > 2 )
    zn.dump("NNodeUncoincide::uncoincide_uncyco after znudge");

    return 0 ; 
}


