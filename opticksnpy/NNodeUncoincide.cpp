#include "PLOG.hh"

#include "OpticksCSG.h"
#include "OpticksCSGMask.h"

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

unsigned NNodeUncoincide::uncoincide()
{
    // canonically invoked via nnode::uncoincide from NCSG::import_r

    nnode* a = NULL ; 
    nnode* b = NULL ; 

    unsigned rc = 0 ; 

    if(m_node->is_root())
    {
        rc = uncoincide_tree();
    }
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

Hmm to avoid this perhaps make another primitive ? 


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

    a->dump_full("uncoincide_union A");
    b->dump_full("uncoincide_union B");

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
        preventing a.z2 == b.z1  ) 

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




unsigned NNodeUncoincide::uncoincide_tree()
{
    unsigned typmsk = m_node->get_type_mask();
    bool uncy   = typmsk == (CSGMASK_UNION | CSGMASK_CYLINDER) ;
    bool uncyco = typmsk == (CSGMASK_UNION | CSGMASK_CYLINDER | CSGMASK_CONE) ;

    if(uncy || uncyco)
    {
         uncoincide_tree_uncyco();
    }
    return 0 ; 
}


struct Primitives 
{
    const nnode* root ; 
    std::vector<const nnode*> prim ; 
    std::vector<nbbox>        bb ; 
    std::vector<unsigned>     zorder ; 

    Primitives(const nnode* root) 
         :
         root(root)
    {
         root->collect_prim(prim);

         for(unsigned i=0 ; i < prim.size() ; i++)
         {
              const nnode* p = prim[i] ; 
              nbbox pbb = p->bbox(); 
              bb.push_back(pbb);
              zorder.push_back(i);
         }
         std::sort(zorder.begin(), zorder.end(), *this );
    }
   
    bool operator()( int i, int j)  
    {
         //return bb[i].min.z > bb[j].min.z ; // descending bb.min.z
         return bb[i].min.z < bb[j].min.z ;    // ascending bb.min.z
    }  

    void dump(const char* msg="Primitives::dump")
    {
          LOG(info) 
              << msg 
              << " treedir " << ( root->treedir ? root->treedir : "-" )
              << " typmsk " << root->get_type_mask_string() 
              << " nprim " << prim.size()
               ; 

         for(unsigned i=0 ; i < prim.size() ; i++)
         {
              unsigned j = zorder[i] ; 
              std::cout << bb[j].desc() << std::endl ; 
         }
    }


};



unsigned NNodeUncoincide::uncoincide_tree_uncyco()
{
    Primitives ps(m_node) ; 
    ps.dump();

    return 0 ; 
}


