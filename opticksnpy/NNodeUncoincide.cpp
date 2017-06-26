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

unsigned NNodeUncoincide::uncoincide()
{
    // canonically invoked via nnode::uncoincide from NCSG::import_r

    nnode* a = NULL ; 
    nnode* b = NULL ; 

    unsigned rc = 0 ; 

    if(is_uncoincidable_subtraction(a,b))
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

bool NNodeUncoincide::is_uncoincidable_union(nnode*& a, nnode*& b)
{
    OpticksCSG_t type = m_node->type ; 
    nnode* left = m_node->left ; 
    nnode* right = m_node->right ; 

    if( type == CSG_UNION ) 
    {
        a = left ; 
        b = right ; 
    }


    bool is_uncoincidable  =  a && b && a->type == CSG_CYLINDER && b->type == CSG_CYLINDER ;
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
 
unsigned NNodeUncoincide::uncoincide_union(nnode* a, nnode* b)
{
    // opticks-tbool 143

    std::cout << std::endl << std::endl ; 
    LOG(info) << "NNodeUncoincide::uncoincide_union"
               << " A " << a->tag()
               << " B " << b->tag()
               ;

    a->dump_full("uncoincide_union A");
    b->dump_full("uncoincide_union B");

    nbbox a_ = a->bbox();
    nbbox b_ = b->bbox();


    if(a_.min.z == b_.max.z )
    {
        LOG(info) << " A.bbox.min.z == B.bbox.max.z ( A//B ) " ;  
    }
    else if(a_.max.z == b_.min.z )
    {
        LOG(info) << " A.bbox.max.z == B.bbox.min.z ( B//A ) " ;  
    }

    std::cout << std::endl << std::endl ; 
    assert(0 && "hari-kari") ; 

    return 0 ; 
}


/*

2017-06-26 13:58:43.247 INFO  [1179933] [NNodeUncoincide::uncoincide_union@143] NNodeUncoincide::uncoincide_union A [ 1:cy] B [ 2:cy]
uncoincide_union A [ 1:cy] PRIM  v:0 bb  mi  (-650.00 -650.00  -23.50)  mx  ( 650.00  650.00   23.50) 
2017-06-26 13:58:43.247 INFO  [1179933] [NNodeDump::dump_prim@93] uncoincide_union A nprim 1
        cy label no-label center {    0.0000    0.0000    0.0000} radius 650.0000 z1 -23.5000 z2 23.5000 gseedcenter {    0.0000    0.0000    0.0000} gtransform 1
uncoincide_union A
 NO transform 
uncoincide_union A
     gtr.t  1.000   0.000   0.000   0.000 
            0.000   1.000   0.000   0.000 
            0.000   0.000   1.000   0.000 
            0.000   0.000   0.000   1.000 

uncoincide_union B [ 2:cy] PRIM  v:0 bb  mi  ( -31.50  -31.50  -58.50)  mx  (  31.50   31.50  -23.50) 
2017-06-26 13:58:43.247 INFO  [1179933] [NNodeDump::dump_prim@93] uncoincide_union B nprim 1
        cy label no-label center {    0.0000    0.0000    0.0000} radius 31.5000 z1 -17.5000 z2 17.5000 gseedcenter {    0.0000    0.0000  -41.0000} gtransform 1
uncoincide_union B
      tr.t  1.000   0.000   0.000   0.000 
            0.000   1.000   0.000   0.000 
            0.000   0.000   1.000   0.000 
            0.000   0.000 -41.000   1.000 

uncoincide_union B
     gtr.t  1.000   0.000   0.000   0.000 
            0.000   1.000   0.000   0.000 
            0.000   0.000   1.000   0.000 
            0.000   0.000 -41.000   1.000 




*/





