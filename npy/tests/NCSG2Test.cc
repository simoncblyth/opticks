// TEST=NCSG2Test om-t

/**
**/

#include <iostream>

#include "BFile.hh"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NBBox.hpp"
#include "NGLMExt.hpp"
#include "NQuad.hpp"
#include "GLMFormat.hpp"

#include "OPTICKS_LOG.hh"

#include "BOpticksKey.hh"
#include "BOpticksResource.hh"


void test_analytic_centering( NCSG* csg )
{
    nbbox bb0 = csg->bbox_analytic() ; 
    nvec4 ce0 = bb0.center_extent() ;
    bool centered0 = ce0.x == 0.f && ce0.y == 0.f && ce0.z == 0.f ; 

    LOG(info) 
        << " bb0 " << bb0.description()
        << " ce0 " << ce0.desc()
        << ( centered0 ? " CENTERED " : " NOT-CENTERED " )
        ; 

    if(!centered0  )
    {
        nnode* root = csg->getRoot(); 
        LOG(info) << " root->transform " << *root->transform ;  
        root->placement = nmat4triple::make_translate( -ce0.x, -ce0.y, -ce0.z );  
        root->update_gtransforms(); 
    } 
    // TODO: nnode::apply_centering() for this
   

    nbbox bb1 = csg->bbox_analytic();  // <-- global frame bbox, even for single primitive 
    nvec4 ce1 = bb1.center_extent() ;
    bool centered1 = ce1.x == 0.f && ce1.y == 0.f && ce1.z == 0.f ; 

    LOG(info) 
        << " bb1 " << bb1.description()
        << " ce1 " << ce1.desc()
        << ( centered1 ? " CENTERED " : " NOT-CENTERED " )
        ; 

    assert( centered1 ); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* lvid = argc > 1 ? argv[1] : "17" ; 
   
    // TODO: streamline this stuff at brap level  
    BOpticksKey::SetKey(NULL) ; 
    bool testgeo(false) ; 
    BOpticksResource okr(testgeo) ;  // no Opticks at this level 
    if( !okr.hasKey() ) return 0 ;  
    okr.setupViaKey(); 

    const char* path = okr.makeIdPathPath("GMeshLibNCSG", lvid );  
    LOG(info) << "[" << path  << "]" ;  

    NCSG* csg = NCSG::Load(path); 
    if(!csg) return 0 ; 

    test_analytic_centering(csg); 

    return 0 ; 
}


