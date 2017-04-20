/**
Tests individual trees::

    NCSGLoadTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-/1

**/

#include <iostream>

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


struct NCSGLoadTest 
{
    NCSGLoadTest( const char* treedir, int index );
    NCSG csg ; 
};
  

NCSGLoadTest::NCSGLoadTest(const char* treedir, int index ) 
    : 
    csg(treedir, index)
{
    csg.load();
    csg.import();
    csg.dump("NCSGLoadTest");

    char* scan_ = getenv("SCAN") ;
    std::string scan = scan_ ? scan_ : "10,10,100,0" ;  
    glm::ivec4 vs = givec4(scan) ;

    std::cout  << " vscan " << vs << std::endl ; 


    nnode* root = csg.getRoot();

    glm::vec3 origin(   0.f, 0.f, 127.f );
    glm::vec3 direction(0.f, 0.f,   1.f );
    glm::vec3 range(   -2.f, 2.f, 0.001f );

    nnode::Scan(*root, origin, direction, range );

}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    NCSGLoadTest tst( argc > 1 ? argv[1] : "$TMP/csg_py/1" , 0 );

    return 0 ; 
}


