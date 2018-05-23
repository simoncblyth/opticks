/*

lldb NCSGIntersectTest  /tmp/blyth/opticks/tboolean-zsphere1--

*/

#include "NPY_LOG.hh"
#include "PLOG.hh"
#include "NCSGList.hpp"
#include "NCSG.hpp"
#include "NCSGIntersect.hpp"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    const char* csgpath = argc > 1 ? argv[1] : NULL ; 
    unsigned verbosity = 0 ; 

    if(csgpath == NULL)
    {
        LOG(warning) << "Expecting 1st argument csgpath directory containing NCSG trees" ; 
        return 0 ;
    } 
   
    NCSGList* ls = NCSGList::Load(csgpath, verbosity );    
    if( ls == NULL )
    {
        LOG(warning) << "FAILED to load NCSG trees from " << csgpath  ; 
        return 0 ;
    }

    ls->dumpDesc();


    unsigned num_tree = ls->getNumTrees();
    NCSGIntersect* csgi = new NCSGIntersect[num_tree] ;
    for(unsigned i=0 ; i < num_tree ; i++ ) csgi[i].init(ls->getTree(i)) ;

    glm::vec4 post0( 1001,   0,0,0.1) ;
    glm::vec4 post1(-1001,   0,0,0.2) ;
    glm::vec4 post2(    0,1001,0,0.3) ;
    glm::vec4 post3(    0,1002,0,0.4) ;

    for(unsigned i=0 ; i < num_tree ; i++ )
    {
        csgi[i].add(0, post0);
        csgi[i].add(0, post1);
        csgi[i].add(0, post2);
        csgi[i].add(0, post3);
    }

    for(unsigned p_=0 ; p_ < 5 ; p_++ )
    {
        unsigned p = 0 ; 
        for(unsigned i=0 ; i < num_tree ; i++ )
        {
            std::cout << csgi[i].desc_dist(p) ; 
        }
        std::cout << std::endl ; 
    } 

    delete [] csgi ; 




    return 0 ; 
}
