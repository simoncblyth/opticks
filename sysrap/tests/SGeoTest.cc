
#include <iostream>
#include "OPTICKS_LOG.hh"
#include "SGeo.hh"

/*
struct SGeoTest : public SGeo
{
    unsigned     getNumMeshes() const { return 0 ; } 
    const char*  getMeshName(unsigned midx) const { return nullptr ; }
    int          getMeshIndexWithName(const char* name, bool startswith) const { return 0 ; }
    int          getFrame(sframe& fr, int ins_idx ) const { return 0 ;  }  
    std::string  descBase() const { return "" ; }
}; 
*/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    std::cout << "SGeo::DefaultDir " << SGeo::DefaultDir() << std::endl ; 
 
    return 0 ; 
}

