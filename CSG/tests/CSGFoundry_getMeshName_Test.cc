#include "OPTICKS_LOG.hh"
#include "SSim.hh"
#include "CSGFoundry.h"

const char* FOLD = getenv("FOLD") ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SSim::Create(); 
    const CSGFoundry* fd = CSGFoundry::Load();
    std::cout << " fd.brief " << fd->brief() << std::endl ;
    std::cout << " fd.desc  " << fd->desc() << std::endl ;

    int num_mn = fd->getNumMeshName() ; 
    std::cout << " num_mn " << num_mn << std::endl ;

    for(int i=0 ; i < num_mn ; i++)
    {
        const char* mn = fd->getMeshName(i) ; 
        int fmi_midx = fd->findMeshIndex(mn);
        int gmi0_midx = fd->getMeshIndexWithName(mn, false);  
        int gmi1_midx = fd->getMeshIndexWithName(mn, true );  

        std::cout 
            << " i " << std::setw(4) << i 
            << " fmi " << std::setw(4) << fmi_midx
            <<  ( fmi_midx == i ? ' ' : 'x' )
            << " gmi0 " << std::setw(4) << gmi0_midx
            <<  ( gmi0_midx == i ? ' ' : 'x' )
            << " gmi1 "  << std::setw(4) << gmi1_midx
            <<  ( gmi1_midx == i ? ' ' : 'x' )
            << " mn " << ( mn ? mn : "-" )
            << std::endl
            ; 
    }

    return 0 ; 
}

