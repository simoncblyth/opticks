// TEST=NPriTest om-t

#include "NPY.hpp"
#include "NPri.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
   
    const char* path = "/usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/c250d41454fba7cb19f3b83815b132c2/1/primaries.npy" ; 

    NPY<float>* p = NPY<float>::load(path) ; 
    if(!p) return 0 ; 

    NPri* pr = new NPri(p); 
    LOG(info) << std::endl << pr->desc(0); 

    int pdgcode = pr->getPDGCode(0) ; 
    LOG(info) << " pdgcode " << pdgcode ; 


    return 0 ; 
}
