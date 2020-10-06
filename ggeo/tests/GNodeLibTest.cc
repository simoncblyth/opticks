#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GNodeLib.hh"

/**
GNodeLibTest
=============

See also ana/GNodeLib.py 

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
   
    Opticks ok(argc, argv);
    ok.configure();
 
    GNodeLib* nlib = GNodeLib::Load(&ok); 
    assert(nlib);  

    LOG(info) << "nlib " << nlib ; 
    nlib->Dump("GNodeLibTest"); 

    unsigned num_transforms = nlib->getNumTransforms(); 
    LOG(info) << " num_transforms " << num_transforms ; 

    glm::mat4 tr0 = nlib->getTransform(0); 
    LOG(info) << " tr(0) " << glm::to_string(tr0) ;  

    glm::mat4 tr1 = nlib->getTransform(num_transforms-1); 
    LOG(info) << " tr(N-1) " << glm::to_string(tr1) ;  


    nlib->dumpVolumes(); 



    return 0 ; 
}
