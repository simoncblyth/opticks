
#include "SOpticksResource.hh"
#include "SPath.hh"
#include "SProp.hh"
#include "NP.hh"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const NP* propcom = SProp::MockupCombination("$IDPath/GScintillatorLib/LS_ori/RINDEX.npy"); 
    if(propcom == nullptr) return 0 ; 
    propcom->dump(); 


    LOG(none)    << " LOG(none) " ; 
    LOG(fatal)   << " LOG(fatal) " ; 
    LOG(error)   << " LOG(error) " ; 
    LOG(warning) << " LOG(warning) " ; 
    LOG(info)    << " LOG(info) " ; 
    LOG(debug)   << " LOG(debug) " ; 
    LOG(verbose) << " LOG(verbose) " ; 



    return 0 ; 
}
