#include "U4.hh"
#include "SOpticksResource.hh"
#include "G4Material.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
     
    const NP* a = SOpticksResource::IDLoad("GScintillatorLib/LS_ori/RINDEX.npy"); 
    G4MaterialPropertyVector* v = U4::MakeProperty(a) ;  
    G4Material* mat = U4::MakeMaterial(v) ;  

    LOG(info) << "mat " << mat ; 
    G4cout << "mat " << *mat << std::endl ; 

    return 0 ; 
}
