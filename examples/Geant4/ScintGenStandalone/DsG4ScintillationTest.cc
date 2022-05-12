
#include "SPath.hh"
#include "SStr.hh"
#include "SOpticksResource.hh"
#include "DsG4Scintillation.h"


G4Material* GetScintillator(const char* name)
{
    const char* idpath = SOpticksResource::IDPath(true);
    std::string ori = SStr::Format("%s_ori", name ); 
    const char* dir = SPath::Resolve(idpath, "GScintillatorLib", ori.c_str(), NOOP );
    std::cout << " dir " << dir << std::endl ; 
    return nullptr ; 
}


int main()
{
    // need to spring scintillator material into life with FASTCOMPONENT SLOWCOMPONENT 
    // in order to build the integrals  

    G4Material* ls = GetScintillator("LS"); 


    DsG4Scintillation* proc = new DsG4Scintillation ; 
    std::cout << " proc " << proc << std::endl ; 


    return 0 ; 
}
