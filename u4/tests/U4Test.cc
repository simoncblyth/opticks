#include "U4.hh"
#include "SOpticksResource.hh"
#include "G4Material.hh"

#include "OPTICKS_LOG.hh"

void test_MakeMaterial()
{
    const NP* a = SOpticksResource::IDLoad("GScintillatorLib/LS_ori/RINDEX.npy"); 
    G4MaterialPropertyVector* v = U4::MakeProperty(a) ;  
    G4Material* mat = U4::MakeMaterial(v) ;  

    LOG(info) << "mat " << mat ; 
    G4cout << "mat " << *mat << std::endl ; 
}

void test_MakeMaterialPropertiesTable()
{
    G4MaterialPropertiesTable* mpt = U4::MakeMaterialPropertiesTable("GScintillatorLib/LS_ori", "FASTCOMPONENT,SLOWCOMPONENT,REEMISSIONPROB", ',' ) ; 
    std::cout << " mpt " << mpt << std::endl ; 
}

void test_MakeScintillator()
{
    G4Material* mat = U4::MakeScintillator(); 
    G4cout << "mat " << *mat << std::endl ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //test_MakeMaterial(); 
    //test_MakeMaterialPropertiesTable(); 
    test_MakeScintillator(); 

     
    return 0 ; 
}
