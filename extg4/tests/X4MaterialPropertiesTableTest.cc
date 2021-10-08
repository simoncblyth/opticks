

#include "OPTICKS_LOG.hh"

#include "X4OpNoviceMaterials.hh"
#include "X4MaterialPropertiesTable.hh"

#include "G4Version.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"


void test_MaterialPropertyNames(const G4MaterialPropertiesTable* mpt)
{
    LOG(info); 
    typedef std::vector<G4String> VS ; 
    VS nn = mpt->GetMaterialPropertyNames()  ;

    std::vector<G4String> names(nn); 
    names.push_back("NonExistingKey"); 

    for(unsigned i=0 ; i < names.size() ; i++) 
    {
        const G4String& name = names[i] ; 

        int idx1 = X4MaterialPropertiesTable::GetPropertyIndex(mpt, name.c_str() ); 
   
#if G4VERSION_NUMBER < 1100
        G4int idx0 = mpt->GetPropertyIndex(name) ; 
#else
        G4int idx0 = -1100 ; 
#endif
        std::cout 
            << " i " << std::setw(3) << i 
            << " name " << std::setw(50) << name 
            << " idx0(G4MaterialPropertiesTable::GetPropertyIndex) " << std::setw(3) << idx0 
            << " idx1(X4MaterialPropertiesTable::GetPropertyIndex) " << std::setw(3) << idx1
            << std::endl 
            ;
    }
}



void test_MaterialConstPropertyNames(const G4MaterialPropertiesTable* mpt)
{
    LOG(info); 
    typedef std::vector<G4String> VS ; 
    VS nn = mpt->GetMaterialConstPropertyNames()  ;

    std::vector<G4String> names(nn); 
    names.push_back("NonExistingKey"); 

    for(unsigned i=0 ; i < names.size() ; i++) 
    {
        const G4String& name = names[i] ; 

        int idx1 = X4MaterialPropertiesTable::GetConstPropertyIndex(mpt, name.c_str() ); 
   
#if G4VERSION_NUMBER < 1100
        G4int idx0 = mpt->GetConstPropertyIndex(name) ; 
#else
        G4int idx0 = mpt->GetConstPropertyIndex(name) ; 
        //G4int idx0 = -1100 ; 
#endif
        std::cout 
            << " i " << std::setw(3) << i 
            << " name " << std::setw(50) << name 
            << " idx0(G4MaterialPropertiesTable::GetConstPropertyIndex) " << std::setw(3) << idx0 
            << " idx1(X4MaterialPropertiesTable::GetConstPropertyIndex) " << std::setw(3) << idx1
            << std::endl 
            ;
    }
}






int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    X4OpNoviceMaterials opnov ; 
    G4Material* water = opnov.water ;
    G4MaterialPropertiesTable* mpt = water->GetMaterialPropertiesTable() ; 

    test_MaterialPropertyNames(mpt); 
    test_MaterialConstPropertyNames(mpt); 

    bool all = true ; 
    X4MaterialPropertiesTable::Dump(mpt, all); 
    
    return 0 ; 
}
