// TEST=G4MaterialPropertiesTableTest om-t

#include <iomanip>
#include "G4MaterialPropertiesTable.hh"

#include "CMPT.hh"
#include "OPTICKS_LOG.hh"

void test_MPT(G4MaterialPropertiesTable* mpt)
{
    typedef G4MaterialPropertyVector MPV ; 
    G4bool warning ; 
    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    LOG(info) << "pns:" << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {
        const std::string& pn = pns[i]; 
        G4int idx = mpt->GetPropertyIndex(pn, warning=true); 
        assert( idx > -1 );  
        MPV* mpv = mpt->GetProperty(idx, warning=false ); 

        std::cout 
            << " pn : " 
            << std::setw(30) << pn 
            << " idx : " 
            << std::setw(5) << idx 
            << " mpv : "
            << std::setw(16) << mpv 
            << std::endl
            ;  
    } 
}


void test_MPTConst(G4MaterialPropertiesTable* mpt)
{
    G4bool warning ; 
    std::vector<G4String> pns = mpt->GetMaterialConstPropertyNames() ;
    LOG(info) << "pns:" << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {
        const std::string& pn = pns[i]; 

        G4bool exists = mpt->ConstPropertyExists( pn.c_str() ) ; 

        G4int idx = mpt->GetConstPropertyIndex(pn, warning=true); 
        assert( idx > -1 );  
        G4double pval = exists ? mpt->GetConstProperty(idx) : 0. ; 

        std::cout 
            << " pn : " 
            << std::setw(30) << pn 
            << " exists : " 
            << std::setw(3) << exists 

            << " idx : " 
            << std::setw(5) << idx 
            << " pval : "
            << std::setw(16) << pval 
            << std::endl
            ;  
    } 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG__(argc, argv);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable(); 

    LOG(info) << " mpt " << mpt ;  

    CMPT::AddDummyProperty( mpt, "A", 5 ); 
    CMPT::AddDummyProperty( mpt, "B", 10 ); 
    CMPT::AddDummyProperty( mpt, "C", 10 ); 

    CMPT::AddConstProperty( mpt, "AA", 5 ); 
    CMPT::AddConstProperty( mpt, "BB", 10 ); 
    CMPT::AddConstProperty( mpt, "CC", 10 ); 


    CMPT::Dump( mpt ) ;    

    test_MPT(mpt); 
    test_MPTConst(mpt); 


    return 0 ; 
}



