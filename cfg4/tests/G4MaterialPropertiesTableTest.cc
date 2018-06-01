#include "G4MaterialPropertiesTable.hh"

#include "CMPT.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG__(argc, argv);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable(); 

    LOG(info) << " mpt " << mpt ;  

    CMPT::AddDummyProperty( mpt, "A", 5 ); 
    CMPT::AddDummyProperty( mpt, "B", 10 ); 

    CMPT::Dump( mpt ) ;    
  

    return 0 ; 
}



