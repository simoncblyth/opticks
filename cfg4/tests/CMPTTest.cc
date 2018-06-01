#include "G4MaterialPropertiesTable.hh"
#include "G4PhysicsOrderedFreeVector.hh"

#include "OPTICKS_LOG.hh"
#include "CMPT.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG__(argc, argv);


    CMPT* mpt = CMPT::MakeDummy() ; 

    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = mpt->GetPropertiesMap() ;

    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)
    {
        G4String pname = it->first ;
        G4MaterialPropertyVector* pvec = it->second ;
        G4MaterialPropertyVector* pvec2 = mpt->GetProperty(pname.c_str()) ;
        assert( pvec == pvec2 ) ;  
 
        LOG(info) << pname << "\n" << *pvec ; 
    }

    return 0 ; 
}


    
 

    return 0 ; 
}
