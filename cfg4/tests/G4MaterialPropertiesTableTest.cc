#include "G4MaterialPropertiesTable.hh"
#include "OPTICKS_LOG.hh"



void addDummyProperty(G4MaterialPropertiesTable* mpt, const char* lkey, unsigned nval)
{
    G4double* ddom = new G4double[nval] ;  
    G4double* dval = new G4double[nval] ;  
    for(unsigned int j=0 ; j < nval ; j++)
    {
        ddom[nval-1-j] = j*100. ; 
        dval[nval-1-j] = j*1000. ; 
    } 
    G4MaterialPropertyVector* mpv = mpt->AddProperty(lkey, ddom, dval, nval); 
    mpv->SetSpline(false); 

    delete [] ddom ;
    delete [] dval ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG__(argc, argv);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable(); 

    LOG(info) << " mpt " << mpt ;  

    addDummyProperty( mpt, "A", 5 ); 
    addDummyProperty( mpt, "B", 10 ); 

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



