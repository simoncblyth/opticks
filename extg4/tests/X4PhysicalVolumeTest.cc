#include "GMaterialLib.hh"

#include "X4PhysicalVolume.hh"
#include "X4MaterialTable.hh"
#include "OpNoviceDetectorConstruction.hh"
#include "Opticks.hh"
#include "SDirect.hh"
#include "OPTICKS_LOG.hh"


G4VPhysicalVolume* construct()
{
    G4VPhysicalVolume* top = NULL ; 
    OpNoviceDetectorConstruction ondc ; 

    // redirect cout and cerr from the Construct
    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {   
       cout_redirect out(coutbuf.rdbuf());
       cerr_redirect err(cerrbuf.rdbuf());
       top = ondc.Construct() ;     
    }   
    std::string _cout = coutbuf.str() ; 
    std::string _cerr = cerrbuf.str() ; 
 
    LOG(trace) << " cout " << _cout ;
    LOG(trace) << " cerr " << _cerr ;
    assert(top);  

    return top ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    // Note that okc.Opticks instanciation is 
    // handled by X4PhysicalVolume ctor in order to
    // set the static geometry key before the instanciation.
   
    // NB cannot grab top via the navigator singleton as not a full Geant4 environment  

    G4VPhysicalVolume* top = construct() ; 

    GGeo* ggeo = X4PhysicalVolume::Convert(top) ;   
    assert(ggeo);  

    Opticks* ok = Opticks::GetOpticks();
    ok->Summary();

    return 0 ; 
}


