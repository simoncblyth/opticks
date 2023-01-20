/**
U4PMTAccessorTest.cc
=====================

See also j/PMTFastSim/tests/PMTAccessorTest.cc that is 
similar to this and has faster dev cycle 


**/

#include "OPTICKS_LOG.hh"
#include "sdirect.h"

#ifdef WITH_PMTFASTSIM
#include "DetectorConstruction.hh"
#include "JPMT.h"
#include "PMTAccessor.h"

#include <CLHEP/Units/SystemOfUnits.h>

#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#ifdef WITH_PMTFASTSIM

    DetectorConstruction* dc = nullptr ; 

    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {   
        sdirect::cout_ out_(coutbuf.rdbuf());
        sdirect::cerr_ err_(cerrbuf.rdbuf());
    
        dc = new DetectorConstruction ; 
        // boot G4Material from files read under $JUNOTOP  with verbosity silenced
    }    
    std::string out = coutbuf.str(); 
    std::string err = cerrbuf.str(); 
    bool VERBOSE = getenv("VERBOSE") != nullptr ;  
    std::cout << sdirect::OutputMessage("DetectorConstruction" , out, err, VERBOSE );  


    const PMTSimParamData* data = PMTAccessor::LoadPMTSimParamData() ; 
    LOG(info) << " data " << *data ; 

    int pmtcat = kPMT_Hamamatsu ; 
    double energy = 5.*CLHEP::eV ; 
    double v = data->get_pmtcat_prop( pmtcat, "ARC_RINDEX" , energy ); 

    LOG(info) << " energy " << energy << " ARC_RINDEX " << v ;  


    const PMTAccessor* acc = PMTAccessor::Create(data) ; 
    LOG(info) << " acc " << acc->desc() ; 


#else
    LOG(fatal) << "not WITH_PMTFASTSIM : nothing to do " ;     
#endif
    return 0 ; 
}

